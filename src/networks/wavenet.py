import math
import os
import os.path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=256,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                        num_channels=residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input, dilation_func):
        x = self.start_conv(input)
        #print(f"After start_conv: {x.size()}")  # Track after initial convolution

        skip = 0 # Adjusted initialization
       
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)
            #print(f"After dilation_func (Layer {i}): {residual.size()}")  # Track after dilation

            filter = self.filter_convs[i](residual)
            filter = F.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = F.sigmoid(gate)
            x = filter * gate
            #print(f"After gated activations (Layer {i}): {x.size()}")  # Track after gated activations

            s = x
            if x.size(2) != 1:
                s = dilate(x, 1, init_dilation=dilation)
                #print(f"After dilate (Layer {i}): {s.size()}")  # Track after dilation

            s = self.skip_convs[i](s)
            #print(f"After skip_convs (Layer {i}): {s.size()}")  # Track after skip convolutions
            # Align sequence lengths of s and skip before addition
            if skip is not 0:
                if s.size(2) != skip.size(2):
                    # Determine the max length
                    max_len = max(s.size(2), skip.size(2))
                    # Pad the shorter tensor to match this length
                    if s.size(2) < max_len:
                        s = F.pad(s, (0, max_len - s.size(2)))
                    else:
                        skip = F.pad(skip, (0, max_len - skip.size(2)))
                    #print(f"After padding alignment (Layer {i}) skip: {skip.size()}, s: {s.size()}")
                
            skip = s + skip
            #print(f"After skip addition (Layer {i}): {skip.size()}")  # Track after skip addition

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]
            #print(f"After residual convs (Layer {i}): {x.size()}")  # Track after residual convolutions

        x = F.relu(skip)
        #print(f"After ReLU on skip: {x.size()}")  # Track after ReLU on skip

        x = F.relu(self.end_conv_1(x))
        #print(f"After end_conv_1: {x.size()}")  # Track after end_conv_1

        x = self.end_conv_2(x)
        #print(f"After end_conv_2: {x.size()}")  # Track after end_conv_2

        return x

    # def wavenet(self, input, dilation_func):

    #     x = self.start_conv(input)
    #     skip = 0
    #     #skip = torch.zeros([input.size(0), self.skip_channels, [16, 256, 9]], dtype=self.dtype)
       
    #     print("skip: 0")
    #     # WaveNet layers
    #     for i in range(self.blocks * self.layers):

    #         #            |----------------------------------------|     *residual*
    #         #            |                                        |
    #         #            |    |-- conv -- tanh --|                |
    #         # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
    #         #                 |-- conv -- sigm --|     |
    #         #                                         1x1
    #         #                                          |
    #         # ---------------------------------------> + ------------->	*skip*

    #         (dilation, init_dilation) = self.dilations[i]

    #         residual = dilation_func(x, dilation, init_dilation, i)
    #         print(f"x size {x.size()}")
    #         # dilated convolution
    #         filter = self.filter_convs[i](residual)
    #         filter = F.tanh(filter)
    #         gate = self.gate_convs[i](residual)
    #         gate = F.sigmoid(gate)
    #         x = filter * gate
    #         print(f"x size filtter gate {x.size()}")
    #         # parametrized skip connection
    #         s = x
            
    #         print(f"s is: {s.size()}")
    #         if x.size(2) != 1:
    #              s = dilate(x, 1, init_dilation=dilation)
    #              print(f"s dilated is: {s.size()}")
    #         s = self.skip_convs[i](s)
    #         print(f"s skip_convs is: {s.size()}")
    #         try:
    #             skip = skip[:, :, -s.size(2):]
    #         except:
    #             skip = 0
    #         if skip is not 0:
    #             print(f"skip is {skip.size()} before add")
    #         skip = s + skip
    #         print(f"skip is {skip.size()} after add")
    #         x = self.residual_convs[i](x)
    #         x = x + residual[:, :, (self.kernel_size - 1):]
    #         print(f"x size end {x.size()}")

    #     x = F.relu(skip)
    #     x = F.relu(self.end_conv_1(x))
    #     x = self.end_conv_2(x)

    #     return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        x = x.transpose(1, 2).contiguous()
        x = x.view(n * l, c)
        return x

    def generate(self,
                 num_samples,
                 first_samples=None,
                 temperature=1.):
        self.eval()
        if first_samples is None:
            first_samples = self.dtype(1).zero_()
        generated = Variable(first_samples, volatile=True)

        num_pad = self.receptive_field - generated.size(0)
        if num_pad > 0:
            generated = constant_pad_1d(generated, self.scope, pad_start=True)
            print("pad zero")

        for i in range(num_samples):
            input = Variable(torch.FloatTensor(1, self.classes, self.receptive_field).zero_())
            input = input.scatter_(1, generated[-self.receptive_field:].view(1, -1, self.receptive_field), 1.)

            x = self.wavenet(input,
                             dilation_func=self.wavenet_dilate)[:, :, -1].squeeze()

            if temperature > 0:
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = Variable(torch.LongTensor([x]))#np.array([x])
            else:
                x = torch.max(x, 0)[1].float()

            generated = torch.cat((generated, x), 0)

        generated = (generated / self.classes) * 2. - 1
        mu_gen = mu_law_expansion(generated, self.classes)

        self.train()
        return mu_gen

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100):
        self.eval()
        if first_samples is None:
            first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        first_samples = Variable(first_samples)

        # reset queues
        for queue in self.dilated_queues:
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        input = Variable(torch.FloatTensor(1, self.classes, 1).zero_())
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate)
            input.zero_()
            input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = np.array([])
        regularizer = torch.pow(Variable(torch.arange(self.classes)) - self.classes / 2., 2)
        regularizer = regularizer.squeeze() * regularize
        tic = time.time()
        for i in range(num_samples):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate).squeeze()

            x -= regularizer

            if temperature > 0:
                # sample from softmax distribution
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = np.array([x])
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            o = (x / self.classes) * 2. - 1
            generated = np.append(generated, o)

            # set new input
            x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.classes)
        return mu_gen


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.dtype = type
        for q in self.dilated_queues:
            q.dtype = self.dtype
        super().cpu()


def load_latest_model_from(location, use_cuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)

    if use_cuda:
        model = torch.load(newest_file)
    else:
        model = load_to_cpu(newest_file)

    return model


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model


# def dilate(x, dilation, init_dilation=1, pad_start=True):
#     """
#     :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
#     :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
#     :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
#     :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
#     """

#     [n, c, l] = x.size()
#     dilation_factor = dilation / init_dilation
#     if dilation_factor == 1:
#         return x

#     # zero padding for reshaping
#     new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
#     if new_l != l:
#         l = new_l
#         x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

#     l_old = int(round(l / dilation_factor))
#     n_old = int(round(n * dilation_factor))
#     l = math.ceil(l * init_dilation / dilation)
#     n = math.ceil(n * dilation / init_dilation)

#     # reshape according to dilation
#     x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
#     x = x.view(c, l, n)
#     x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

#     return x

# def dilate(x, dilation, init_dilation=1, pad_start=True):
#     [n, c, l] = x.size()

#     dilation_factor = dilation / init_dilation
#     if dilation_factor == 1:
#         return x

#     # Calculate the new length after dilation
#     new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
#     if new_l != l:
#         x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

#     # Reshape to apply dilation, preserving batch size
#     x = x.view(n, c, -1, dilation)
#     x = x.permute(0, 3, 1, 2)  # Permute to bring dilation next to batch dimension
#     x = x.contiguous().view(n, c, -1)

#     return x

# def dilate(x, dilation, init_dilation=1, pad_start=True):
#     [n, c, l] = x.size()

#     dilation_factor = dilation / init_dilation
#     if dilation_factor == 1:
#         return x

#     # Calculate the new length after dilation
#     new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
#     if new_l != l:
#         x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

#     # Check if reshaping is valid
#     total_elements = n * c * new_l
#     if total_elements % (n * c * dilation) != 0:
#         raise ValueError("Invalid reshaping dimensions for dilation")

#     # Reshape to apply dilation, preserving batch size
#     new_third_dim = total_elements // (n * c * dilation)
#     x = x.view(n, c, new_third_dim, dilation)
#     x = x.permute(0, 3, 1, 2)  # Permute to bring dilation next to batch dimension
#     x = x.contiguous().view(n, c, -1)

#     return x

# def dilate(x, dilation, init_dilation=1, pad_start=True):
#     [n, c, l] = x.size()

#     # Calculate the required padding and the new length after dilation
#     required_padding = (dilation - (l % dilation)) % dilation
#     new_l = l + required_padding

#     if required_padding > 0:
#         x = F.pad(x, (0, required_padding), mode='constant', value=0)

#     # Reshape the tensor to apply dilation
#     x = x.view(n, c, new_l // dilation, dilation)
#     x = x.permute(0, 2, 1, 3)  # Change to (n, new_l // dilation, c, dilation)
#     x = x.contiguous().view(n, new_l // dilation, -1)

#     return x

# def dilate(x, dilation, init_dilation=1, pad_start=True):
#     [n, c, l] = x.size()

#     # Calculate the required padding and the new length after dilation
#     required_padding = (dilation - (l % dilation)) % dilation
#     new_l = l + required_padding

#     if required_padding > 0:
#         x = F.pad(x, (0, required_padding), mode='constant', value=0)

#     # Apply dilation while preserving the order of dimensions
#     x = x.unfold(dimension=2, size=dilation, step=dilation)

#     return x

def dilate(x, dilation, init_dilation=1, pad_start=True):
    [n, c, l] = x.size()

    # Calculate the required padding and the new length after dilation
    required_padding = (dilation - (l % dilation)) % dilation
    new_l = l + required_padding

    if required_padding > 0:
        x = F.pad(x, (0, required_padding), mode='constant', value=0)

    # Apply dilation while preserving the order of dimensions
    x = x.unfold(dimension=2, size=dilation, step=dilation)
    x = x.contiguous().view(n, c, -1)  # Reshape to remove the extra dimension

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(Function):
    # def __init__(self, target_size, dimension=0, value=0, pad_start=False):
    #     super(ConstantPad1d, self).__init__()
    #     self.target_size = target_size
    #     self.dimension = dimension
    #     self.value = value
    #     self.pad_start = pad_start

    @staticmethod
    def forward(ctx, input, target_size, dimension=0, value=0, pad_start=False):
        ctx.save_for_backward(input)
        ctx.dimension = dimension
        ctx.pad_start = pad_start
        ctx.num_pad = target_size - input.size(dimension)

        num_pad = target_size - input.size(dimension)
        assert num_pad >= 0, 'target size has to be greater than input size'

        

        size = list(input.size())
        size[dimension] = target_size
        output = input.new(*tuple(size)).fill_(value)
        c_output = output

        # crop output
        if pad_start:
            c_output = c_output.narrow(dimension, num_pad, c_output.size(dimension) - num_pad)
        else:
            c_output = c_output.narrow(dimension, 0, c_output.size(dimension) - num_pad)

        c_output.copy_(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        dimension = ctx.dimension
        pad_start = ctx.pad_start
        num_pad = ctx.num_pad
        input_size = input.size()
        grad_input = grad_output.new(*input_size).zero_()
        cg_output = grad_output

        # crop grad_output
        if pad_start:
            cg_output = cg_output.narrow(dimension, num_pad, cg_output.size(dimension) - num_pad)
        else:
            cg_output = cg_output.narrow(dimension, 0, cg_output.size(dimension) - num_pad)

        grad_input.copy_(cg_output)
        return grad_input, None, None, None, None, None


# def constant_pad_1d(input,
#                     target_size,
#                     dimension=0,
#                     value=0,
#                     pad_start=False):
#     return ConstantPad1d(target_size, dimension, value, pad_start)(input)

def constant_pad_1d(input, target_size, dimension=0, value=0, pad_start=False):
    return ConstantPad1d.apply(input, target_size, dimension, value, pad_start)