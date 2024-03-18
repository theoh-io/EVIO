import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal

# class SubNet(nn.Module):
#     '''
#     The subnetwork that is used in LMF for video and audio in the pre-fusion stage
#     '''

#     def __init__(self, in_size, hidden_size, dropout):
#         '''
#         Args:
#             in_size: input dimension
#             hidden_size: hidden layer dimension
#             dropout: dropout probability
#         Output:
#             (return value in forward) a tensor of shape (batch_size, hidden_size)
#         '''
#         super(SubNet, self).__init__()
#         self.norm = nn.BatchNorm1d(in_size)
#         self.drop = nn.Dropout(p=dropout)
#         self.linear_1 = nn.Linear(in_size, hidden_size)
#         self.linear_2 = nn.Linear(hidden_size, hidden_size)
#         self.linear_3 = nn.Linear(hidden_size, hidden_size)

#     def forward(self, x):
#         '''
#         Args:
#             x: tensor of shape (batch_size, in_size)
#         '''
#         normed = self.norm(x)
#         dropped = self.drop(normed)
#         y_1 = F.relu(self.linear_1(dropped))
#         y_2 = F.relu(self.linear_2(y_1))
#         y_3 = F.relu(self.linear_3(y_2))

#         return y_3


# class TextSubNet(nn.Module):
#     '''
#     The LSTM-based subnetwork that is used in LMF for text
#     '''

#     def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
#         '''
#         Args:
#             in_size: input dimension
#             hidden_size: hidden layer dimension
#             num_layers: specify the number of layers of LSTMs.
#             dropout: dropout probability
#             bidirectional: specify usage of bidirectional LSTM
#         Output:
#             (return value in forward) a tensor of shape (batch_size, out_size)
#         '''
#         super(TextSubNet, self).__init__()
#         self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_1 = nn.Linear(hidden_size, out_size)

#     def forward(self, x):
#         '''
#         Args:
#             x: tensor of shape (batch_size, sequence_len, in_size)
#         '''
#         _, final_states = self.rnn(x)
#         h = self.dropout(final_states[0].squeeze())
#         y_1 = self.linear_1(h)
#         return y_1

from fightingcv_attention.attention.ExternalAttention import ExternalAttention

class EAF(nn.Module):
    def __init__(self, input_dims, output_dim, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (img_dim, ev_dim, imu_dim)
            /!\ to be reviewed: dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3 /!\ why those values
        '''
        super(EAF, self).__init__()

        #input=torch.randn(50,49,512)
        hidden_dim=8
        self.img_in = input_dims[0]
        self.ev_in = input_dims[1]
        self.imu_in = input_dims[2]
        input_dim=self.img_in+self.ev_in+self.imu_in
        print(f"input dim {input_dim}")
        self.ea = ExternalAttention(d_model=input_dim,S=hidden_dim)

    def forward(self, img_x, ev_x, imu_x):
        #input= torch.cat((img_x, ev_x, imu_x), dim=1)
        input=torch.randn(50,49,512)
        print(input.shape)
        output=self.ea(input)
        return output

class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, dropouts, output_dim, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (img_dim, ev_dim, imu_dim)
            /!\ to be reviewed: dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3 /!\ why those values
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of img, event and imu
        self.img_in = input_dims[0]
        self.ev_in = input_dims[1]
        self.imu_in = input_dims[2]

        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.img_prob = dropouts[0]
        self.ev_prob = dropouts[1]
        self.imu_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.img_factor = Parameter(torch.Tensor(self.rank, self.img_in + 1, self.output_dim))
        self.ev_factor = Parameter(torch.Tensor(self.rank, self.ev_in + 1, self.output_dim))
        self.imu_factor = Parameter(torch.Tensor(self.rank, self.imu_in + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal(self.img_factor)
        xavier_normal(self.ev_factor)
        xavier_normal(self.imu_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, features):
        '''
        Args:
            img_x: tensor of shape (batch_size, img_in)
            ev_x: tensor of shape (batch_size, ev_in)
            imu_x: tensor of shape (batch_size, sequence_len, imu_in)
        '''
        img_x = features[:,:self.img_in]
        print(f"img shape {img_x.size()}")
        ev_x= features[:,self.img_in:(self.img_in+self.ev_in)]
        print(f"event shape {ev_x.size()}")
        imu_x=features[:,(self.img_in+self.ev_in):]
        print(f"imu shape {imu_x.size()}")
        batch_size = img_x.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if img_x.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _img_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), img_x), dim=1)
        _ev_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), ev_x), dim=1)
        _imu_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), imu_x), dim=1)

        fusion_img = torch.matmul(_img_h, self.img_factor)
        fusion_ev = torch.matmul(_ev_h, self.ev_factor)
        fusion_imu = torch.matmul(_imu_h, self.imu_factor)
        fusion_zy = fusion_img * fusion_ev * fusion_imu

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output
    
class Concat(nn.Module):
    '''
    Simplest form of Multimodal Fusion
    '''

    def __init__(self, input_dim, output_dim=256, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (feat_img_dim, feat_ev_dim,feat_ imu_dim)
            output_dim - int, specifying the size of output
        Output:
            (return value in forward) a scalar value between -3 and 3 /!\ why those values
        '''
        super(Concat, self).__init__()

        # dimensions are specified in the order of img, event and imu
        #/!\ feature dimension in latent space !
        print(f"fusion input dim: {input_dim}")
        self.img_in = input_dim[0]
        self.ev_in = input_dim[1]
        self.imu_in = input_dim[2]
        self.use_softmax = use_softmax

        # print(f"img dim {input_dims[0]}, sum {self.img_in}")
        input_dim = self.img_in + self.ev_in + self.imu_in
        print(f"Sum input dim {input_dim}")
        self.output_dim = output_dim


        self.module = nn.Sequential(
                    nn.Linear(input_dim, 128),  
                    nn.ReLU(),
                    nn.Linear(128, self.output_dim))    

    def forward(self, features):
        '''
        Args:
        features vector contains
            img_x: tensor of shape (batch_size, img_in)
            ev_x: tensor of shape (batch_size, ev_in)
            imu_x: tensor of shape (batch_size, sequence_len, imu_in)
        '''
        print(f"(fusion forward) dim of concatenated input: {features.size()}")
        output= self.module(features)
        if self.use_softmax:
            output = F.softmax(output)
        return output