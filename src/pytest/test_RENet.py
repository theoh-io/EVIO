import unittest
from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.utils import util
from collections import OrderedDict

import random
import numpy as np

# Test to create the dataloader for UZH

class TestRENet(unittest.TestCase):

    def setUp(self):
        self.setUp_dataloader()
        self.setUp_RENet()

    def setUp_dataloader(self):
        print("setup")
        self.config = ConfigParser().get_config()
        self.dataloader = CustomDatasetDataLoader(self.config, is_for="train")
        self.dataloader = self.dataloader.load_data()
        # print dimensions of input and target
        self.data_size = len(self.dataloader)
        print('#training images = %d' % self.data_size)

    def old_setUp_RENet(self):
        from src.networks.zz_rgbevnet import REFusion
        self.rgb_ev_module=REFusion()
        print(self.rgb_ev_module)

    def setUp_RENet(self):
        from src.networks.Mod_Det import MOD_Det, RENet

        K=3 #length of action tube
        arch='resnet101'
        set_head_conv=-1

        if set_head_conv != -1:
            head_conv = set_head_conv
        elif 'dla' in arch:
            head_conv = 256
        elif 'resnet' in arch:
            head_conv = 256

        num_classes=10 #this part will be later removed
        branch_info = {'hm': num_classes,
                           'mov': 2 * K,
                           'wh': 2 * K}
        #def create_model(arch, branch_info, head_conv, K, flip_test=False):
        # num_layers1 = 101
        # num_layers2 = 18
        # arch = arch[:arch.find('_')] if '_' in arch else arch
        # model = MOD_Net(arch, num_layers1, num_layers2, branch_info, head_conv, K)
        # return model


    #def create_inference_model(arch, branch_info, head_conv, K, flip_test=False):
        num_layers1 = 101
        num_layers2 = 18
        arch = arch[:arch.find('_')] if '_' in arch else arch
        backbone = RENet(arch, num_layers1, num_layers2)
        detector = MOD_Det(backbone, branch_info, arch, head_conv, K)
        #return backbone, branch
        #detector = MODDetector(backbone, branch_info, arch, head_conv, K) #backbone, branch_info, arch, head_conv, K

        #preprocessing a single frame to setup the dataloader (take example of this)
        # prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process, detector.pre_process_single_frame)
        # data_loader = torch.utils.data.DataLoader(
        #     prefetch_dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=False,
        #     drop_last=False,
        #     worker_init_fn=worker_init_fn)

        num_iters = len(self.dataloader)

        for iter, data in enumerate(self.dataloader):

            #outfile = data['outfile']

            detections = detector.run(data)
            print(detections)
            # for i in range(len(outfile)):
            #     with open(outfile[i], 'wb') as file:
            #         pickle.dump(detections[i], file)

            # Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            #     iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

            
        
    def inferBatch(self):
        for batch, labels in self.dataloader:
            print(f"input sample: {batch.keys()}")
            print(f"image shape {batch['img'].size()}")
            print(f"labels shape {labels.size()}")
            break
        image=batch['img'][0]
        ev=batch['ev'][0]
        fused_feat=self.rgb_ev_module(image, ev)
        print(fused_feat)

if __name__ == '__main__':
    
    test_uzhfpv = TestRENet()
    test_uzhfpv.setUp()
    seed=random.seed(42)
    #Accessing a Batch via DataLoader
    test_uzhfpv.inspectBatch()

    

    

    

