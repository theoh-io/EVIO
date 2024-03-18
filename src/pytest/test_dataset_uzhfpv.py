import unittest
from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.utils import util
from collections import OrderedDict

import random
import numpy as np

# Test to create the dataloader for UZH

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        print("setup")
        self.config = ConfigParser().get_config()
        self.dataloader = CustomDatasetDataLoader(self.config, is_for="train")
        self.dataloader = self.dataloader.load_data()
        # print dimensions of input and target
        self.data_size = len(self.dataloader)
        print('#training images = %d' % self.data_size)

    def inspectSample(self):
        #generate a random number to pick a test sample
        rand_num=np.random.randn()
        print(rand_num)
        #visualize some images
        sample_img=self.dataloader._images[rand_num]
        print(f"sample img: {sample_img.size()}")
        #print imus between 2 frames
        sample_imu= self.data_loader._imu[rand_num:(rand_num+10)]

        #print ensemble of events betwee 2 frames
        
    def inspectBatch(self):
        for batch, labels in self.dataloader:
            print(f"input sample: {batch.keys()}")
            print(f"image shape {batch['img'].size()}")
            print(f"labels shape {labels.size()}")

if __name__ == '__main__':
    
    test_uzhfpv = TestDataLoader()
    test_uzhfpv.setUp()
    seed=random.seed(42)
    #Accessing directly the sample within dataset
    #test_uzhfpv.inspectSample()
    #Accessing a Batch via DataLoader
    test_uzhfpv.inspectBatch()

    

    

    

