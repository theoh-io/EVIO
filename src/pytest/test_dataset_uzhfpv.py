import unittest
from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.utils import util
from collections import OrderedDict

class TestUZHFPVDataset(unittest.TestCase):
    def setUp(self):
        print("setup")
        self.config = ConfigParser().get_config()
        self.data_loader = CustomDatasetDataLoader(self.config, is_for="train")
        self.dataset = self.data_loader.load_data()
        # print dimensions of input and target
        self.dataset_size = len(self.dataset)
        print('#training images = %d' % self.dataset_size)


    # def test_dataset_loading(self):
    #     input=[]
    #     for i, data in enumerate(self.dataset):
    #         for i in self.config['uzh_fpv']['inputs']:
    #             input.extend(data[i])
    #         #create a ne
    #         target = data['target']

    #         self.assertIsNotNone(input)
    #         self.assertIsNotNone(target)

    #         # self.assertEqual(img.shape[0], self.config['batch_size'])
    #         # self.assertEqual(target.shape[0], self.config['batch_size'])

    #         break

if __name__ == '__main__':
    
    #config = ConfigParser().get_config()
    test_uzhfpv = TestUZHFPVDataset()
    test_uzhfpv.setUp()
    #test_uzhfpv.test_dataset_loading()