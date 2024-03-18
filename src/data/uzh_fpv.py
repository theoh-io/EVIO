import os.path
from src.data.dataset import DatasetBase
import numpy as np
from torchvision.datasets.utils import download_url
import tarfile
import sys
import pickle
from PIL import Image
from tqdm import tqdm
import pandas as pd


class SyncedImg:
    #to add attribute to distingish type of data
    #method to acces data by name
    #handle output data
    def __init__(self, attributes):
        self._imgs=[]
        self._ts=[]
        #self.root_path = root_path  # Root path where data files are stored
        #self.data = {attr: [] for attr in attributes}  # Initialize data storage
        # Initialize each attribute as an empty list
    #     if "image_name" in attributes:
    #         attributes[1]="image"
    #     for attr in attributes:
    #         setattr(self, attr, [])

    # def parse_data_raw(self, raw_data):
    #     for attr, value in vars(self):
    #         # attribute i is equal to raw_data[:,1]

    # def load_img_from_path(self, path):
    #     pass
        


class UZHFPVDataset(DatasetBase):
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(UZHFPVDataset, self).__init__(opt, is_for, subset, transform, dataset_type)
        self._name = 'uzh_fpv' #replace by config dataset type
        # init meta
        self._init_meta(opt)
        # read dataset
        self._read_dataset()
        # read meta
        #self._read_meta()

    def _init_meta(self, opt):
        #self._rgb = not opt[self._name]["use_bgr"]
        self._root = opt[self._name]["data_dir"]
        self._data_folder = opt[self._name]["data_folder"]
        #self._meta_file = opt[self._name]["meta_file"]
        #self._url = opt[self._name]["url"]
        #self._filename = opt[self._name]["filename"]
        #self._tgz_md5 = opt[self._name]["tgz_md5"]

        self._inputs= opt[self._name]["inputs"]["modalities"]
        #replace by dynamic creation for the name of var
        # for i in self._inputs:
        #     self.attr_"transforms_{}".format(self._is_for)

        self._attr_imu=opt[self._name]["inputs"]["imu"]
        self._attr_events=opt[self._name]["inputs"]["events"]
        self._attr_images=opt[self._name]["inputs"]["images"]
        self._imgsz = opt["dataset"]["image_size"]
        self._output = opt[self._name]["output"]

        self._imus=np.array([])
        self._events=np.array([])
        # self._imus=InputData(self._attr_imu)
        # self._events=InputData(self._attr_events)
        #self._images=SyncedImg(self._attr_images)
        self._dataset_size=[] #might be different size for each input modality
        #self._data={'images': self._images, 'events': self._events, 'imus': self._imus}

        if self._is_for == "train":
            self._ids_filename = self._opt[self._name]["train_ids_file"]
        elif self._is_for == "val":
            self._ids_filename = self._opt[self._name]["val_ids_file"]
        elif self._is_for == "test":
            self._ids_filename = self._opt[self._name]["test_ids_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def __getitem__(self, index):
        #assert (index < self._dataset_size)

        #write a function to check that index is inside acceptable size 
        # also check that the timestamps across modalities are matching

        # get data
        img=self._images[index]
        #ts_img=self._images._ts[index]
        imu=self._imus[index]

        #how to handle the frequency to index the events
        event=self._events[index]

        target=self._targets[index]
        # print("get item")
        # print(f"ts_img: {ts_img}")
        # print(f"event {event}")
        # print(f"imu {imu}")
        
        # pack data
        sample = {'img': img, 'imu': imu, 'event': event}

        # later will get rid of the ts in sample but use it to integrate events and IMU

        # apply transformations
        if self._transform is not None:
            sample = self._transform(sample)

        return sample, target

    def __len__(self):
        #review how I should Implement this correctly
        #print("in len")
        #print(self._dataset_size)
        return self._dataset_size[0][0][0]
        #return self._dataset_size
    
    def _read_images(self, file_path, file_name):
        with open(file_path, 'r') as f:
            temp_images=[]
            temp_timestamps=[]
            # Skip the first line
            f.readline()
            # Read the rest of the file line by line
            for line in f:
                fields = line.split()
                ts = float(fields[1])  # Convert timestamp to float
                image_path = fields[-1]#.decode('utf-8')  # use the last field as the file path
                image_path=os.path.join(self._root, self._data_folder, file_name, image_path)
                #img=self._load_images(image_path)
                img = Image.open(image_path)
                # Convert the image to a NumPy array
                img = np.array(img)
                # add the image to data 
                #print(data)
                
                temp_images.append(img)
                temp_timestamps.append(ts)
                #temp_data["images"].append((timestamp, img))  # Append as a tuple
            # Stack and reshape images
            print(f"ri: before vstack: {np.array(temp_images).shape}")
            np_images=np.array(temp_images)
            #np_images = np.vstack(temp_images).reshape(-1, 1, self._imgsz[0], self._imgsz[1])
            #np_images=np_images.transpose((0, 2, 3, 1))  # convert to HWC
            print(f"ri: after vstack: {np_images.shape}")
            #print(fields)
        return np_images, np_images
    
    def _read_events(self, file_path):
        with open(file_path, 'r') as f:
            temp_events=[]
            temp_ts=[]

            total_rows = sum(1 for row in open(file_path, 'rb')) - 1  # Subtract 1 to account for the header row
            #max_rows = total_rows // 8
            max_rows=50000
            print(f"number of rows: {total_rows}, max rows {max_rows}")
            # Skip the first line
            f.readline()
            
            row=0
            t = tqdm(total=max_rows, desc="Loading Data")
            while row<max_rows:
                line = f.readline()
                if not line:  # Break if end of file is reached
                    break
                #print(line)
                # for line in tqdm(lines):
                data = line.split()
                data = [float(item) for item in data]
                    #print(data)
                temp_events.append(data)
                #temp_data["events"].append(data)
                row+=1
                t.update()
            t.close()
            #chunk_size = 1000
            # with pd.read_csv(file_path, chunksize=chunk_size, skiprows=1, header=None) as reader:
            #     # Set up the tqdm progress bar
            #     t = tqdm(total=max_rows, desc="Loading Data")

            #     rows_processed = 0
            #     for chunk in reader:
            #         # Process each chunk here
            #         temp_data[i].extend(chunk.values.tolist().split())
            #         rows_processed += chunk_size
            #         # Update the progress bar
            #         t.update(len(chunk))
            #         if rows_processed >= max_rows:
            #             break

            #     t.close()  # Close the progress bar after all chunks are processed

        return np.array(temp_events), np.array(temp_ts)
    
    def _read_imu(self, file_path):
        with open(file_path, 'r') as f:
            temp_imu=[]
            temp_ts=[]
            total_rows = sum(1 for row in open(file_path, 'rb')) - 1  # Subtract 1 to account for the header row
            print(f"number of rows: {total_rows}")
            # Skip the first line
            f.readline()
            
            while True:
                line = f.readline()
                if not line:  # Break if end of file is reached
                    break
                #print(line)
                data = line.split()
                data = [float(item) for item in data]
                #print(data)
                temp_imu.append(data)

            # Use pandas to read the file in chunks (for example, 10000 lines at a time)
            # chunksize = 1
            # for chunk in pd.read_csv(file_path, chunksize=chunksize, skiprows=1, header=None, delimiter=' '):
            #     temp_data[i]=chunk.values.tolist()
            #     #temp_data[i].extend(chunk.values.tolist())
        return np.array(temp_imu), np.array(temp_ts)
    
    def _read_gt(self, file_name):
        temp_gt=[]
        o=self._output+'.txt'
        file_path = os.path.join(self._root, self._data_folder, file_name, o)
        with open(file_path, 'r') as f:
            #Skip the first line
            f.readline()
            for line in f:
                data = line.split()
                data = [float(item) for item in data]
                temp_gt.append(data)
        return np.array(temp_gt)

    def _read_dataset(self):
        # read ids
        use_ids_filepath = os.path.join(self._root, self._ids_filename) #datasets/uzh_fpv/train_ids.txt
        valid_ids_root = self._read_valid_ids(use_ids_filepath) # ['Track1/txt']
        temp_data = {'images': [], 'events': [], 'imu':[]}  # Initialize as a dictionary with lists
        temp_ts= {'images': [], 'events':[], 'imu':[]}
        #self._targets = []
        for file_name in valid_ids_root: #loop for the different tracks
            for i in self._inputs: #loop for different modalities
                print(f"reading dataset: {i}")
                input_name=i+'.txt'
                #datasets/uzh_fpv/Tracks/Track1/txt/images.txt
                file_path = os.path.join(self._root, self._data_folder, file_name, input_name)
                    # if i not in self._data:
                    #     self._data[i] = []
                    #     print(f"unexpected error of key name in self._data {self._data[i]}")
                if i=='images':
                    temp_data['images'], temp_ts['images']=self._read_images(file_path, file_name)
                elif i=='events':
                    temp_data['events'], temp_ts['events']=self._read_events(file_path)
                elif i =="imu":
                    temp_data['imu'], temp_ts['imu'] = self._read_imu(file_path)
            self._targets=self._read_gt(file_name)      
            
        #add the stacking logic for the input data here (integration constant, event rpz...)
        temp_data=self.stack_data(temp_data)


        self._images= temp_data['images']
        print(f"read_dataset, _images shape: {self._images.shape}")
        #self._images._ts=temp_ts['images']
        self._events=temp_data['events']
        self._imus=temp_data['imu']

        self._dataset_size.append((self._images.shape,self._events.shape, self._imus.shape))
        # self._data['images']=np.vstack(self._data['images']).reshape(-1, 1, self._imgsz[0], self._imgsz[1])
        #self._data = self._data.transpose((0, 2, 3, 1))  # convert to HWC
        
        # print(self._data.keys())
        # for i in self._inputs:
        #     print(i)
        #     print(self._data[i].shape)
        # print(self._data['images'].shape)
        #self._data['images']=np.vstack(self._data['images']).reshape(-1, self._imgsz[0], self._imgsz[1])
        #self._data = np.vstack(self._data).reshape(-1, 3, 32, 32)
        

        # dataset size
        # for i in self._inputs:
        #     self._dataset_size.append(len(temp_data[i]))
        print(f"size of the dataset: {self._dataset_size}")
        print(f"size of the output: {self._targets.shape}")

    def stack_data(self, data):
        #implement the stacking logic here
        pass
        return data

    # def _load_images(self, image_path):
    #     # list_images = []
    #     # for image_path in list_image_path:
    #         # Open the image
    #     image = Image.open(image_path)
    #     # Convert the image to a NumPy array
    #     image_array = np.array(image)
    #     image_resized=np.reshape(image_array, (1, self._imgsz[0], self._imgsz[1]))
    #         # Add the image to the list of images
    #         # list_images.append(image_array)
    #     #print(f"in load img img resized {image_resized.size()}")
    #     # # Create a stack of images
    #     # list_images=np.vstack(list_images).reshape(-1, self._imgsz[0], self._imgsz[1])
    #     return image_resized

    # def _read_meta(self):
    #     path = os.path.join(self._root, self._data_folder, self._meta_file)
    #     with open(path, 'rb') as infile:
    #         if sys.version_info[0] == 2:
    #             data = pickle.load(infile)
    #         else:
    #             data = pickle.load(infile, encoding='latin1')
    #         self.classes = data["label_names"]
    #     self._class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _read_valid_ids(self, file_path):
        ids = np.loadtxt(file_path, dtype=str)
        return np.expand_dims(ids, 0) if ids.ndim == 0 else ids

    # def _download(self):
    #     # check already downloaded
    #     if os.path.isdir(os.path.join(self._root, self._data_folder)):
    #         return

    #     # download file
    #     print("It will take aprox 15 min...")
    #     download_url(self._url, self._root, self._filename, self._tgz_md5)

    #     # extract file
    #     with tarfile.open(os.path.join(self._root, self._filename), "r:gz") as tar:
    #         tar.extractall(path=self._root)