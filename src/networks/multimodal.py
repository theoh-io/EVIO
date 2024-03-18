from src.networks.networks import NetworkBase

import torch
import torch.nn as nn
from torchvision import models

class MultiModalNetwork(NetworkBase):
    def __init__(self, inputs, input_dims, output_dim, img_encoder=None, ev_encoder=None, imu_encoder=None, fusion_method=None, decoding_strat=None, num_classes=None):
        super(MultiModalNetwork, self).__init__()

        self.input_type=inputs
        self._in_dims=input_dims
        self._out_dim=output_dim

        self.feat_dim=self.init_encoders(img_encoder, ev_encoder, imu_encoder)
        self.fusion_output_dim=32
        self.init_fusion(fusion_method)
        self._fused_feat_dim=self.fusion_module.output_dim
        self.init_decoder(decoding_strat)
    
    def forward(self, image, events, imu):
        # Modality Specific Processing 
        features=self.process(image, events, imu)
        # Feature-space Fusion 
        fused_features=self.fuse(features)
        # Decoding 
        output=self.decode(fused_features)
        return output

    def init_encoders(self, img_encoder, ev_encoder, imu_encoder):

        feature_dim=[]
        if "images" in self.input_type:
            # Image module
            self.image_dim=self._in_dims[0]
            print(f"image_dim is {self.image_dim}")
            img_feature=self.init_img_encoder(img_encoder)
            feature_dim.append(img_feature)
            
        if "events" in self.input_type:
            # Events module
            self.event_dim=self._in_dims[1]
            ev_feature=self.init_ev_encoder(ev_encoder)
            feature_dim.append(ev_feature)
            
        if "imu" in self.input_type:
            # IMU module
            self.imu_dim=self._in_dims[2]
            imu_feature=self.init_imu_encoder(imu_encoder)
            feature_dim.append(imu_feature)
        
        return feature_dim

    def init_img_encoder(self, img_encoder):
        test_input = torch.randn(self.image_dim).unsqueeze(0)
        print(f"test image dim {test_input.shape}")
        if not img_encoder or img_encoder=="resnet18":
            self.image_module = models.resnet101(pretrained=True)
            #num_ftrs_image = self.image_module.fc.in_features
            # self.image_module.fc = nn.Identity()
            #print(f"num ft img: {num_ftrs_image}")
            test_output=self.image_module(test_input).squeeze()
            test_output_size=test_output.size(dim=0)
            print(f"test img output dim is {test_output_size}")
        else:
            print(f"Img encoder: {img_encoder} not implemented")
        
        return test_output_size

    def init_ev_encoder(self, ev_encoder):
        print(f"event dim: {self.event_dim}")
        test_input= torch.randn(self.event_dim).unsqueeze(0)
        print(f"test event dim {test_input.shape}")
        if not ev_encoder or ev_encoder=="linear":
            self.events_module = nn.Sequential(
                nn.Linear(self.event_dim, 64),  # Assuming 'events' has 4 features
                nn.ReLU(),
                nn.Linear(64, 64)
            ) 
            #self.events_module=models.resnet101(pretrained=True)
            test_output=self.events_module(test_input).squeeze()
            test_output_size=test_output.size(dim=0)
            print(f"test ev output dim is {test_output_size}")

                   
        else:
            print(f"Event encoder: {ev_encoder} not implemented")

        return test_output_size

    def init_imu_encoder(self, imu_encoder):
        test_input= torch.randn(self.imu_dim).unsqueeze(0)
        print(f"test imu dim {test_input.shape}")
        if not imu_encoder or imu_encoder=="linear":
            self.imu_module = nn.Sequential(
                nn.Linear(self.imu_dim, 64),  # Assuming 'imu' has 7 features
                nn.ReLU(),
                nn.Linear(64, 64)
            )
        elif imu_encoder=="wavenet":
            print("using wavenet")
            from .wavenet import WaveNetModel
            self.imu_module =  WaveNetModel(classes= self.imu_dim)
        else:
            print(f"IMU encoder: {imu_encoder} not implemented")

        test_output=self.imu_module(test_input).squeeze()
        test_output_size=test_output.size(dim=0)
        print(f"test imu output dim is {test_output_size}")

        return test_output_size

    def init_fusion(self, fusion_method):
        # Fusion layer
        # if not fusion_method or fusion_method=="linear":
        #     self.fusion_module = nn.Linear(num_ftrs_image + 64 + 64, 512)

        print(f"feature dim in fusion : {self.feat_dim}")
        if fusion_method=="linear" or fusion_method=="concatenation":
            from src.networks.fusion import Concat
            self.fusion_module = Concat(self.feat_dim, self.fusion_output_dim)
        elif fusion_method=="low_rank":
            from src.networks.fusion import LMF
            dropout=[0, 0.1, 0.15, 0.2, 0.3, 0.5]
            rank = 16
            self.fusion_module = LMF(self.feat_dim, dropout, self.fusion_output_dim, rank).cuda()
        elif fusion_method=="attention":
            print("not implemented yet !!")
        else:
            print(f"fusion method not implemented: {fusion_method}")

    def init_decoder(self, decoding_strat):
        # Output layer
        if not decoding_strat:
            self.decoder_module = nn.Linear(32, self._out_dim)
        elif decoding_strat=="simple_fc":
            print("using simple decoding")
            from .decoder import SimpleDecoder
            self.decoder_module=SimpleDecoder(32, self._out_dim)
    
    def process(self, image, events, imu):
        # Image processing
        #print(f"image shape: {image.shape}")
        image_features = self.image_module(image)
        # Events processing
        events_features = self.events_module(events)
        #events_features = events_features.unsqueeze(1)
        # IMU processing
        imu_features = self.imu_module(imu)
        #imu_features = imu_features.unsqueeze(1)
        print(f"Image features size: {image_features.size()}")
        print(f"Events features size: {events_features.size()}")
        print(f"IMU features size: {imu_features.size()}")

        features = torch.cat((image_features, events_features, imu_features), dim=1)
        return features
    
    def fuse(self, features):
        # Fusion (make sure to be on cuda gpu and converted to float)
        # DTYPE = torch.cuda.FloatTensor
        fused_features = self.fusion_module(features)
        print(fused_features.shape)
        return fused_features
    
    def decode(self, feature):
        output = self.decoder_module(feature)
        print(output.shape)
        return output

