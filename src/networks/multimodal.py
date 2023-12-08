from src.networks.networks import NetworkBase
import torch
import torch.nn as nn
from torchvision import models

class MultiModalNetwork(NetworkBase):
    def __init__(self, inputs, input_dims, output_dim, img_encoder=None, ev_encoder=None, imu_encoder=None, fusion_method=None, decoding_strat=None, num_classes=None):
        super(MultiModalNetwork, self).__init__()

        if "images" in inputs:
            # Image module
            self.image_dim=input_dims[0]
            print(self.image_dim)
            if not img_encoder or img_encoder=="resnet18":
                self.image_module = models.resnet18(pretrained=True)
                num_ftrs_image = self.image_module.fc.in_features
                self.image_module.fc = nn.Identity()
            else:
                print(f"Img encoder: {img_encoder} not implemented")
        if "events" in inputs:
            # Events module
            self.event_dim=input_dims[1]
            print(self.event_dim)
            if not ev_encoder or ev_encoder=="linear":
                self.events_module = nn.Sequential(
                    nn.Linear(self.event_dim, 64),  # Assuming 'events' has 4 features
                    nn.ReLU(),
                    nn.Linear(64, 64)
                )        
            else:
                print(f"Event encoder: {ev_encoder} not implemented")
        if "imu" in inputs:
            # IMU module
            self.imu_dim=input_dims[2]
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
                print(f"Img encoder: {img_encoder} not implemented")

        # Fusion layer
        if not fusion_method or fusion_method=="linear":
            self.fusion = nn.Linear(num_ftrs_image + 64 + 64, 512)

        # Output layer
        if not decoding_strat:
            print("here")
            self.output = nn.Linear(512, output_dim)
        elif decoding_strat=="simple_fc":
            print("using simple decoding")
            from .decoder import SimpleDecoder
            self.output=SimpleDecoder()

    def forward(self, image, events, imu):
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
        # Fusion
        combined = torch.cat([image_features, events_features, imu_features], dim=1)
        combined = torch.relu(self.fusion(combined))

        # Output
        output = self.output(combined)
        #print(output.shape)
        return output
