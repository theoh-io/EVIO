from src.options.config_parser import ConfigParser
from src.networks.networks import NetworksFactory
from src.models.models import ModelsFactory
import torch

# get config
config = ConfigParser().get_config()

# Remove 'num_classes' key
if "num_classes" in config["networks"]["reg"]["hyper_params"]:
    del config["networks"]["reg"]["hyper_params"]["num_classes"]

# get size
dataset_type=config["dataset"]["type"]
modalities=config[dataset_type]["inputs"]
n_modalities=len(modalities)
print(f"number of modalities {n_modalities}")

#image params
B = config["dataset"]["batch_size"]
S1, S2 = config["dataset"]["image_size"]
T= config["dataset"]["target_nc"]
n_img = config["dataset"]["img_nc"]

#other Modalities
E=config["dataset"]["event_dim"]
I=config["dataset"]["imu_dim"]

#output target
T=config["dataset"]["target_nc"]

# create network
model_type=config["model"]["type"]

print("in test print config")
print(config["networks"]["reg"])
#nn_type = config["networks"]["reg"]["type"]
#nn_hyper_params = config["networks"]["reg"]["hyper_params"]
model= ModelsFactory.get_by_name(model_type, config)
#nn = NetworksFactory.get_by_name(nn_type, **nn_hyper_params)

nn=model._reg
# forward pass
img = torch.ones([B, n_img, S1, S2])
ev=torch.ones([B,E])
imu=torch.ones([B,I])
#imu=torch.ones([B,I,10])
y=nn(img, ev, imu)
#nn.print()
print(f"result vs expected target: {y.shape}, {B, T}")
y = nn.forward(img, ev, imu)

#Backward pass
input={"img": img, "event": ev, "imu": imu, "target": y}
loss=model.forward(input)
print(loss)
