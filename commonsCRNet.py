import io

from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torch

from pythonModel import CSRNet

def get_model():
    model = CSRNet()
    model.load_state_dict(torch.load('./modelCRNet.pt', map_location='cpu')) # Where we upload our model (Download model to local)
    model.eval()
    return model
