import torchvision.transforms as standard_transforms
from .SHHA import SHHA
from .data_loader import DataLoader

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def loading_data(data_root):
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    # create the training dataset
    #train_set = SHHA(data_root, train=True, transform=transform, patch=True, flip=True)
    # create the validation dataset
    #val_set = SHHA(data_root, train=False, transform=transform)

#------------------------------------
    train_path = '/home/zaki/Documents/Master/Code/image/CrowdCounting-P2PNet-main/DATA_ROOT/train'
    train_gt_path = '/home/zaki/Documents/Master/Code/image/CrowdCounting-P2PNet-main/DATA_ROOT/train_den'
    val_path = '/home/zaki/Documents/Master/Code/image/CrowdCounting-P2PNet-main/DATA_ROOT/val'
    val_gt_path = '/home/zaki/Documents/Master/Code/image/CrowdCounting-P2PNet-main/DATA_ROOT/val_den'
   
    # create the training dataset
    train_set = DataLoader(train_path, train_gt_path,transform ,shuffle=True, gt_downsample=True)
    # create the validation dataset
    val_set = DataLoader(val_path, val_gt_path, transform,shuffle=False, gt_downsample=True)
    
    
    return train_set, val_set
