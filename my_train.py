
# from instance_segmentation_deprecated.my_utils import get_model_instance_segmentation
import os
import sys
# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

import numpy as np

from torch_utils.engine import train_one_epoch, evaluate
from torch_utils.coco_utils import get_coco_api_from_dataset
from torch_utils.helper import *

from utils import * 
from transform import *
from dataset import *
from model import *

# configs_path = os.path.join(parentdir, 'configs')
# configs_file_path = os.path.join(configs_path, 'default_config.yml')

data_path = 'data' # os.path.join(parentdir, 'data')
sperm_dataset_path = os.path.join(data_path, 'mixed_coco')
annotations_file_path = os.path.join(sperm_dataset_path, 'annotations.json')

model_path = "model.pth"


def main():
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    # use our dataset and defined transformations
    albumentations_transforms = get_albumentations_transforms()
    dataset = COCODataset(root_dir = os.path.join(sperm_dataset_path, 'images'), \
        coco_path=annotations_file_path, transforms = albumentations_transforms)

    train_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = 3, shuffle = True, num_workers = 2, collate_fn=collate_fn
    )
    

    # our dataset has two classes only - background and sperm
    num_classes = 2
    # model = get_instance_segmentation_model(num_classes = num_classes)
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.9)

    num_epochs = 70
    print("starting to train")
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq = 10)
        lr_scheduler.step()

    torch.save(model, model_path)

    print("THAT IS IT !")



if __name__ == '__main__':
    main()
    
