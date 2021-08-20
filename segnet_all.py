
import os
import cv2
import torch
import numpy as np 

from utils import process_coco
from dataset import to_uint8_tensor, image_to_float_tensor
from model import get_segnet
from torch_utils.coco_utils import convert_coco_poly_to_mask

class SegNetSpermDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, coco_path, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        # process coco file
        images, categories = process_coco(coco_path)
        self.images = images
        self.categories = categories
        self.num_classes = len(self.categories)

    def __getitem__(self, idx):

        def masks_to_mask(masks):
            # print("type:", type(masks))
            # print("shape:", masks.shape)
            if isinstance(masks, torch.Tensor):
                mask = masks.numpy().transpose(1,2,0).sum(axis=2).astype(np.int64)
            else: mask = masks.transpose(1,2,0).sum(axis=2).astype(np.int64)
            return mask

        # get one image dict from processed coco file
        image_dict = self.images[idx]
        abs_image_path = os.path.join(self.root_dir, image_dict["file_name"])

        image = cv2.imread(abs_image_path)
        h, w=  image.shape[:2]

        # parse annotations
        segmentations = [annot['segmentation'] for annot in image_dict['annotations']]
        category_ids = [annot['category_id'] for annot in image_dict['annotations']]

        # create masks from coco segmentation polygons
        masks = convert_coco_poly_to_mask( segmentations, h, w)
        mask = masks_to_mask(masks)

        return image_to_float_tensor(image), to_uint8_tensor(mask)

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = list(target.to(device) for target in targets) 
        print("::: {}  {}".format(len(images), len(targets)))
        predicted = model(images)

        loss = criterion(predicted, targets)
        print("epoch:{}  loss:{}".format(epoch, loss))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



data_path = 'data' # os.path.join(parentdir, 'data')
SpermDS = os.path.join(data_path, 'sperm_dataset')
Annots = os.path.join(SpermDS, 'annotations.json')
model_path = "model.pth"

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = SegNetSpermDataset(root_dir = os.path.join(SpermDS, 'images'), coco_path=Annots, transforms = None)
    train_data_loader = torch.utils.data.DataLoader( dataset, batch_size = 3, shuffle = True, num_workers = 2, collate_fn=collate_fn)

    num_classes = 2
    model = get_segnet(num_classes)
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.9)

    num_epochs = 20
    print("starting to train")
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_data_loader, device, epoch)
        lr_scheduler.step()

    torch.save(model, model_path)

    print("THAT IS IT !")



if __name__ == '__main__':
    main()