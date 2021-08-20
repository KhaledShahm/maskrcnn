import os
import sys
sys.path.append('torch_utils')
import numpy as np
import torch
from PIL import Image
import cv2 

from torchvision.transforms import functional as F
import albumentations as A
from albumentations.core.composition import BboxParams, Compose
# from pycocotools.coco import COCO
from pycocotools import coco as co

from torch_utils.coco_utils import *
from torch_utils.coco_eval import *

from my_utils import *
from utils import *

# from transform import get_albumentations_transforms

class PennFudanDataset(object):
    def __init__(self, root, transforms = None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



class COCODataset(object):
    """
    Compatible with any coco style annotation file, annotations must include
    segmentation mask (polygon coordinates). Bboxes are created from masks.
    Arguments:
        root_dir: Root directory that contains image files. Relative image
        file locations from coco file will be joined with this root_dir while
        iterating.
        coco_path: Path to the coco style annotation file.
        transforms: Albumentations compose object.
    """

    def __init__(self, root_dir, coco_path, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        # process coco file
        images, categories = process_coco(coco_path)
        self.images = images
        self.categories = categories
        self.num_classes = len(self.categories)

    def __getitem__(self, idx):
        # get one image dict from processed coco file
        image_dict = self.images[idx]

        # parse image path
        relative_image_path = image_dict["file_name"]
        # get absolute image path
        abs_image_path = os.path.join(self.root_dir, relative_image_path)
        # load image
        image = cv2.imread(abs_image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # parse annotations
        segmentations = []
        category_ids = []

        # find if negative sample
        is_negative_sample = False
        if len(image_dict["annotations"]) == 0:
            is_negative_sample = True
            voc_bboxes = []

        if not is_negative_sample:
            for annotation in image_dict["annotations"]:
                # get segmentation polygons
                segmentations.append(annotation["segmentation"])
                # get category id
                category_id = annotation["category_id"]
                category_ids.append(category_id)

            # create masks from coco segmentation polygons
            masks = convert_coco_poly_to_mask(
                segmentations, height=image.shape[0], width=image.shape[1]
            )

            # create coco and voc bboxes from coco segmentation polygons
            (coco_bboxes, voc_bboxes) = convert_coco_poly_to_bbox(
                segmentations, height=image.shape[0], width=image.shape[1]
            )

            mask = masks_to_mask(masks)
            masks = list(masks.numpy())
            
            if self.transforms is not None:
                # arrange transform data
                data = {
                    "image": image,
                    "bboxes": voc_bboxes,
                    "masks": masks,
                    "class_labels": category_ids,
                }
                # apply transform
                augmented = self.transforms(image = image, masks = masks, bboxes = voc_bboxes, class_labels= category_ids)
                # get augmented image and bboxes
                image = augmented["image"]
                voc_bboxes = augmented["bboxes"]
                category_ids = augmented["class_labels"]
                # get masks
                masks = augmented["masks"]
                mask = masks_to_mask(np.array(masks))


        # check again if augmentation result is negative sample
        if (not is_negative_sample) and (self.transforms is not None):
            if len(augmented["bboxes"]) == 0:
                is_negative_sample = True

        # convert everything into a torch.Tensor
        target = {}

        # boxes
        if (not is_negative_sample) and (
            not voc_bboxes == []
        ):  # if not negative taret and voc_bboxes is not empty
            target["boxes"] = boxes = to_float32_tensor(voc_bboxes)
        else:  # negative target
            target["boxes"] = boxes = torch.zeros((0, 4), dtype=torch.float32)

        # labels
        if not is_negative_sample:  # positive target
            target["labels"] = to_int64_tensor(category_ids)
        else:  # negative target
            target["labels"] = torch.zeros(0, dtype=torch.int64)

        # masks
        if not is_negative_sample:  # positive target
            target["masks"] = to_uint8_tensor(masks)
        else:  # negative target
            target["masks"] = torch.zeros(
                0, image.shape[0], image.shape[1], dtype=torch.uint8
            )

        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["image_id"] = torch.tensor([idx])
        num_objects = len(target["boxes"])
        target["iscrowd"] = torch.zeros((num_objects,), dtype=torch.int64)

        # normalize image
        image = image / np.max(image)
        return image_to_float_tensor(image), target # target

    def __len__(self):
        return len(self.images)


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = co.COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        

        # number of objects in the image
        num_objs = len(coco_annotation)

        # masks
        w, h = img.size
        masks = np.zeros((num_objs, h, w))

        for obj in range(num_objs):
            masks[obj, :, :] = self.coco.annToMask(coco_annotation[obj])
 

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        
        # Labels (In my case, I have only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation['path'] = path
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation['masks'] = masks
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd


        img = np.array(img)
        masks_list = list(my_annotation['masks'])

        if self.transforms is not None:
            class_labels = labels.numpy()
            transformed = self.transforms(image = img, masks = masks_list, bboxes = boxes, class_labels=class_labels)
        
            img = transformed['image']
            masks_list = transformed['masks']
            boxes = transformed['bboxes']

        img = F.to_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        masks = torch.as_tensor(np.array(masks_list), dtype=torch.float32)
        
        if len(list(masks.shape)) == 1:
            masks = torch.zeros(0, img.shape[0], img.shape[1], dtype=torch.float32)

        my_annotation["boxes"] = boxes
        my_annotation['masks'] = masks
        return img, my_annotation


    def __len__(self):
        return len(self.ids)


def to_float32_tensor(to_be_converted):
    return torch.as_tensor(to_be_converted, dtype=torch.float32)


def to_int64_tensor(to_be_converted):
    return torch.as_tensor(to_be_converted, dtype=torch.int64)


def to_uint8_tensor(to_be_converted):
    return torch.as_tensor(to_be_converted, dtype=torch.uint8)


def image_to_float_tensor(image):
    # Converts numpy images to pytorch format
    return torch.from_numpy(image.transpose(2, 0, 1)).float()





def show_image(image):
    cv2.imshow('mask', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():

    dataset = PennFudanDataset('dataset/PennFudanPed')
    image, target = dataset[0]
    print(target.keys())
    masks = target['masks']
    for i in range(masks.shape[0]):
        mask = masks[i]
        print(np.unique(mask))
        mask = np.array(mask*255, dtype = np.uint8)
        show_image(mask)

    return


    # images_path = r'dataset\sperm\images'
    # annotations = r'dataset\sperm\annotations.json'
    # coco = co.COCO(annotations)

    # # cat_ids = coco.getCatIds()
    # img_ids = coco.getImgIds()
    # anns_ids = coco.getAnnIds(imgIds = 9)
    # anns = coco.loadAnns(anns_ids)

    # image = cv2.imread(os.path.join(images_path, coco.imgs[0]['file_name']))
    # h, w = image.shape[:2]
    # mask = np.zeros((h, w))
    # ann_id = 1
    # for ann in anns:
    #     mask = np.maximum(mask, coco.annToMask(ann)*ann['category_id']*ann_id)
    #     ann_id += 1
    # # mask = np.array((mask/mask.max())*255, dtype = np.uint8)
    # objects = np.unique(mask)
    # objects = objects[1:] # remove the background label
    # print(mask.shape, '  ', objects.shape)
    # masks = mask == objects[:, None, None]
    # print(masks.shape)
    # for i in range(masks.shape[0]):
    #     mask = masks[i, :, :]
    #     mask = np.array((mask/mask.max())* 255, dtype = np.uint8)

    return


def masks_to_mask(masks):
  if isinstance(masks, torch.Tensor):
    mask = masks.numpy().transpose(1,2,0).sum(axis=2).astype(np.int64)
  else: mask = masks.transpose(1,2,0).sum(axis=2).astype(np.int64)
  return mask


def main():

    root = r'data\sperm_dataset\images'
    annotations = r'data\sperm_dataset\annotations.json'
    coco = myOwnDataset(root, annotations, get_albumentations_transforms())

    data = coco[1]
    image, annotations = data
    image = image.numpy().transpose(1,2,0)
    # mask = annotations['masks'].numpy().transpose(1,2,0).sum(axis=2)
    mask = annotations['masks'].numpy().transpose(1,2,0).sum(axis=2)

    image = np.array((image/image.max())*255, dtype = np.uint8)
    mask = np.array((mask/mask.max())*255, dtype = np.uint8)


    boxes = annotations['boxes'].numpy().astype(int)
    boxes = list(boxes)
    
    boxed_image = image.copy()
    for box in boxes:
        x, y, x2, y2 = box
        x_center = x + (x2//2)
        y_center = y + (y2//2)
        boxed_image = cv2.rectangle(boxed_image, (int(x), int(y)), (int(x2), int(y2)), (255,0,0), 2)
    
    # show_image(image)
    # show_image(mask)
    # show_image(boxed_image)

def buid_yolo_dataset():
    root = r'data\sperm_dataset\images'
    annotations = r'data\sperm_dataset\annotations.json'
    coco = myOwnDataset(root, annotations, get_albumentations_transforms())

    images_dir = os.path.join('yolo_data', 'images')
    boxed_images_dir = os.path.join('yolo_data', 'boxed_images')
    labels_dir = os.path.join('yolo_data', 'labels')


    for i in range(20):

        data = coco[random.randint(0, len(coco)-1)]
        image, annotations = data
        file_name = annotations['path'].split('.')[0]
        image_path = os.path.join(images_dir, file_name + '_{}_'.format(i) + '.jpg')
        label_path = os.path.join(labels_dir, file_name+ '_{}_'.format(i)+'.txt')
        boxed_image_path = os.path.join(boxed_images_dir, file_name + '_{}_'.format(i) + '.jpg')

        image = image.numpy().transpose(1,2,0)
        # mask = annotations['masks'].numpy().transpose(1,2,0).sum(axis=2)
        
        mask = annotations['masks'].numpy().transpose(1,2,0).sum(axis=2)

        image = np.array((image/image.max())*255, dtype = np.uint8)
        mask = np.array((mask/mask.max())*255, dtype = np.uint8)


        boxes = annotations['boxes'].numpy().astype(int)
        boxes = list(boxes)
        if len(boxes ) == 0 : continue 
        boxed_image = image.copy()
        file = open(label_path, 'w') 
        for box in boxes:
            x, y, x2, y2 = box
            height = (y2 - y) / image.shape[0]
            width = (x2 - x)  / image.shape[1]
            y_center = (y + (height//2)) / image.shape[0]
            x_center = (x + (width//2)) / image.shape[1]
            file.write(str(0)+' '+str(x_center)+' '+str(y_center)+' '+str(width)+' '+str(height)+'\n')
            boxed_image = cv2.rectangle(boxed_image, (int(x), int(y)), (int(x2), int(y2)), (255,0,0), 2)
        file.close()
        cv2.imwrite(image_path, image)
        cv2.imwrite(boxed_image_path, boxed_image)

    # print(file_name)
    # show_image(image)
    # print(image.shape)
    # show_image(boxed_image)



def test_dataset():
    def my_collate_fn(batch):
        return tuple(zip(*batch))
    sperm_dataset_path = os.path.join('data', 'morphology_coco')
    annotations_file_path = os.path.join(sperm_dataset_path, 'annotations.json')

    albumentations_transforms = get_albumentations_transforms( )
    dataset = COCODataset(root_dir = os.path.join(sperm_dataset_path, 'images'), \
        coco_path=annotations_file_path, transforms = albumentations_transforms)

    train_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = 3, shuffle = True, num_workers = 2, collate_fn=my_collate_fn
    )

    i = dataset[1]
    print(type(i))
    
if __name__ == "__main__":

    test_dataset()
    