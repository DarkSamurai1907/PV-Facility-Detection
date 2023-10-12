import os
import torch
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def normalize_to_binary(tensor):
    if tensor.max() == tensor.min():
        return tensor

    tensor_min = tensor.min()
    tensor_range = tensor.max() - tensor_min
    normalized_tensor = (tensor - tensor_min) / tensor_range

    binary_tensor = torch.where(normalized_tensor < 0.5, torch.tensor(0.0, device=tensor.device),
                                torch.tensor(1.0, device=tensor.device))

    return binary_tensor


data_dir = "imagesv2"

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: normalize_to_binary(x))

])


class PVDetectionDataset(Dataset):
    def __init__(self, coco_annotation_file, image_directory, image_transform=None, mask_transform=None):
        self.coco = COCO(coco_annotation_file)
        self.image_directory = image_directory
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_id = self.coco.getImgIds()[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.image_directory, img_info['file_name'])  # Construct the image path
        image = Image.open(image_path)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = torch.zeros((img_info['height'], img_info['width']), dtype=torch.float32)

        for ann in anns:
            mask += self.coco.annToMask(ann)

        if self.image_transform and self.mask_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask