# Dataloaders for DeepSeagrass dataset

# Import packages
import torch
import numpy as np
import os, random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

import imgaug.augmenters as iaa

### Dataset Classes ###

# Image dataset: this dataset randomly chooses samples for each batch and performs data augmentation for training
class ImageDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir, train=True):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.train = train

    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        row = 4
        col = 7

        width = int(image.shape[1])
        height = int(image.shape[0])

        # Divide the full image into a grid of patches
        grid, _, _ = self.img_to_grid(image,row,col)

        all_patches = []
        for patch in grid:
            patch_crop = self.cropper(patch, int(np.floor(width / col)), int(np.floor(height / row)))
            all_patches.append(torch.unsqueeze(self.transform_func(Image.fromarray(patch_crop)), dim=0))

        all_patches_torch = torch.cat(all_patches, dim=0)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        label_new = label.clone()  # Clone to avoid modifying the original tensor

        if self.train:
            label_new[label == 1] = 0  # Change all 1s to 0
            label_new[label == 2] = 1  # Change all 2s to 1

        return all_patches_torch, label_new

    def transform_func(self, image):
        'Transform into a pytorch Tensor'
        transform_list = []
        transform_list.append(transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC))  # this is for full size (original) model
        transform_list.append(transforms.CenterCrop(256))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])) # imagenet
        transform = transforms.Compose(transform_list)

        return transform(image).float()

    def img_to_grid(self, img, row,col):
        ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
        hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
        grid = [img[j:jj+1,i:ii+1,:] for j,jj in ww for i,ii in hh]
        return grid, len(ww), len(hh)

    def cropper(self, images, width, height):

        seq = iaa.Sequential([
            iaa.CropToFixedSize(width=width, height=height)
        ])

        return seq.augment_image(images)


### Dataloaders ###

# Dataloader for training on images
def loadDataValSplit(train_class_list, train_path, test_class_list, test_path, batch_size=12, num_workers=4): 
    'Loads data into generator object'
    all_images_array = np.array([])
    all_labels_array = np.array([])    

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(train_class_list)):
        img_list = [f for f in os.listdir(os.path.join(train_path, train_class_list[category])) ] 
        
        for i in range(len(img_list)):
            all_images_array = np.append(all_images_array, os.path.join(train_class_list[category], img_list[i]))
            all_labels_array = np.append(all_labels_array, category)       

    all_indexes = list(range(0, np.shape(all_images_array)[0]))

    random.Random(4).shuffle(all_indexes) # Use a seed to ensure the train/val split is always the same

    train_indexes = all_indexes[:(int(0.8*len(all_indexes)))] # 0.8 for tuning, but use all 1.0 for training the final model
    val_indexes = all_indexes[(int(0.8*len(all_indexes))):] # 0.8

    train_images_array = all_images_array[train_indexes]
    val_images_array = all_images_array[val_indexes]

    train_labels_array = all_labels_array[train_indexes]
    val_labels_array = all_labels_array[val_indexes]

    target = torch.from_numpy(train_labels_array.astype(np.int32))
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])

    weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Now create the dataloaders  
    train_dataset = ImageDataset(train_images_array, train_labels_array, train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weighted_sampler, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    val_dataset = ImageDataset(val_images_array, val_labels_array, train_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)


    test_images_array = np.array([])
    test_labels_array = np.array([])    

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(test_class_list)):
        test_img_list = [f for f in os.listdir(os.path.join(test_path, test_class_list[category])) ] 
        
        for i in range(len(test_img_list)):
            test_images_array = np.append(test_images_array, os.path.join(test_class_list[category], test_img_list[i]))
            test_labels_array = np.append(test_labels_array, category)       

    test_dataset = ImageDataset(test_images_array, test_labels_array, test_path, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return train_dataloader, val_dataloader, test_dataloader


### Helper functions ###

# We need this to ensure each worker has a different random seed
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)