# Dataloaders for DGS dataset

# Import packages
import torch
import numpy as np
import os, random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

import imgaug.augmenters as iaa

### Dataset Classes ###

# Patch dataset: this dataset randomly chooses samples for each batch and performs data augmentation for training
class PatchDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir, num_classes=4, augment=True, cropsize=256):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.augment = augment
        self.cropsize = cropsize
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path)
        image = np.array(image)
        image = np.expand_dims(image, 0)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        # Preprocess and augment data
        if self.augment == True:
            image = self.augmentor(image)

        image = np.squeeze(image)
        image = Image.fromarray(image)
        image_transformed = self.transform_func(image)

        return image_transformed, label

    def transform_func(self, image):
        'Transform into a pytorch Tensor'

        transform_list = []
        transform_list.append(transforms.Resize((self.cropsize,self.cropsize), interpolation=transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.CenterCrop(self.cropsize))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])) # imagenet values
        transform = transforms.Compose(transform_list)

        return transform(image).float()

    def augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        often = lambda aug: iaa.Sometimes(0.7, aug)

        # Original augmentations
        seq = iaa.Sequential([            
            # Best Augmentation Strategy: Colour Augmentation
            iaa.Fliplr(0.5),
            often(
                iaa.WithChannels(0, iaa.Add((-30, 30))) # RGB = 0,1,2
                ),
            sometimes(
                iaa.LinearContrast((0.5, 2.0))
                ),
            sometimes(
                iaa.AddToBrightness((-30, 30))
                ),
            sometimes(
                iaa.GaussianBlur(sigma=(0,0.5))
                )
        ], random_order=True) # apply augmenters in random order
       
        return seq.augment_images(images)

# Test patch dataset: no augmentations as for training
class TestPatchDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir, cropsize=256):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.cropsize = cropsize
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path)
        image_transformed = self.transform_func(image)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        return image_transformed, label

    def transform_func(self, image):
        'Transform into a pytorch Tensor'
        transform_list = []

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Resize((self.cropsize,self.cropsize), interpolation=transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.CenterCrop(self.cropsize))
        transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)

        return transform(image).float()

### Dataloaders ###


# Dataloader for reading in the test patch dataset
def loadTestSetOrig(class_list, test_path, batch_size=12, num_workers=4):
    'Loads data into generator object'
    test_images_array = np.array([])
    test_labels_array = np.array([])          

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(test_path, class_list[category])) if ( (f.endswith(".jpg") or f.endswith(".png")) )]
        
        for i in range(len(img_list)):
            test_images_array = np.append(test_images_array, os.path.join(class_list[category], img_list[i]))
            test_labels_array = np.append(test_labels_array, category)         
    
    # Now create the dataloaders  
    test_dataset = TestPatchDataset(test_images_array, test_labels_array, test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return test_dataloader

# Dataloader for training on patches
def loadDataValSplit(class_list, train_path, batch_size=12, augment=False, num_workers=4, cropsize=256):
    'Loads data into generator object'
    all_images_array = np.array([])
    all_labels_array = np.array([])    

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(train_path, class_list[category])) ] 
        
        for i in range(len(img_list)):
            all_images_array = np.append(all_images_array, os.path.join(class_list[category], img_list[i]))
            all_labels_array = np.append(all_labels_array, category)       

    all_indexes = list(range(0, np.shape(all_images_array)[0]))

    random.Random(4).shuffle(all_indexes) # Use a seed to ensure the train/val split is always the same

    train_indexes = all_indexes[:(int(0.8*len(all_indexes)))] # 0.8
    val_indexes = all_indexes[(int(0.8*len(all_indexes))):] # 0.8

    train_images_array = all_images_array[train_indexes]
    val_images_array = all_images_array[val_indexes]

    train_labels_array = all_labels_array[train_indexes]
    val_labels_array = all_labels_array[val_indexes]

    ### Using a weighted over-sampler to mitigate class imbalance ###
    target = torch.from_numpy(train_labels_array.astype(np.int32))
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])

    weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Now create the dataloaders  
    train_dataset = PatchDataset(train_images_array, train_labels_array, train_path, num_classes=len(class_list), augment=augment, cropsize=cropsize)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weighted_sampler, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    val_dataset = PatchDataset(val_images_array, val_labels_array, train_path, num_classes=len(class_list), augment=False, cropsize=cropsize)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return train_dataloader, val_dataloader

### Helper functions ###

# We need this to ensure each worker has a different random seed
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)