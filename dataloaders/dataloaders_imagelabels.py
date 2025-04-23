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
    def __init__(self, image_path_array, labels_array, image_dir, augment=True, w_resize=1328, h_resize=760):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.augment = augment
        self.w_resize = w_resize
        self.h_resize = h_resize

    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).resize((self.w_resize,self.h_resize))
        image = np.array(image)
        image = np.expand_dims(image, 0)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        label_new = label.clone()  # Clone to avoid modifying the original tensor
        label_new[label == 1] = 0  # Change all 1s to 0
        label_new[label == 2] = 1  # Change all 2s to 1

        # Preprocess and augment data
        if self.augment == True:
            image = self.augmentor(image)

        image = np.squeeze(image)
        image_transformed = self.transform_func(image)

        return image_transformed, label_new

    def transform_func(self, image):
        'Transform into a pytorch Tensor'

        transform_list = []

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



# Test image dataset: no augmentations as for training
class TestImageDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir, w_resize=1328, h_resize=760):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.w_resize = w_resize
        self.h_resize = h_resize
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).resize((self.w_resize,self.h_resize))
        image_transformed = self.transform_func(image)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        return image_transformed, label

    def transform_func(self, image):
        'Transform into a pytorch Tensor'
        transform_list = []

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)

        return transform(image).float()

### Dataloaders ###

# Dataloader for training on images
def loadDataValSplit(class_list, train_path, batch_size=12, augment=False, num_workers=4, w_resize=1328, h_resize=760): 
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
    train_dataset = ImageDataset(train_images_array, train_labels_array, train_path, augment=augment, w_resize=w_resize, h_resize=h_resize)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weighted_sampler, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    val_dataset = ImageDataset(val_images_array, val_labels_array, train_path, augment=False, w_resize=w_resize, h_resize=h_resize)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return train_dataloader, val_dataloader

# Dataloader for reading in the test patch dataset
def loadTestSetOrig(class_list, test_path, batch_size=12, num_workers=4, w_resize=1328, h_resize=760):
    'Loads data into generator object'
    test_images_array = np.array([])
    test_labels_array = np.array([])          

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(test_path, class_list[category])) if ( (f.endswith(".jpg") or f.endswith(".png")) )]
        
        for i in range(len(img_list)):
            test_images_array = np.append(test_images_array, os.path.join(class_list[category], img_list[i]))
            test_labels_array = np.append(test_labels_array, category)         
   
    else:
        # Now create the dataloaders  
        test_dataset = TestImageDataset(test_images_array, test_labels_array, test_path, w_resize=w_resize, h_resize=h_resize)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return test_dataloader


### Helper functions ###

# We need this to ensure each worker has a different random seed
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)