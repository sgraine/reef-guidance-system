# Import packages
import os, time, random
from torchvision.models import mobilenet_v3_small
from torchvision.ops import sigmoid_focal_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dataloaders import loadDataValSplit#, loadTestSetOrig
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import wandb
import torchmetrics
from dataloaders.dataloaders_imagelabels import loadDataValSplit
#from torchsummary import summary
import numpy as np
import imgaug.augmenters as iaa
import argparse

# We need this to ensure each worker has a different random seed
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def labels_to_image(labels, H, W, n_rows, n_cols, discrete=True):
    B = labels.shape[0]
    patch_h = H // n_rows
    patch_w = W // n_cols

    img = labels.view(B, n_rows, n_cols)

    if discrete:
        intensity_map = torch.tensor([0, 122, 254], dtype=torch.uint8, device=labels.device)
        img = intensity_map[img]
        img = img.repeat_interleave(patch_h, dim=1)
        img = img.repeat_interleave(patch_w, dim=2)
        img = img.unsqueeze(1).float()
    else:
        # probabilities in [0,1]
        img = img.repeat_interleave(patch_h, dim=1)
        img = img.repeat_interleave(patch_w, dim=2)
        img = img.unsqueeze(1)  # keep float

    # resize to width 640 (same as before)
    W1 = 640
    scale = W1 / W
    H1 = int(H * scale)

    img = F.interpolate(img, size=(H1, W1), mode='nearest')

    return img


def labels_to_grid(labels, n_rows, n_cols):
    B = labels.shape[0]
    img = labels.view(B, 1, n_rows, n_cols)
    return img


# Patch dataset: this dataset randomly chooses samples for each batch and performs data augmentation for training
class PatchDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        
        self.transform = transforms.Compose([transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
            				      transforms.CenterCrop(256),
            				      transforms.ToTensor(),
            				      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path)
        
        #vis_image = Image.open(os.path.join(self.image_dir, filename))
        vis_image = np.array(image)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        label_new = label.clone()  # Clone to avoid modifying the original tensor
           
        row = 4
        col = 7

        width = int(vis_image.shape[1])
        height = int(vis_image.shape[0])

        # Divide the full image into a grid of patches
        grid, _, _ = self.img_to_grid(vis_image,row,col)
        
        patch_h = height // row
        patch_w = width // col
        
        all_patches = [self.transform(Image.fromarray(patch[:patch_h, :patch_w])) for patch in grid]
        all_patches_torch = torch.stack(all_patches)

        return all_patches_torch, label_new

    def transform_func(self, image):
        'Transform into a pytorch Tensor'
        return self.transform(image).float()

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

# Dataloader for training SeaCLIP on patches already labeled by CLIP
def loadDataValSplit(class_list, train_path, batch_size=12, efficientnet=False, augment=False, num_workers=4): #435
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

    train_labels_tensor = torch.from_numpy(train_labels_array.astype(np.int64))
    class_counts = torch.bincount(train_labels_tensor)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels_tensor]

    weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    if efficientnet:
        # Now create the dataloaders  
        train_dataset = EfficientPatchDataset(train_images_array, train_labels_array, train_path, num_classes=len(class_list), augment=augment)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

        val_dataset = EfficientPatchDataset(val_images_array, val_labels_array, train_path, num_classes=len(class_list), augment=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    else:
        # Now create the dataloaders  
        train_dataset = PatchDataset(train_images_array, train_labels_array, train_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weighted_sampler, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

        val_dataset = PatchDataset(val_images_array, val_labels_array, train_path)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return train_dataloader, val_dataloader

class CNN_MLP(nn.Module):
    def __init__(self):
        super(CNN_MLP, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(64, 1)
        )



    def forward(self, x):
        if x.dim() == 3:  
            x = x.unsqueeze(1)  # Add batch dim if missing
        
        x = x.float()
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPS results.')
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to save')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="dgs-mlp")

    dataset_path = args.train_data
    class_list = ["No-Deploy", "Deploy"]

    epochs = 100
    learning_rate = 0.00025 #0.0001
    batch_size = 32
    augment = True

    wandb.config.batch_size = batch_size

    # train_dataloader, val_dataloader = loadDataValSplit(class_list, os.path.join(dataset_path, "train-patches"), batch_size, efficientnet=True, augment=augment, num_workers=1)
    train_dataloader, val_dataloader = loadDataValSplit(class_list, os.path.join(dataset_path, "ImageWeakLabelling"), batch_size, efficientnet=False, augment=augment, num_workers=8)
   
    class_list = ["No-Deploy","Deploy"]

    ### Load patch classifier ###
    model_load = mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_load.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, 3))

    model_name = args.model_name
    model_path = os.path.join(args.output_folder, f"{model_name}.pt")

    model_load.load_state_dict(torch.load(model_path, weights_only=True))

    for p in model_load.parameters():
        p.requires_grad = False

    model_load.eval()
    model_load.to(device)

    ### Prepare deploy model ###

    deploy_model = CNN_MLP()
    deploy_model.train()
    deploy_model.to(device)

    for param in deploy_model.parameters():
        param.requires_grad = True

    # Loss and optimizer
    num_neg = 2000.0
    num_pos = 1000.0
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = optim.Adam(deploy_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr = 0.00003)

    ##### Accuracy Metrics #####
    acc_metric_train = torchmetrics.classification.BinaryAccuracy().to(device)
    acc_metric_val = torchmetrics.classification.BinaryAccuracy().to(device)

    ##### Precision Metrics #####
    prec_metric_train = torchmetrics.classification.BinaryPrecision().to(device)
    prec_metric_val = torchmetrics.classification.BinaryPrecision().to(device)

    ##### Recall Metrics #####
    recall_metric_train = torchmetrics.classification.BinaryRecall().to(device)
    recall_metric_val = torchmetrics.classification.BinaryRecall().to(device)

    F1_score = torchmetrics.classification.BinaryF1Score().to(device)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader

    #model_name = f"model-{int(time.time())}"
    print(f"################################# MODEL NAME: {model_name} ########################################")
    wandb.config.model_name = model_name

    best_acc = -1.0
    best_f1 = 0.0
    min_loss = 1000    
    for epoch in range(epochs):

        metrics = {}

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        ####################### TRAINING PHASE #########################
        phase = 'train'
        print("*** Phase: "+phase+" ***")

        deploy_model.train()
        running_loss_train = 0.0       

        for inputs, labels in tqdm(dataloaders[phase]):
            optimizer.zero_grad()
            
            #print(f'Input shape {inputs.shape}')  

            inputs = inputs.float().to(device)                # [B, 3, H, W]
            #print(f'\nLabels before processing: {labels}')
            labels = labels.to(device)                            # [B, H, W]

            B, P, C, H, W = inputs.shape
            inputs_flat = inputs.view(B * P, C, H, W)

            #print(inputs.shape, labels.shape)
            with torch.set_grad_enabled(False):
            	patch_logits = model_load(inputs_flat).detach()   # Forward through patch classifier
            	patch_probs = torch.softmax(patch_logits, dim=1) # Convert to probabilities
            	coral_probs = patch_probs[:, 0]       # [B*P]
            	deploy_probs = patch_probs[:, 1]      # [B*P]
            	nodeploy_probs = patch_probs[:, 2]    # [B*P]
            	
            	coral_probs = coral_probs.view(B, P)  # Reshape into [B, P]
            	deploy_probs = deploy_probs.view(B, P)  # Reshape into [B, P]
            	nodeploy_probs = nodeploy_probs.view(B, P)  # Reshape into [B, P]

            	coral_grid = labels_to_grid(labels=coral_probs, n_rows=4, n_cols=7)
            	deploy_grid = labels_to_grid(labels=deploy_probs, n_rows=4, n_cols=7)
            	nodeploy_grid = labels_to_grid(labels=nodeploy_probs, n_rows=4, n_cols=7)
            	
            	# stack along channel dimension -> [B, 3, n_rows, n_cols]
            	class_img = torch.cat([coral_grid, deploy_grid, nodeploy_grid], dim=1)
            	

            # forward pass - track history if only in train
            with torch.set_grad_enabled(True):
                outputs = deploy_model(class_img)
                labels_torch = labels.float().unsqueeze(1)

                #print(outputs, labels_torch)
                #loss = criterion(outputs, labels_torch)
                loss = sigmoid_focal_loss(inputs=outputs, targets=labels_torch, alpha=0.5, gamma=1.5, reduction='mean')
                loss.backward()
                optimizer.step() 

                preds_torch = (torch.sigmoid(outputs) > 0.5).int()
                labels_torch = labels_torch.int()
                
                #print("\noutputs:", outputs.detach().cpu().numpy().round(3).T)
                #print("\npreds_torch: ", preds_torch.detach().cpu().numpy().T)
                #print("labels_torch:", labels_torch.detach().cpu().numpy().T)
                #print("batch acc:", (preds_torch == labels_torch).float().mean().item())
                
                acc_metric_train.update(preds_torch.detach(), labels_torch.detach())
                prec_metric_train.update(preds_torch.detach(), labels_torch.detach())
                recall_metric_train.update(preds_torch.detach(), labels_torch.detach())

            
            running_loss_train = running_loss_train + loss.detach()

        acc_train = acc_metric_train.compute()
        prec_train = prec_metric_train.compute()
        recall_train = recall_metric_train.compute()
        
        print(f'acc_train: {acc_train.item()} | f1_train: {2 * (prec_train.item() * recall_train.item())/(prec_train.item() + recall_train.item()) }')
        print(f'prec_train: {prec_train.item()} | recall_train: {recall_train.item()}')   

        wandb.log({"train_acc": acc_train.item(), "train_prec": prec_train.item(),
                   "train_recall": recall_train.item()}, step=epoch)

        train_loss = running_loss_train / len(dataloaders[phase])

        metrics[phase+'_loss'] = train_loss.item()
        metrics[phase+'_pa'] = acc_train.item()


        ####################### VALIDATION PHASE #########################
        phase = 'val'
        print("*** Phase: "+phase+" ***")

        deploy_model.eval()
        running_loss_val = 0.0

        for inputs, labels in tqdm(dataloaders[phase]):

            inputs = inputs.float().to(device)             # [B, 3, H, W]
            labels = labels.to(device)                            # [B, H, W]

            B, P, C, H, W = inputs.shape
            inputs_flat = inputs.view(B * P, C, H, W)

            # forward pass - track history if only in train
            with torch.set_grad_enabled(False):
                patch_logits = model_load(inputs_flat).detach()   # Forward through patch classifier
                patch_probs = torch.softmax(patch_logits, dim=1) # Convert to probabilities
                
                # [B*P]
                coral_probs = patch_probs[:, 0]
                deploy_probs = patch_probs[:, 1]
                nodeploy_probs = patch_probs[:, 2]
                
                # Reshape into [B, P]
                coral_probs = coral_probs.view(B, P)
                deploy_probs = deploy_probs.view(B, P)
                nodeploy_probs = nodeploy_probs.view(B, P)
                
                coral_grid = labels_to_grid(labels=coral_probs, n_rows=4, n_cols=7)
                deploy_grid = labels_to_grid(labels=deploy_probs, n_rows=4, n_cols=7)
                nodeploy_grid = labels_to_grid(labels=nodeploy_probs, n_rows=4, n_cols=7)
                
                # stack along channel dimension -> [B, 3, n_rows, n_cols]
                class_img = torch.cat([coral_grid, deploy_grid, nodeploy_grid], dim=1)
                outputs = deploy_model(class_img)
                labels_torch = labels.float().unsqueeze(1)

                #loss = criterion(outputs, labels_torch)
                loss = sigmoid_focal_loss(inputs=outputs, targets=labels_torch, alpha=0.5, gamma=1.5, reduction='mean')

                preds_torch = (torch.sigmoid(outputs) > 0.5).int()
                labels_torch = labels_torch.int()
                acc_metric_val.update(preds_torch.detach(), labels_torch.detach())
                prec_metric_val.update(preds_torch.detach(), labels_torch.detach())
                recall_metric_val.update(preds_torch.detach(), labels_torch.detach())

                F1_score.update(preds_torch.detach(), labels_torch.detach())
            
            running_loss_val = running_loss_val + loss.detach()

        val_loss = running_loss_val / len(dataloaders[phase])

        acc_val = acc_metric_val.compute()
        prec_val = prec_metric_val.compute()
        recall_val = recall_metric_val.compute()

        f1_val = F1_score.compute()
        print(f'acc_val: {acc_val.item()} | f1_val: {f1_val.item()}')
        print(f'val_prec: {prec_val.item()} | val_recall: {recall_val.item()}')
        wandb.log({"val_acc": acc_val.item(), "val_prec": prec_val.item(), 
                   "val_recall": recall_val.item(), "val_f1": f1_val.item()}, step=epoch)

        metrics[phase+'_loss'] = val_loss.item()
        metrics[phase+'_pa'] = acc_val.item()

        scheduler.step(val_loss.item()) 

        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
        wandb.log({"lr":curr_lr}, step=epoch)

        #if acc_val > best_acc:
        if f1_val > best_f1:
            #best_acc = acc_val
            best_f1 = f1_val
            torch.save(deploy_model.state_dict(), os.path.join(args.output_folder, f"{model_name}_best.pt"))

        if epoch % 10 == 0:
            torch.save(deploy_model.state_dict(), os.path.join(args.output_folder, f"{model_name}_last.pt"))

        acc_metric_train.reset()
        prec_metric_train.reset()
        recall_metric_train.reset()

        acc_metric_val.reset()
        prec_metric_val.reset()
        recall_metric_val.reset()

        F1_score.reset()

        wandb.log(metrics, step=epoch)

    # Save the final model weights
    torch.save(deploy_model.state_dict(), os.path.join(args.output_folder, f"{model_name}_last.pt"))
