# Import packages
import os, time, random
from pathlib import Path
from torchvision.models import mobilenet_v3_small
from torchvision.ops import sigmoid_focal_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import wandb
import torchmetrics
import numpy as np
import pandas as pd
import argparse

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

        return all_patches_torch, label_new, image_path

    def transform_func(self, image):
        'Transform into a pytorch Tensor'
        return self.transform(image).float()

    def img_to_grid(self, img, row,col):
        ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
        hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
        grid = [img[j:jj+1,i:ii+1,:] for j,jj in ww for i,ii in hh]
        return grid, len(ww), len(hh)

# Dataloader for training SeaCLIP on patches already labeled by CLIP
def loadTestData(class_list, train_path, batch_size=12, num_workers=4): #435
    'Loads data into generator object'
    all_images_array = np.array([])
    all_labels_array = np.array([])    

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(train_path, class_list[category])) ] 
        
        for i in range(len(img_list)):
            all_images_array = np.append(all_images_array, os.path.join(class_list[category], img_list[i]))
            all_labels_array = np.append(all_labels_array, category)       

    test_dataset = PatchDataset(all_images_array, all_labels_array, train_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return test_dataloader

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
    parser = argparse.ArgumentParser(description='Script for evaluating image classification model at patch level.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the model weights file')
    return parser.parse_args()

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()
    dataset_path = args.dataset_path
    class_list = ["No-Deploy", "Deploy"]

    batch_size = 10
    
    ### Load patch classifier ###
    model_class = mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_class.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_class.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, 3))

    class_model_path = args.model_weights

    model_class.load_state_dict(torch.load(class_model_path, weights_only=True))
    for p in model_class.parameters():
        p.requires_grad = False
    model_class.eval()
    model_class.to(device)

    ### Prepare deploy model ###
    model_name_deploy = "CNNv4best"
    deploy_model_path = f"outputs/models/emix/{model_name_deploy}.pt"

    deploy_model = CNN_MLP()
    deploy_model.load_state_dict(torch.load(deploy_model_path, weights_only=True))
    for p in deploy_model.parameters():
        p.requires_grad = False
    deploy_model.eval()
    deploy_model.to(device)

    threshold = 0.5

    print(f"################################# MODEL NAME: {model_name_deploy} ########################################")
    
    folder_names = ["Combined"]
    #folder_names = ["Site2/Sequence1"]
    #folder_names = ["Site2/Sequence1", "Site2/Sequence2", "Site3/Sequence1", "Site4/Sequence1", "Site4/Sequence2", "Site5/Sequence1", "Site5/Sequence2", "Site5/Sequence3", "Site5/Sequence4"]
    #folder_names = ["Site2/Sequence1", "Site3/Sequence1", "Site4/Sequence1", "Site4/Sequence2", "Site5/Sequence1", "Site5/Sequence3", "Site5/Sequence4"]
    avg_acc = 0.0
    avg_prec = 0.0
    avg_rec = 0.0
    deploy_folder = []
    nodeploy_folder = []
    for folder_name in folder_names: 
        test_dataloader = loadTestData(class_list, os.path.join(dataset_path, folder_name), batch_size, num_workers=8)

        ##### Metrics #####
        #'''
        acc_metric_val = torchmetrics.classification.BinaryAccuracy().to(device)
        prec_metric_val = torchmetrics.classification.BinaryPrecision().to(device)
        recall_metric_val = torchmetrics.classification.BinaryRecall().to(device)
        F1_score = torchmetrics.classification.BinaryF1Score().to(device)
        '''
        acc_metric_val = torchmetrics.classification.MulticlassAccuracy(num_classes=2, average="macro").to(device)
        prec_metric_val = torchmetrics.classification.MulticlassPrecision(num_classes=2, average="macro").to(device)
        recall_metric_val = torchmetrics.classification.MulticlassRecall(num_classes=2, average="macro").to(device)
        F1_score = torchmetrics.classification.MulticlassF1Score(num_classes=2, average="macro").to(device)
        '''

        conf_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=len(class_list), normalize='none').to(device)
        conf_matrix_norm = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=len(class_list), normalize='true').to(device)

        
        for inputs, labels, metadata in tqdm(test_dataloader):

            inputs = inputs.float().to(device)             # [B, 3, H, W]
            labels = labels.to(device)                            # [B, H, W]

            B, P, C, H, W = inputs.shape
            inputs_flat = inputs.view(B * P, C, H, W)

            # forward pass - track history if only in train
            with torch.set_grad_enabled(False):
                patch_logits = model_class(inputs_flat).detach()   # Forward through patch classifier
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


                preds_torch = (torch.sigmoid(outputs) > threshold).int()

                numpy_preds = np.squeeze(preds_torch.detach().cpu().numpy()).astype(int)
                metadata = np.array(metadata).T
                deploy_folder.extend(metadata[numpy_preds > threshold])
                nodeploy_folder.extend(metadata[numpy_preds <= threshold])

                labels_torch = labels_torch.int()
                acc_metric_val.update(preds_torch.detach(), labels_torch.detach())
                prec_metric_val.update(preds_torch.detach(), labels_torch.detach())
                recall_metric_val.update(preds_torch.detach(), labels_torch.detach())
                F1_score.update(preds_torch.detach(), labels_torch.detach())
                conf_matrix.update(preds_torch.detach(), labels_torch.detach())
                conf_matrix_norm.update(preds_torch.detach(), labels_torch.detach())

        acc_val = acc_metric_val.compute()
        prec_val = prec_metric_val.compute()
        recall_val = recall_metric_val.compute()
        f1_val = F1_score.compute()
        conf_mat = conf_matrix.compute()
        conf_mat_norm = conf_matrix_norm.compute()

        avg_acc  += acc_val.item()
        avg_prec += prec_val.item()
        avg_rec  += recall_val.item()
        

        print("Site:", folder_name)
        print("Deployment Threshold:", threshold)
        print("Accuracy:", acc_val.item())
        print("Precision and Recall (for No-Deploy, Deploy):", prec_val, recall_val)
        print("F1 Scores (for No-Deploy, Deploy):", f1_val)
        print("\nConfusion Matrix (Rows: Actual classes, Columns: Predicted classes):")
        print(conf_mat)
        print("\nNormalized Confusion Matrix (Percentages) (Rows: Actual classes, Columns: Predicted classes):")
        normalized_confusion_df = pd.DataFrame(conf_mat_norm.cpu(), index=class_list, columns=class_list)
        print(normalized_confusion_df.map(lambda x: f"{x * 100:.2f}%"))


    mat = conf_mat.cpu().numpy()
    w_deploy = 0.28475
    w_nodeploy = 0.71525

    print("##########  MANUAL NO-DEPLOY COMPUTATION  ########")
    tp_nodeploy = mat[0, 0]
    fp_nodeploy = mat[1, 0]
    fn_nodeploy = mat[0, 1]
    prec_nodeploy = tp_nodeploy / (tp_nodeploy + fp_nodeploy) if tp_nodeploy + fp_nodeploy > 0.0 else 0.0
    rec_nodeploy = tp_nodeploy / (tp_nodeploy + fn_nodeploy) if tp_nodeploy + fn_nodeploy > 0.0 else 0.0
    f1_nodeploy = 2.0 * prec_nodeploy * rec_nodeploy / (prec_nodeploy + rec_nodeploy) if prec_nodeploy + rec_nodeploy > 0.0 else 0.0  
    print("Precision:", prec_nodeploy)
    print("Recall:", rec_nodeploy)
    print("F1 Scores:", f1_nodeploy)

    print("##########  MANUAL DEPLOY COMPUTATION  ########")
    tp_deploy = mat[1, 1]
    fp_deploy = mat[0, 1]
    fn_deploy = mat[1, 0]
    prec_deploy = tp_deploy / (tp_deploy + fp_deploy) if tp_deploy + fp_deploy > 0.0 else 0.0
    rec_deploy =  tp_deploy / (tp_deploy + fn_deploy) if tp_deploy + fn_deploy > 0.0 else 0.0
    f1_deploy = 2.0 * prec_deploy * rec_deploy / (prec_deploy + rec_deploy) if prec_deploy + rec_deploy > 0.0 else 0.0 
    beta = 0.5
    beta_sq = beta * beta
    f1b_deploy = (1.0 + beta_sq) * prec_deploy * rec_deploy / (beta_sq * prec_deploy + rec_deploy) 
    
    print("Precision:", prec_deploy)
    print("Recall:", rec_deploy)
    print("F1 Scores:", f1_deploy)
    print("F1-Beta Scores:", f1b_deploy)

    print("##########  MANUAL WEIGHTED AVERAGE  ########")
    print("Precision:", w_deploy * prec_deploy + w_nodeploy * prec_nodeploy)
    print("Recall:", w_deploy * rec_deploy + w_nodeploy * rec_nodeploy)
    print("F1 Scores:", w_deploy * f1_deploy + w_nodeploy * f1_nodeploy)
    
    print("##########  PYTORCH VALS  ########")
    avg_acc  = avg_acc / len(folder_names)
    avg_prec = avg_prec / len(folder_names)
    avg_rec  = avg_rec / len(folder_names)
    print("Accuracy:", avg_acc)
    print("Precision Deploy:", avg_prec)
    print("Recall Deploy:", avg_rec)
    print("F1 Scores:", 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec))

    root_path = args.output_path
    save_path = os.path.join(root_path, "GPSResults/PatchCNN_0_5")
    deploy_save_path = os.path.join(save_path, "Deploy")
    for deploy_path in tqdm(deploy_folder):
        img_name = Path(deploy_path).name
        image = Image.open(deploy_path)

        # Check if the image has EXIF metadata
        exif_data = image.info.get('exif')

        output_path = os.path.join(deploy_save_path, img_name)
        image.save(output_path, exif=exif_data)

    nodeploy_save_path = os.path.join(save_path, "No-Deploy")
    nodeploy_folder = np.array(nodeploy_folder)
    print(f"NoDeploy Path: {nodeploy_save_path}")
    for nodeploy_path in tqdm(nodeploy_folder):
        img_name = Path(nodeploy_path).name
        image = Image.open(nodeploy_path)

        # Check if the image has EXIF metadata
        exif_data = image.info.get('exif')

        output_path = os.path.join(nodeploy_save_path, img_name)
        image.save(output_path, exif=exif_data)
