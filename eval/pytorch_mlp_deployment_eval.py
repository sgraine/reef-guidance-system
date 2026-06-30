import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import imgaug.augmenters as iaa
import torchmetrics
import argparse

from pathlib import Path

from vis.video_frames import video_frame

print("Imports done")


# --- Image-level model (spatial aggregation of patch predictions) ---
class PatchGridClassifier(nn.Module):
    def __init__(self, patch_class_dim, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.conv_model = nn.Sequential(
            nn.Conv2d(patch_class_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, patch_preds):
        B, N, C = patch_preds.shape
        gh, gw = self.grid_size
        x = patch_preds.view(B, gh, gw, C).permute(0, 3, 1, 2)  # [B, C, gh, gw]
        return self.conv_model(x)
    
### Data Transformation Functions ###
def transform_func(image):
    'Transform into a pytorch Tensor'
    transform_list = []
    transform_list.append(transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC))  # this is for full size (original) model
    transform_list.append(transforms.CenterCrop(256))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])) # imagenet
    transform = transforms.Compose(transform_list)

    return transform(image).float()

### Helper Functions ###
def img_to_grid(img, row,col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
    grid = [img[j:jj+1,i:ii+1,:] for j,jj in ww for i,ii in hh]
    return grid, len(ww), len(hh)

def cropper(images, width, height):

    seq = iaa.Sequential([
        iaa.CropToFixedSize(width=width, height=height)
    ])

    return seq.augment_image(images)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for evaluating mlp classification model at patch level.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--model1_weights', type=str, required=True, help='Path to the model1 weights file')
    parser.add_argument('--model2_weights', type=str, required=True, help='Path to the model2 weights file')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.model1_weights

    ######################## LOAD PATCH MODEL ########################
    ###### mobilenet_v3_small #####
    patch_model =models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = patch_model.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 2 classes
    patch_model.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, 3))

    patch_model.load_state_dict(torch.load(model_path))

    patch_model = patch_model.to(device)
    patch_model.eval()

    for param in patch_model.parameters():
        param.requires_grad = False

    ######################## CREATE MAPPING MODEL ########################
    model_path = args.model2_weights
    file_path = Path(model_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    patch_class_dim = patch_model(torch.randn(1, 3, 256, 256).to(device)).shape[1]
    new_model = PatchGridClassifier(patch_class_dim, (7, 4))

    new_model.load_state_dict(torch.load(model_path)) 

    new_model = new_model.to(device)
    new_model.eval()

    for param in new_model.parameters():
        param.requires_grad = False

    print("Models loaded.")

    input_folder = args.dataset_path
    output_folder = args.output_path
    
    # Class names (folders) in your dataset
    class_names = ['No-Deploy','Deploy'] 

    thresholds = [0.3] #, 0.4, 0.5, 0.6, 0.7, 0.8

    for threshold in thresholds:
        ##### Accuracy Metrics #####
        acc_metric_train = torchmetrics.classification.BinaryAccuracy().to(device)
        acc_metric_val = torchmetrics.classification.BinaryAccuracy().to(device)

        ##### Precision Metrics #####
        per_class_prec_metric_train = torchmetrics.classification.BinaryPrecision().to(device)
        per_class_prec_metric_val = torchmetrics.classification.BinaryPrecision().to(device)

        ##### Recall Metrics #####
        per_class_recall_metric_train = torchmetrics.classification.BinaryRecall().to(device)
        per_class_recall_metric_val = torchmetrics.classification.BinaryRecall().to(device)

        F1_score_metric_train = torchmetrics.classification.BinaryF1Score().to(device)
        F1_score_metric_val = torchmetrics.classification.BinaryF1Score().to(device)

        conf_matrix = torchmetrics.classification.BinaryConfusionMatrix(normalize='none').to(device)
        conf_matrix_norm = torchmetrics.classification.BinaryConfusionMatrix(normalize='true').to(device)

        for class_num in range(0,2):

            class_name = class_names[class_num]

            current_path = os.path.join(input_folder, class_name)    

            all_images = [i for i in os.listdir(current_path)]

            step = 0
            print("Processing images...")
            for filename in tqdm(all_images):
                
                vis_image = Image.open(os.path.join(current_path, filename)).convert('RGB')
                vis_image = np.array(vis_image)

                row = 4
                col = 7

                width = int(vis_image.shape[1])
                height = int(vis_image.shape[0])

                # Divide the full image into a grid of patches
                grid, _, _ = img_to_grid(vis_image,row,col)

                all_patches = []
                for patch in grid:
                    patch_crop = cropper(patch, int(np.floor(width / col)), int(np.floor(height / row)))
                    all_patches.append(torch.unsqueeze(transform_func(Image.fromarray(patch_crop)), dim=0))

                all_patches_torch = torch.cat(all_patches, dim=0).to(device)

                inputs = torch.squeeze(all_patches_torch, dim=0)
                

                # forward pass - track history if only in train
                with torch.set_grad_enabled(False):
                    patch_preds = patch_model(inputs)
                    outputs = new_model(torch.unsqueeze(patch_preds, 0))

                    sigmoid_func = torch.nn.Sigmoid()
                    outputs = sigmoid_func(outputs)
                    preds_torch = (outputs >= threshold)[0].int()

                    soft_batch = torch.nn.Softmax(dim=1)
                    patches_soft =  soft_batch(patch_preds)
                    outputs_patch_preds = torch.argmax(patches_soft, dim=1).int()


                labels_torch = torch.tensor([class_num]).int().to(device)

                acc_train = acc_metric_train(preds_torch, labels_torch)
                per_class_prec_train = per_class_prec_metric_train(preds_torch, labels_torch)
                per_class_recall_train = per_class_recall_metric_train(preds_torch, labels_torch)
                per_class_f1_train = F1_score_metric_train(preds_torch, labels_torch)

                conf_mat = conf_matrix(preds_torch, labels_torch)
                conf_mat_norm = conf_matrix_norm(preds_torch, labels_torch)

                video_frame(outputs_patch_preds.detach().cpu().numpy(), preds_torch, vis_image, os.path.join(output_folder, class_name, filename[:-4]))
                

        print("#####################################")
        print("Threshold =", threshold)

        acc_train = acc_metric_train.compute()
        per_class_prec_train = per_class_prec_metric_train.compute()
        per_class_recall_train = per_class_recall_metric_train.compute()
        per_class_f1_train = F1_score_metric_train.compute()

        conf_mat = conf_matrix.compute()
        conf_mat_norm = conf_matrix_norm.compute()

        print("Accuracy:", acc_train.item())
        print("Precision and Recall:", per_class_prec_train, per_class_recall_train)
        print("F1 Scores:", per_class_f1_train)
        print("Average F1 Score:", per_class_f1_train.item())

        print("\nConfusion Matrix (Rows: Actual classes, Columns: Predicted classes):")
        print(conf_mat)
        print("\nNormalized Confusion Matrix (Percentages) (Rows: Actual classes, Columns: Predicted classes):")
        normalized_confusion_df = pd.DataFrame(conf_mat_norm.cpu(), index=class_names, columns=class_names)
        print(normalized_confusion_df.map(lambda x: f"{x * 100:.2f}%"))