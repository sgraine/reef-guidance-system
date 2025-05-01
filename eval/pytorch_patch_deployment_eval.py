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

print("Imports done")

### Data Transformation Functions ###
def transform_func(image):
    'Transform into a pytorch Tensor'
    transform_list = []
    transform_list.append(transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC))  # this is for full size (original) model
    transform_list.append(transforms.CenterCrop(256))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])) # imagenet
    transform = transforms.Compose(transform_list)

    return transform(image)

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

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = "outputs/models/pytorch/model-1745448701CKPT.pt"
    model_load = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_load.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, 3))

    model_load.load_state_dict(torch.load(model_path))

    model_load = model_load.to(device)
    model_load.eval()

    input_folder = '../CleanData/Evaluation/Deployment/Combined'
    # input_folder = '../Rosbag-images'

    # Class names (folders) in your dataset
    class_names = ['No-Deploy','Deploy'] 

    ##### Accuracy Metrics #####
    acc_metric_test = torchmetrics.Accuracy(num_classes = len(class_names), task='multiclass', multidim_average='global').to(device)

    ##### Precision Metrics #####
    per_class_prec_metric_test = torchmetrics.Precision(num_classes = len(class_names), average='none', task='multiclass', multidim_average='global').to(device)

    ##### Recall Metrics #####
    per_class_recall_metric_test = torchmetrics.Recall(num_classes = len(class_names), average='none', task='multiclass', multidim_average='global').to(device)
    
    ##### F1 Metrics #####
    F1_score_per_class = torchmetrics.F1Score(num_classes=len(class_names), average='none', task='multiclass', multidim_average='global').to(device)

    conf_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=len(class_names), normalize='none').to(device)
    conf_matrix_norm = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=len(class_names), normalize='true').to(device)

    for class_num in range(0,2):

        class_name = class_names[class_num]

        current_path = os.path.join(input_folder, class_name)    

        all_images = [i for i in os.listdir(current_path)]

        step = 0
        print("Processing images...")
        for filename in tqdm(all_images):
            
            vis_image = Image.open(os.path.join(current_path, filename))
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

            soft = torch.nn.Softmax(dim=0)

            with torch.no_grad(): 
                outputs_batch = model_load(all_patches_torch)
                soft_batch = torch.nn.Softmax(dim=1)
                outputs_batch_soft =  soft_batch(outputs_batch)
                outputs_batch_preds = torch.argmax(outputs_batch_soft, dim=1).int()

            # print(outputs_batch_preds)
            
            # Compute ratio
            zero_count = (outputs_batch_preds == 2).sum(dim=0) # 2 corresponds with Deploy class for patch classifier
            ratio = zero_count / outputs_batch_preds.shape[0]

            threshold = 0.5 # tunable hyperparameter
            deploy = (ratio > threshold).int()

            preds_torch = torch.tensor([deploy]).int().to(device)
            labels_torch = torch.tensor([class_num]).int().to(device)

            acc_test = acc_metric_test(preds_torch, labels_torch)
            per_class_prec_test = per_class_prec_metric_test(preds_torch, labels_torch)
            per_class_recall_test = per_class_recall_metric_test(preds_torch, labels_torch)
            per_class_f1_test = F1_score_per_class(preds_torch, labels_torch)
            conf_mat = conf_matrix(preds_torch, labels_torch)
            conf_mat_norm = conf_matrix_norm(preds_torch, labels_torch)

    acc_test = acc_metric_test.compute()
    per_class_prec_test = per_class_prec_metric_test.compute()
    per_class_recall_test = per_class_recall_metric_test.compute()
    per_class_f1_test = F1_score_per_class.compute()
    f1_test = torch.mean(per_class_f1_test)
    conf_mat = conf_matrix.compute()
    conf_mat_norm = conf_matrix_norm.compute()

    print("Deployment Threshold:", threshold)
    print("Accuracy:", acc_test.item())
    print("Precision and Recall (for No-Deploy, Deploy):", per_class_prec_test, per_class_recall_test)
    print("F1 Scores (for No-Deploy, Deploy):", per_class_f1_test)
    print("Average F1 Score:", f1_test.item())
    print("\nConfusion Matrix (Rows: Actual classes, Columns: Predicted classes):")
    print(conf_mat)
    print("\nNormalized Confusion Matrix (Percentages) (Rows: Actual classes, Columns: Predicted classes):")
    normalized_confusion_df = pd.DataFrame(conf_mat_norm.cpu(), index=class_names, columns=class_names)
    print(normalized_confusion_df.map(lambda x: f"{x * 100:.2f}%"))