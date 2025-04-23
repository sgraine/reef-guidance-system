import os
from PIL import Image
import numpy as np
import pandas as pd
# import onnxruntime as ort
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import imgaug.augmenters as iaa
import torchmetrics

print("Imports done")

### Helper Functions ###
def transform_func(image):
    'Transform into a pytorch Tensor'

    transform_list = []

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])) # imagenet values
    transform = transforms.Compose(transform_list)

    return transform(image).float()

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

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

    model_path = "models/pytorch/model-1745366065CKPT.pt"

    model_load = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_load.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, 2))

    model_load.load_state_dict(torch.load(model_path))

    model_load = model_load.to(device)
    model_load.eval()

    # Class names (folders) in your dataset
    class_names = ['No-Deploy','Deploy']  # Replace with your actual class names

    input_folder = '../CleanData/Evaluation/Deployment/Combined'

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


    label_dict = {}
    inference_dict = {} 

    for class_num in range(0,2):
        step = 0
        class_name = class_names[class_num]

        current_path = os.path.join(input_folder, class_name)
        # current_path = input_folder        

        all_images = [i for i in os.listdir(current_path)]

        print("Processing images...")
        for filename in tqdm(all_images):

            image = Image.open(os.path.join(current_path, filename)).resize((1328,760))
            image_torch = transform_func(image).to(device)
            image_torch = torch.unsqueeze(image_torch, dim=0)
            
            vis_image = Image.open(os.path.join(current_path, filename))
            vis_image = np.array(vis_image)

            with torch.no_grad(): 
                outputs = model_load(image_torch)
                soft = torch.nn.Softmax(dim=1)
                outputs_soft = soft(outputs)
                outputs_pred = torch.argmax(outputs_soft, dim=1).int()
            
            preds_torch = outputs_pred
            labels_torch = torch.tensor([class_num]).int().to(device)

            acc_test = acc_metric_test(preds_torch, labels_torch)
            per_class_prec_test = per_class_prec_metric_test(preds_torch, labels_torch)
            per_class_recall_test = per_class_recall_metric_test(preds_torch, labels_torch)
            per_class_f1_test = F1_score_per_class(preds_torch, labels_torch)
            conf_mat = conf_matrix(preds_torch, labels_torch)
            conf_mat_norm = conf_matrix_norm(preds_torch, labels_torch)

            break
        break

    acc_test = acc_metric_test.compute()
    per_class_prec_test = per_class_prec_metric_test.compute()
    per_class_recall_test = per_class_recall_metric_test.compute()
    per_class_f1_test = F1_score_per_class.compute()
    f1_test = torch.mean(per_class_f1_test)
    conf_mat = conf_matrix.compute()
    conf_mat_norm = conf_matrix_norm.compute()

    print("Accuracy:", acc_test.item())
    print("Precision and Recall (for No-Deploy, Deploy):", per_class_prec_test, per_class_recall_test)
    print("F1 Scores (for No-Deploy, Deploy):", per_class_f1_test)
    print("Average F1 Score:", f1_test.item())
    print("\nConfusion Matrix (Rows: Actual classes, Columns: Predicted classes):")
    print(conf_mat)
    print("\nNormalized Confusion Matrix (Percentages) (Rows: Actual classes, Columns: Predicted classes):")
    normalized_confusion_df = pd.DataFrame(conf_mat_norm.cpu(), index=class_names, columns=class_names)
    print(normalized_confusion_df.map(lambda x: f"{x * 100:.2f}%"))
    
