# Import packages
import os
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
from dataloaders.dataloaders_patchlabels import loadTestSetOrig
from tqdm import tqdm
import torchmetrics
import pandas as pd
import wandb
import sys

print("Packages imported successfully.")

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def normalize_logits(logits):
    mean = logits.mean()
    std = logits.std()
    normalized_logits = (logits - mean) / std
    return normalized_logits

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project="DGS-Patch-Eval", entity="sgraine")
    # wandb.init(mode="disabled")

    class_list = ["No-Deploy","Coral","Deploy"]

    test_dataloader = loadTestSetOrig(class_list, "../CleanData/Evaluation/Patches/Combined-TestSplit", batch_size=12, num_workers=2)
   
    model_name = "1745448890CKPT"
    model_path = "outputs/models/pytorch/model-"+model_name+".pt" # mobilenet on ecologist patches
    wandb.config.model_path = model_path

    ###### mobilenet_v3_small #####
    model_load = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_load.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, len(class_list)))

    ###### Mobilenet_v3_large #####
    # model_load = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")
    # wandb.config.model_type = 'MobileNet_V3-LARGE'

    # # Get the number of features in the last layer
    # num_ftrs = model_load.classifier[3].in_features

    # # Replace the last layer (classifier) with a new one for 4 classes
    # model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, len(class_list)))

    ###### EfficientNet-B0 #####
    # model_load = models.efficientnet_b0(pretrained=True)

    # # Get the number of features in the last layer
    # num_ftrs = model_load.classifier[1].in_features

    # # Replace the last layer (classifier) with a new one for 4 classes
    # model_load.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, len(class_list)))

    ###### EfficientNet-B7 #####
    # model_load = models.efficientnet_b7(pretrained=False)

    # # Get the number of features in the last layer
    # num_ftrs = model_load.classifier[1].in_features

    # # Replace the last layer (classifier) with a new one for 4 classes
    # model_load.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, len(class_list)))

    ###### Resnet-18 #####
    # model_load = models.resnet18(pretrained=True)
    # model_load.fc = nn.Sequential(nn.Linear(512, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, len(class_list)))

    model_load.load_state_dict(torch.load(model_path))

    model_load = model_load.to(device)
    model_load.eval()

    print("Loaded the model successfully.")
   
    ##### Accuracy Metrics #####
    acc_metric_test = torchmetrics.Accuracy(num_classes = len(class_list), task='multiclass', multidim_average='global').to(device)

    ##### Precision Metrics #####
    per_class_prec_metric_test = torchmetrics.Precision(num_classes = len(class_list), average='none', task='multiclass', multidim_average='global').to(device)

    ##### Recall Metrics #####
    per_class_recall_metric_test = torchmetrics.Recall(num_classes = len(class_list), average='none', task='multiclass', multidim_average='global').to(device)
    
    ##### F1 Metrics #####
    F1_score_per_class = torchmetrics.F1Score(num_classes=len(class_list), average='none', task='multiclass', multidim_average='global').to(device)

    conf_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=len(class_list), normalize='none').to(device)
    conf_matrix_norm = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=len(class_list), normalize='true').to(device)

    dataloaders = {}
    dataloaders['test'] = test_dataloader

    phase = 'test'
    soft = torch.nn.Softmax(dim=1)

    step = 0
    preds_list = []
    labels_list = []

    print("Iterating through the test dataset...")
    for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.float().to(device)                  # [B, 3, H, W]
        labels = labels.to(device)                            # [B, H, W]

        # forward pass - track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model_load(torch.squeeze(inputs))  # [batch_size, num_classes]   
            outputs_total = soft(outputs) # [batch_size, num_classes]

            preds_torch = torch.argmax(outputs_total, dim=1).int()  # [1]

            labels_torch = labels.int()

            acc_test = acc_metric_test(preds_torch, labels_torch)
            per_class_prec_test = per_class_prec_metric_test(preds_torch, labels_torch)
            per_class_recall_test = per_class_recall_metric_test(preds_torch, labels_torch)
            per_class_f1_test = F1_score_per_class(preds_torch, labels_torch)
            conf_mat = conf_matrix(preds_torch, labels_torch)
            conf_mat_norm = conf_matrix_norm(preds_torch, labels_torch)

            preds_list.append(preds_torch.cpu().numpy())
            labels_list.append(labels_torch.cpu().numpy())

        # step += 1
        # if step == 5:
        #     break

    # Calculate metrics
    acc_test = acc_metric_test.compute()
    per_class_prec_test = per_class_prec_metric_test.compute()
    per_class_recall_test = per_class_recall_metric_test.compute()
    per_class_f1_test = F1_score_per_class.compute()
    f1_test = torch.mean(per_class_f1_test)
    conf_mat = conf_matrix.compute()
    conf_mat_norm = conf_matrix_norm.compute()

    with open("outputs/results/patch_results_"+model_name+".txt", "w") as f:
        sys.stdout = f  # Redirect print output to file

        print("Accuracy:", acc_test.item())
        print("Precision and Recall (for No-Deploy, Coral, Deploy):", per_class_prec_test, per_class_recall_test)
        print("F1 Scores (for No-Deploy, Coral, Deploy):", per_class_f1_test)
        print("Average F1 Score:", f1_test.item())
        print("\nConfusion Matrix (Rows: Actual classes, Columns: Predicted classes):")
        print(conf_mat)
        print("\nNormalized Confusion Matrix (Percentages) (Rows: Actual classes, Columns: Predicted classes):")
        normalized_confusion_df = pd.DataFrame(conf_mat_norm.cpu(), index=class_list, columns=class_list)
        print(normalized_confusion_df.map(lambda x: f"{x * 100:.2f}%"))

    metrics = {}
    metrics['accuracy'] = acc_test.item()
    metrics['mean f1'] = f1_test.item()

    metrics['no_dep_prec'] = per_class_prec_test[0]
    metrics['no_dep_rec'] = per_class_recall_test[0]
    metrics['no_dep_f1'] = per_class_f1_test[0]

    metrics['coral_prec'] = per_class_prec_test[1]
    metrics['coral_rec'] = per_class_recall_test[1]
    metrics['coral_f1'] = per_class_f1_test[1]

    metrics['deploy_prec'] = per_class_prec_test[2]
    metrics['deploy_rec'] = per_class_recall_test[2]
    metrics['deploy_f1'] = per_class_f1_test[2]

    wandb.log(metrics)