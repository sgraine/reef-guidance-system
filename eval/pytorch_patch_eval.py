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

    class_list = ["No-Deploy","Coral","Deploy"]

    test_dataloader = loadTestSetOrig(class_list, "../CleanData/Evaluation/Patches/Combined-TestSplit", batch_size=12, num_workers=2)
   
    model_path = "models/pytorch/model-1745379497CKPT.pt"

    ###### mobilenet_v3_small #####
    model_load = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_load.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, len(class_list)))

    ###### EfficientNet-B0 #####
    # model_load = models.efficientnet_b0(pretrained=True)

    # # Get the number of features in the last layer
    # num_ftrs = model_load.classifier[1].in_features

    # # Replace the last layer (classifier) with a new one for 4 classes
    # model_load.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, len(class_list)))

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

    print("Accuracy:", acc_test.item())
    print("Precision and Recall (for No-Deploy, Coral, Deploy):", per_class_prec_test, per_class_recall_test)
    print("F1 Scores (for No-Deploy, Coral, Deploy):", per_class_f1_test)
    print("Average F1 Score:", f1_test.item())
    print("\nConfusion Matrix (Rows: Actual classes, Columns: Predicted classes):")
    print(conf_mat)
    print("\nNormalized Confusion Matrix (Percentages) (Rows: Actual classes, Columns: Predicted classes):")
    normalized_confusion_df = pd.DataFrame(conf_mat_norm.cpu(), index=class_list, columns=class_list)
    print(normalized_confusion_df.map(lambda x: f"{x * 100:.2f}%"))