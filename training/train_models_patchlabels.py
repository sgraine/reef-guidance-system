# Import packages
import os, time
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, resnet18, efficientnet_b0
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloaders.dataloaders_patchlabels import loadDataValSplit
import torch.optim as optim
from tqdm import tqdm
import wandb
import torchmetrics
# from torchsummary import summary
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss).to(device)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # wandb.init(mode="disabled")
    wandb.init(project="DGS-April", entity="sgraine")

    dataset_path = "../CleanData/Training - CLIPPatchLabelling/Combined"
    # dataset_path = "../CleanData/Training - ChatGPTLabelling/PatchTrainSplit"
    # dataset_path = "../CleanData/Evaluation/Patches/Combined-TrainSplit" 
    wandb.config.dataset = dataset_path
    class_list = ["No-Deploy","Coral","Deploy"]

    epochs = 200
    learning_rate = 0.0001 
    batch_size = 96 
    augment = True

    wandb.config.batch_size = batch_size

    train_dataloader, val_dataloader = loadDataValSplit(class_list, dataset_path, batch_size, augment=augment, num_workers=1)

    ###### Mobilenet_v3_small #####
    model = mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
    wandb.config.model_type = 'MobileNet_V3'

    # Get the number of features in the last layer
    num_ftrs = model.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, len(class_list)))

    #%%%%%%%%%%%%%%%%%%% Alternate models %%%%%%%%%%%%%%%%%%%%%%%%%

    ###### Mobilenet_v3_large #####
    # model = mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")
    # wandb.config.model_type = 'MobileNet_V3-LARGE'
    # model.classifier[3] = nn.Linear(in_features=1280, out_features=len(class_list))
    
    # # ###### Resnet-18 #####
    # model = resnet18(pretrained=True)
    # wandb.config.model_type = 'resnet18'
    # model.fc = nn.Sequential(nn.Linear(512, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, len(class_list)))

    ###### EfficientNet-B0 #####
    # model = efficientnet_b0(pretrained=True)
    # wandb.config.model_type = 'efficientnet_b0'

    # # Get the number of features in the last layer
    # num_ftrs = model.classifier[1].in_features

    # # Replace the last layer (classifier) with a new one for 4 classes
    # model.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, len(class_list)))
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    model.train()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    # class_weights = torch.tensor([1,2,8], dtype=torch.float)
    # wandb.config.class_weights = "[1,2,8]"

    # Move the weights to the correct device (GPU or CPU)
    # class_weights = class_weights.to(device)

    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    ### Weighting for Focal Loss ###
    # class_counts = np.array([37906,26200,24755])
    # num_classes = 3
    # total_samples = 88861

    class_counts = np.array([5460,1508,2046])
    total_samples = 9014

    class_weights = []
    for count in class_counts:
        weight = 1 / (count / total_samples)
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights).to(device)
    # print(class_weights)
    wandb.config.class_weights = str(class_weights)
    criterion = FocalLoss(alpha=class_weights, gamma=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr = 0.00003)

    ##### Accuracy Metrics #####
    acc_metric_train = torchmetrics.Accuracy(num_classes = len(class_list), task='multiclass').to(device)
    acc_metric_val = torchmetrics.Accuracy(num_classes = len(class_list), task='multiclass').to(device)

    ##### Precision Metrics #####
    per_class_prec_metric_train = torchmetrics.Precision(num_classes = len(class_list), average='none', task='multiclass').to(device)
    per_class_prec_metric_val = torchmetrics.Precision(num_classes = len(class_list), average='none', task='multiclass').to(device)

    ##### Recall Metrics #####
    per_class_recall_metric_train = torchmetrics.Recall(num_classes = len(class_list), average='none', task='multiclass').to(device)
    per_class_recall_metric_val = torchmetrics.Recall(num_classes = len(class_list), average='none', task='multiclass').to(device)

    F1_score_per_class = torchmetrics.F1Score(num_classes=len(class_list), average='none', task='multiclass').to(device)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader

    model_name = f"model-{int(time.time())}"
    print("################################# MODEL NAME: ########################################")
    print(model_name)
    wandb.config.model_name = model_name

    soft = torch.nn.Softmax(dim=1)

    best_acc = -1.0
    min_loss = 1000

    for epoch in range(epochs):

        metrics = {}
        test_metrics = {}

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        ####################### TRAINING PHASE #########################
        phase = 'train'
        print("*** Phase: "+phase+" ***")

        model.train()
        running_loss_train = 0.0       

        for inputs, labels in tqdm(dataloaders[phase]):
            optimizer.zero_grad()  

            inputs = inputs.float().to(device)                  # [B, 3, H, W]
            labels = labels.to(device)                            # [B, H, W]

            # forward pass - track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()

                outputs = soft(outputs)
                preds_torch = torch.argmax(outputs.detach(), dim=1).int()
                labels_torch = labels.int().detach()

                acc_train = acc_metric_train(preds_torch, labels_torch)
                per_class_prec_train = per_class_prec_metric_train(preds_torch, labels_torch)
                per_class_recall_train = per_class_recall_metric_train(preds_torch, labels_torch)

                optimizer.step() 
            
            running_loss_train = running_loss_train + loss.detach()

        acc_train = acc_metric_train.compute()
        per_class_prec_train = per_class_prec_metric_train.compute()
        per_class_recall_train = per_class_recall_metric_train.compute()

        wandb.log({"prec_coral_train":per_class_prec_train[1],   # ["No-Deploy","Coral","Deploy"]
            "prec_deploy_train":per_class_prec_train[2],
            "prec_no-deploy_train":per_class_prec_train[0]}, step=epoch)

        wandb.log({"recall_coral_train":per_class_recall_train[1], 
            "recall_deploy_train":per_class_recall_train[2],
            "recall_no-deploy_train":per_class_recall_train[0]}, step=epoch)

        train_loss = running_loss_train / len(dataloaders[phase])

        metrics[phase+'_loss'] = train_loss.item()
        metrics[phase+'_pa'] = acc_train.item()

        ####################### VALIDATION PHASE #########################
        phase = 'val'
        print("*** Phase: "+phase+" ***")

        model.eval()
        running_loss_val = 0.0

        for inputs, labels in tqdm(dataloaders[phase]):
            optimizer.zero_grad()  

            inputs = inputs.float().to(device)                  # [B, 3, H, W]
            labels = labels.to(device)                          # [B, H, W]

            # forward pass - track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                outputs = soft(outputs)
                preds_torch = torch.argmax(outputs.detach(), dim=1).int()
                labels_torch = labels.int().detach()

                acc_val = acc_metric_val(preds_torch, labels_torch)
                per_class_prec_val = per_class_prec_metric_val(preds_torch, labels_torch)
                per_class_recall_val = per_class_recall_metric_val(preds_torch, labels_torch)

                per_class_f1_val = F1_score_per_class(preds_torch, labels_torch)
            
            running_loss_val = running_loss_val + loss.detach()

        val_loss = running_loss_val / len(dataloaders[phase])

        acc_val = acc_metric_val.compute()
        per_class_prec_val = per_class_prec_metric_val.compute()
        per_class_recall_val = per_class_recall_metric_val.compute()

        per_class_f1_val = F1_score_per_class.compute()

        wandb.log({"prec_coral_val":per_class_prec_val[1], 
            "prec_deploy_val":per_class_prec_val[2],
            "prec_no-deploy_val":per_class_prec_val[0]}, step=epoch)

        wandb.log({"recall_coral_val":per_class_recall_val[1], 
            "recall_deploy_val":per_class_recall_val[2],
            "recall_no-deploy_val":per_class_recall_val[0]}, step=epoch)

        metrics[phase+'_loss'] = val_loss.item()
        metrics[phase+'_pa'] = acc_val.item()

        scheduler.step(val_loss) 
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
        wandb.log({"lr":curr_lr}, step=epoch)

        # Save the best accuracy model weights
        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), "outputs/models/pytorch/"+str(model_name)+'CKPT.pt')

        # Save the lowest loss model weights
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), "outputs/models/pytorch/"+str(model_name)+'_loss_CKPT.pt')

        acc_metric_train.reset()
        per_class_prec_metric_train.reset()
        per_class_recall_metric_train.reset()

        acc_metric_val.reset()
        per_class_prec_metric_val.reset()
        per_class_recall_metric_val.reset()

        F1_score_per_class.reset()

        wandb.log(metrics, step=epoch)

    # Save the final model weights
    torch.save(model.state_dict(), "outputs/models/pytorch/"+str(model_name)+'_FINAL.pt')