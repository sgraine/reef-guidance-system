# Import packages
import os, time
from torchvision.models import mobilenet_v3_small
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloaders.dataloaders_mlp import loadDataValSplit
import torch.optim as optim
from tqdm import tqdm
import wandb
import torchmetrics
import numpy as np

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

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # wandb.init(mode="disabled")
    wandb.init(project="DGS-MLP", entity="sgraine")

    dataset_path = "../CleanData/Training - ImageWeakLabelling/Combined" 
    test_path = '../CleanData/Evaluation/Deployment/Combined'
    wandb.config.dataset = dataset_path
    train_class_list = ["No-Deploy","Coral","Deploy"]
    test_class_list = ["No-Deploy","Deploy"]

    epochs = 50 #10
    learning_rate = 0.0001
    batch_size = 1

    wandb.config.batch_size = batch_size

    train_dataloader, val_dataloader, test_dataloader = loadDataValSplit(train_class_list, dataset_path, test_class_list, test_path, batch_size, num_workers=1)
   
    model_name = "model-1745448701CKPT"
    model_path = "outputs/models/pytorch/"+model_name+".pt"

    ######################## LOAD PATCH MODEL ########################
    ###### mobilenet_v3_small #####
    patch_model = mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
    wandb.config.model_type = 'MobileNet_V3'

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

    ######################## CREATE NEW MAPPING MODEL ########################

    patch_class_dim = patch_model(torch.randn(1, 3, 256, 256).to(device)).shape[1]
    new_model = PatchGridClassifier(patch_class_dim, (7, 4))

    new_model = new_model.to(device)
    new_model.train()

    for param in new_model.parameters():
        param.requires_grad = True

    print("Models loaded.")

    optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr = 0.00003)

    ##### Accuracy Metrics #####
    acc_metric_train = torchmetrics.classification.BinaryAccuracy().to(device)
    acc_metric_val = torchmetrics.classification.BinaryAccuracy().to(device)
    acc_metric_test = torchmetrics.classification.BinaryAccuracy().to(device)

    ##### Precision Metrics #####
    per_class_prec_metric_train = torchmetrics.classification.BinaryPrecision().to(device)
    per_class_prec_metric_val = torchmetrics.classification.BinaryPrecision().to(device)
    per_class_prec_metric_test = torchmetrics.classification.BinaryPrecision().to(device)

    ##### Recall Metrics #####
    per_class_recall_metric_train = torchmetrics.classification.BinaryRecall().to(device)
    per_class_recall_metric_val = torchmetrics.classification.BinaryRecall().to(device)
    per_class_recall_metric_test = torchmetrics.classification.BinaryRecall().to(device)

    F1_score_metric_train = torchmetrics.classification.BinaryF1Score().to(device)
    F1_score_metric_val = torchmetrics.classification.BinaryF1Score().to(device)
    F1_score_metric_test = torchmetrics.classification.BinaryF1Score().to(device)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader
    dataloaders['test'] = test_dataloader

    model_name = f"model-{int(time.time())}"
    print("################################# MODEL NAME: ########################################")
    print(model_name)
    wandb.config.model_name = model_name

    sigmoid_func = torch.nn.Sigmoid()

    best_acc = -1.0
    best_prec = -1.0
    min_loss = 1000

    for epoch in range(epochs):

        metrics = {}
        test_metrics = {}

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        ####################### TRAINING PHASE #########################
        phase = 'train'
        print("*** Phase: "+phase+" ***")

        new_model.train()
        running_loss_train = 0.0       

        for inputs, labels in tqdm(dataloaders[phase]):
            optimizer.zero_grad()  

            inputs = inputs.float().to(device)                  # [B, 28, 3, H, W]
            inputs = torch.squeeze(inputs, dim=0)
            labels = torch.unsqueeze(labels, 0).float().to(device)                            # [B, H, W]

            patch_preds = patch_model(inputs)

            # forward pass - track history if only in train
            with torch.set_grad_enabled(True):
                outputs = new_model(torch.unsqueeze(patch_preds, 0))

                loss = criterion(outputs, labels)
                outputs = sigmoid_func(outputs)
                # print(outputs)

                preds_torch = (outputs >= 0.5).int()[0]
                labels_torch = labels.int().detach()[0]
                # print(preds_torch, labels_torch)
                # print(preds_torch.shape, labels_torch.shape)

                acc_train = acc_metric_train(preds_torch, labels_torch)
                per_class_prec_train = per_class_prec_metric_train(preds_torch, labels_torch)
                per_class_recall_train = per_class_recall_metric_train(preds_torch, labels_torch)
                per_class_f1_train = F1_score_metric_train(preds_torch, labels_torch)

                loss.backward()
                optimizer.step() 
            
            running_loss_train = running_loss_train + loss.detach()

            # break

        train_loss = running_loss_train / len(dataloaders[phase])

        acc_train = acc_metric_train.compute()
        per_class_prec_train = per_class_prec_metric_train.compute()
        per_class_recall_train = per_class_recall_metric_train.compute()
        per_class_f1_train = F1_score_metric_train.compute()

        metrics[phase+'_loss'] = train_loss.item()
        metrics[phase+'_pa'] = acc_train.item()
        metrics[phase+'_prec'] = per_class_prec_train.item()
        metrics[phase+'_rec'] = per_class_recall_train.item()
        metrics[phase+'_f1'] = per_class_f1_train.item()

        ####################### VALIDATION PHASE #########################
        phase = 'val'
        print("*** Phase: "+phase+" ***")

        new_model.eval()
        running_loss_val = 0.0

        for inputs, labels in tqdm(dataloaders[phase]):
            optimizer.zero_grad()  

            inputs = inputs.float().to(device)                  # [B, 28, 3, H, W]
            inputs = torch.squeeze(inputs, dim=0)
            labels = torch.unsqueeze(labels, 0).float().to(device)                            # [B, H, W]

            patch_preds = patch_model(inputs)

            # forward pass - track history if only in train
            with torch.set_grad_enabled(True):
                outputs = new_model(torch.unsqueeze(patch_preds, 0))

                loss = criterion(outputs, labels)
                outputs = sigmoid_func(outputs)

                preds_torch = (outputs >= 0.5).int()[0]
                labels_torch = labels.int().detach()[0]

                acc_val = acc_metric_val(preds_torch, labels_torch)
                per_class_prec_val = per_class_prec_metric_val(preds_torch, labels_torch)
                per_class_recall_val = per_class_recall_metric_val(preds_torch, labels_torch)
                per_class_f1_val = F1_score_metric_val(preds_torch, labels_torch)
            
            running_loss_val = running_loss_val + loss.detach()

            # break

        val_loss = running_loss_val / len(dataloaders[phase])

        acc_val = acc_metric_val.compute()
        per_class_prec_val = per_class_prec_metric_val.compute()
        per_class_recall_val = per_class_recall_metric_val.compute()
        per_class_f1_val = F1_score_metric_val.compute()

        metrics[phase+'_loss'] = val_loss.item()
        metrics[phase+'_pa'] = acc_val.item()
        metrics[phase+'_prec'] = per_class_prec_val.item()
        metrics[phase+'_rec'] = per_class_recall_val.item()
        metrics[phase+'_f1'] = per_class_f1_val.item()

        # scheduler.step(val_loss) 
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
        wandb.log({"lr":curr_lr}, step=epoch)

        if per_class_f1_val > best_acc:
            best_acc = per_class_f1_val
            torch.save(new_model.state_dict(), "outputs/models/pytorch/"+str(model_name)+'_f1_CKPT.pt')

        if per_class_prec_val > best_prec:
            best_prec = per_class_prec_val
            torch.save(new_model.state_dict(), "outputs/models/pytorch/"+str(model_name)+'_prec_CKPT.pt')

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(new_model.state_dict(), "outputs/models/pytorch/"+str(model_name)+'_loss_CKPT.pt')

        wandb.log(metrics, step=epoch)

        ####################### TEST PHASE #########################
        phase = 'test'
        print("*** Phase: "+phase+" ***")

        new_model.eval()
        running_loss_test = 0.0

        for inputs, labels in tqdm(dataloaders[phase]):
            optimizer.zero_grad()  

            inputs = inputs.float().to(device)                  # [B, 28, 3, H, W]
            inputs = torch.squeeze(inputs, dim=0)
            labels = torch.unsqueeze(labels, 0).float().to(device)                            # [B, H, W]

            patch_preds = patch_model(inputs)

            # forward pass - track history if only in train
            with torch.set_grad_enabled(True):
                outputs = new_model(torch.unsqueeze(patch_preds, 0))

                loss = criterion(outputs, labels)
                outputs = sigmoid_func(outputs)

                preds_torch = (outputs >= 0.5).int()[0]
                labels_torch = labels.int().detach()[0]

                acc_test = acc_metric_test(preds_torch, labels_torch)
                per_class_prec_test = per_class_prec_metric_test(preds_torch, labels_torch)
                per_class_recall_test = per_class_recall_metric_test(preds_torch, labels_torch)

                per_class_f1_test = F1_score_metric_test(preds_torch, labels_torch)
            
            running_loss_test = running_loss_test + loss.detach()

            # break

        test_loss = running_loss_test / len(dataloaders[phase])

        acc_test = acc_metric_test.compute()
        per_class_prec_test = per_class_prec_metric_test.compute()
        per_class_recall_test = per_class_recall_metric_test.compute()
        per_class_f1_test = F1_score_metric_test.compute()

        test_metrics[phase+'_loss'] = test_loss.item()
        test_metrics[phase+'_pa'] = acc_test.item()
        test_metrics[phase+'_prec'] = per_class_prec_test.item()
        test_metrics[phase+'_rec'] = per_class_recall_test.item()
        test_metrics[phase+'_f1'] = per_class_f1_test.item()

        wandb.log(test_metrics, step=epoch)

        acc_metric_train.reset()
        per_class_prec_metric_train.reset()
        per_class_recall_metric_train.reset()
        F1_score_metric_train.reset()

        acc_metric_val.reset()
        per_class_prec_metric_val.reset()
        per_class_recall_metric_val.reset()
        F1_score_metric_val.reset()

        acc_metric_test.reset()
        per_class_prec_metric_test.reset()
        per_class_recall_metric_test.reset()
        F1_score_metric_test.reset()

        print(metrics)
        print(test_metrics)



    # Save the final model weights
    torch.save(new_model.state_dict(), "outputs/models/pytorch/"+str(model_name)+'_FINAL.pt')