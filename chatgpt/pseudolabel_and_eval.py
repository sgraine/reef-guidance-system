# Import packages
import os, sys
import numpy as np
import torch
from tqdm import tqdm
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from chatgpt.chatgpt_script import VLMGPT

vlm = VLMGPT()

print("Chatgpt setup successfully.")

def classify_image(image_path):
    response = vlm.action(image_path)
    return response

# Dataloader for reading in the patch dataset
def loadSetOrig(class_list, test_path, batch_size=12, num_workers=4):
    'Loads data into generator object'
    test_images_array = np.array([])
    test_labels_array = np.array([])          

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(test_path, class_list[category])) if ( (f.endswith(".jpg") or f.endswith(".png")) )]
        
        for i in range(len(img_list)):
            test_images_array = np.append(test_images_array, os.path.join(class_list[category], img_list[i]))
            test_labels_array = np.append(test_labels_array, category)         

    # Set a fixed seed
    seed = 42
    generator = torch.Generator().manual_seed(seed)

    # Now create the dataloaders  
    test_dataset = PatchDataset(test_images_array, test_labels_array, test_path, num_classes=len(class_list))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False, generator=generator)

    return test_dataloader

# Patch dataset
class PatchDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir, num_classes=4):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.num_classes = num_classes
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image_transformed = np.array(Image.open(image_path)) # this line for paligemma

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        return image_transformed, image_path, label


if __name__ == '__main__':

    class_list = ["No-Deploy","Coral","Deploy"]

    test_dataloader = loadSetOrig(class_list, "../CleanData/Evaluation/Patches/Combined-TrainSplit", batch_size=1, num_workers=1)  # "../CleanData/Evaluation/Patches/Combined-TestSplit"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    ##### Accuracy Metrics #####
    acc_metric_test = torchmetrics.Accuracy(num_classes = len(class_list), task='multiclass', multidim_average='global').to(device)

    ##### Precision Metrics #####
    per_class_prec_metric_test = torchmetrics.Precision(num_classes = len(class_list), average='none', task='multiclass', multidim_average='global').to(device)

    ##### Recall Metrics #####
    per_class_recall_metric_test = torchmetrics.Recall(num_classes = len(class_list), average='none', task='multiclass', multidim_average='global').to(device)
    
    ##### F1 Metrics #####
    F1_score = torchmetrics.F1Score(num_classes=len(class_list), task='multiclass', multidim_average='global').to(device)
    F1_score_per_class = torchmetrics.F1Score(num_classes=len(class_list), average='none', task='multiclass', multidim_average='global').to(device)

    conf_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=len(class_list), normalize='none').to(device)

    dataloaders = {}
    dataloaders['test'] = test_dataloader

    phase = 'test'

    step = 0
    error_list = []

    with open("outputs/pseudo-label-outputs.txt", "w") as f:
        sys.stdout = f  # Redirect print output to file

        print("Iterating through the test dataset...")
        for _, image_path, labels in tqdm(dataloaders[phase]):
            
            labels = labels.to(device)                            # [B, H, W]
            labels_torch = labels.int()

            try:
                pred, _ = classify_image(image_path[0])

                preds_torch = torch.tensor(pred, dtype=torch.int32).to(device)

            except Exception as e: 
                print(f"An error occurred: {str(e)}")
                continue

            preds_torch = torch.unsqueeze(preds_torch, 0)

            print(image_path, "Predicted Class:", preds_torch.item(), "True Class:", labels_torch.item())

            acc_test = acc_metric_test(preds_torch, labels_torch)
            per_class_prec_test = per_class_prec_metric_test(preds_torch, labels_torch)
            per_class_recall_test = per_class_recall_metric_test(preds_torch, labels_torch)
            f1_test = F1_score(preds_torch, labels_torch)
            per_class_f1_test = F1_score_per_class(preds_torch, labels_torch)
            conf_mat = conf_matrix(preds_torch, labels_torch)

            break

        # Calculate metrics
        acc_test = acc_metric_test.compute()
        per_class_prec_test = per_class_prec_metric_test.compute()
        per_class_recall_test = per_class_recall_metric_test.compute()
        f1_test = F1_score.compute()
        per_class_f1_test = F1_score_per_class.compute()
        conf_mat = conf_matrix.compute()

        print("Accuracy", acc_test.item())
        print("Precision and Recall", per_class_prec_test, per_class_recall_test)
        print("F1 Scores", f1_test, per_class_f1_test)

        print(conf_mat)