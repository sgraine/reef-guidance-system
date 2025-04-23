# Reef Guidance System

## Summary

**Project:** Reef Guidance System AI  
**Author:** Scarlett Raine ([sg.raine@qut.edu.au](mailto:sg.raine@qut.edu.au))

These scripts are for the Reef Guidance System AI functionality and encompass all tasks from data pre-processing and pseudo-labelling, training models, data analysis and model evaluation, and visualisation of model outputs.

> **Note:** This is research code only. It may not be optimised for efficiency or presentation. The code may contain bugs despite efforts to avoid them.

Main packages used:
- **[Pixi](https://pixi.sh/latest/)** – for package management and running scripts
- **PyTorch** – for training and evaluation
- **ONNX** – for deployment on the Jetson

---

## Pixi Setup

This repo uses `pixi.toml` to define the environment and task runners.

To install Pixi (Linux/MacOS):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Once installed, run tasks using:

```bash
pixi run -e cuda [task]    # With GPU
pixi run [task]            # CPU only
```

The task names are in the square brackets `[task-name]` below, beside each of the scripts.  If GPU (cuda) is needed, it is indicated by the [cuda] beside the task name. If the script does not have a task name in square brackets beside it, then the script is imported by others and not directly executable.
---

## Scripts

### 1. ChatGPT Pseudo-labelling

```
chatgpt/
├── chatgpt_script.py
├── pseudolabel_and_eval.py
```

- `chatgpt_script.py` | script which initializes ChatGPT from the API and creates the necessary prompting structure
- [chatgpt-label] `pseudolabel_and_eval.py` | script that reads in the functions from chatgpt_script, and then pseudo-labels a folder of image patches. It also calculates the metrics against provided ground truth labels for the same patches

### 2. Training

```
training/
├── train_models_patchlabels.py   
├── train_models_imagelabels.py   
├── convert_onnx.py               
```

- [cuda][train] `train_models_patchlabels.py` | script to train a classification model on a dataset labelled at the patch level
- [cuda][train-image] `train_models_imagelabels.py` | script to train a classification model on a dataset labelled at the image level
- [cuda][convert-onnx] `convert_onnx.py` | this script converts a pytorch model into an onnx model, useful for inference on a jetson

### 3. Dataloaders

```
dataloaders/
├── dataloaders_patchlabels.py    
├── dataloaders_imagelabels.py    
```

- `dataloaders_patchlabels.py` | pytorch dataloaders which load in patches
- `dataloaders_imagelabels.py` | pytorch dataloaders which load in whole images

### 4. Evaluation

```
eval/
├── pytorch_patch_deployment_eval.py   
├── pytorch_patch_eval.py              
├── pytorch_image_eval.py              
```

- [cuda][eval-deploy] `pytorch_patch_deployment_eval.py` | evaluates a pytorch model on the whole image deployment task (classifies each patch first, then decides deploy/no-deploy using threshold
- [cuda][eval-patches] `pytorch_patch_eval.py` | evaluates a pytorch model on a test patch dataset (where every patch has been assigned a ground truth label)
- [cuda][eval-images] `pytorch_image_eval.py` | evaluates a pytorch model trained on whole images

### 5. Visualisation

```
vis/
├── coral_gps.py                  
├── infer_onnx_save_gps.py        
```

- [cuda][vis-coral-gps] `coral_gps.py` | a script to create a gps map of the coral coverage based on model predictions (based on the number of coral patches in an image)
- [vis-onnx-gps] `infer_onnx_save_gps.py` | a script that takes an onnx model and performs inference on a folder of images, resulting in a gps track of the model decisions. If there are associated labels for the images i.e. the images are saved in deploy/no-deploy directories, then it will also create a ground truth gps track for comparison

### 6. Jetson Inference Files

```
infer/
# Jetson deployment-related scripts (TBD)
```

### 7. Utility Files

```
utils/
├── create_video_from_frames.py             
├── randomise_and_recover_images.py         
├── calculate_observer_variability.py       
├── create_patches_from_images.py           
├── create_patches_from_images_noclasses.py 
```

- [] `create_video_from_frames.py` | creates a .mp4 video from a folder containing sequential image frames
- [] `randomise_and_recover_images.py` | this script has functions which can rename a folder of images such that they are randomised, and then name them back to their original filenames
- [calc-obs-var] `calculate_observer_variability.py` | given two folders, each containing sub-directories for the labels for images, this will calculate the agreement between the two label sets
- [] `create_patches_from_images.py` | a script that takes whole images and creates a grid of x by y, saving each grid cell as a patch
- [] `create_patches_from_images_noclasses.py` | a script that takes whole images and creates a grid of x by y, saving each grid cell as a patch; but when the images are not labelled

---

## Saved Models

```
outputs/
└── models/
    ├── pytorch/
    │   ├── ecologist_model.pt
    │   ├── chatgpt_model.pt
    └── onnx/
        └── best_model.onnx
```

- Model trained on ecologist patches
- Model trained on ChatGPT patches

---

## Saved Maps

```
outputs/
└── maps/
```

- GPS tracks of best model predictions  
- GPS tracks of all labelled sites (Amy)  
- GPS tracks by other ecologists  
- Coral coverage estimation maps
