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
├── chatgpt_script.py             # Initializes ChatGPT API and prompt structure
├── pseudolabel_and_eval.py       # [chatgpt-label] Pseudo-labels image patches with ChatGPT and evaluates against ground truth
```

### 2. Training

```
training/
├── train_models_patchlabels.py   # [cuda][train] Patch-level model training
├── train_models_imagelabels.py   # [cuda][train-image] Image-level model training
├── convert_onnx.py               # [cuda][convert-onnx] Converts PyTorch model to ONNX
```

### 3. Dataloaders

```
dataloaders/
├── dataloaders_patchlabels.py    # Pytorch dataloaders to load patch-labelled datasets
├── dataloaders_imagelabels.py    # Pytorch dataloaders to load image-labelled datasets
```

### 4. Evaluation

```
eval/
├── pytorch_patch_deployment_eval.py   # [cuda][eval-deploy] Evaluates a pytorch model on the whole image deployment task (classifies each patch first, then decides deploy/no-deploy using threshold)
├── pytorch_patch_eval.py              # [cuda][eval-patches] Evaluates a pytorch model on a test patch dataset (where every patch has been assigned a ground truth label)
├── pytorch_image_eval.py              # [cuda][eval-images] Evaluates a pytorch model trained on whole images
```

### 5. Visualisation

```
vis/
├── coral_gps.py                  # [cuda][vis-coral-gps] Creates a GPS map of estimated coral coverage (based on the number of coral patches)
├── infer_onnx_save_gps.py        # [vis-onnx-gps] ONNX inference and GPS track generation - if there are associated labels for the images i.e. the images are saved in deploy/no-deploy directories, then it will also create a ground truth gps track for comparison
```

### 6. Jetson Inference Files

```
infer/
# Jetson deployment-related scripts (TBD)
```

### 7. Utility Files

```
utils/
├── create_video_from_frames.py             # Creates a .mp4 video from a folder containing sequential image frames
├── randomise_and_recover_images.py         # Functions which can rename a folder of images such that they are randomised, and then name them back to their original filenames
├── calculate_observer_variability.py       # [calc-obs-var] Given two folders, each containing sub-directories for the labels for images, this will calculate the agreement between the two label sets
├── create_patches_from_images.py           # Grid-based patch creation from labelled images
├── create_patches_from_images_noclasses.py # Grid-based patch creation for unlabelled images
```

---

## Saved Models

- Model trained on ecologist patches
- Model trained on ChatGPT patches

```
outputs/
└── models/
    ├── pytorch/
    │   ├── ecologist_model.pt
    │   ├── chatgpt_model.pt
    └── onnx/
        └── best_model.onnx
```

---

## Saved Maps

- GPS tracks of best model predictions  
- GPS tracks of all labelled sites (Amy)  
- GPS tracks by other ecologists  
- Coral coverage estimation maps

```
outputs/
└── maps/
```


