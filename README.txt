#########################################
################# SUMMARY ###############
#########################################

Project: Deployment Guidance System AI 
Author: Scarlett Raine (sg.raine@qut.edu.au)

Description: These scripts are for the DGS AI functionality, and encompass all tasks from data pre-processing and pseudo-labelling, training models, data analysis and model evaluation, and visualisation of model outputs. Please note that this is research code only, so will likely not be optimised for efficiency or speed, and the code itself is written for function and not necessarily presentation.  The scripts below may contain errors or bugs (all efforts made to avoid this). The main packages used are Pixi for package management and running scripts, Pytorch for training and evaluating models, and ONNX for model deployment on the Jetson. 

%%%%%% Pixi %%%%%

The directory has a pixi.toml file which is used to create the pixi environment and run the tasks as specified.  If pixi is not already installed, you can install it easily using the instructions here: https://pixi.sh/latest/. For completeness, on Linux/MacOS you can run:

	curl -fsSL https://pixi.sh/install.sh | sh

Once installed, you can use the following commands to run the codebase:

	pixi run -e cuda [task]

or if a GPU is not needed:

	pixi run [task]

The task names are in the square brackets below, beside each of the scripts.  If GPU (cuda) is needed, it is indicated by the [cuda] beside the task name. If the script does not have the square brackets beside it, then it is not an executable script (likely imported by another script)


%%%%%%% Scripts %%%%%

1. ChatGPT pseudo-labelling
chatgpt/
- chatgpt_script.py | script which initializes ChatGPT from the API and creates the necessary prompting structure
- [chatgpt-label] pseudolabel_and_eval.py | script that reads in the functions from chatgpt_script, and then pseudo-labels a folder of image patches. It also calculates the metrics against provided ground truth labels for the same patches

2. Training
training/
- [cuda][train] train_models_patchlabels.py | script to train a classification model on a dataset labelled at the patch level
- [cuda][train-image] train_models_imagelabels.py | script to train a classification model on a dataset labelled at the image level
- [cuda][convert-onnx] convert_onnx.py | this script converts a pytorch model into an onnx model, useful for inference on a jetson

3. Dataloaders
dataloaders/
- dataloaders_patchlabels.py | pytorch dataloaders which load in patches
- dataloaders_imagelabels.py | pytorch dataloaders which load in whole images

4. Evaluation
eval/
- [cuda][eval-deploy] pytorch_patch_deployment_eval.py | evaluates a pytorch model on the whole image deployment task (classifies each patch first, then decides deploy/no-deploy using threshold
- [cuda][eval-patches] pytorch_patch_eval.py | evaluates a pytorch model on a test patch dataset (where every patch has been assigned a ground truth label)
- [cuda][eval-images] pytorch_image_eval.py | evaluates a pytorch model trained on whole images

5. Visualisation
vis/
- [cuda][vis-coral-gps] coral_gps.py | a script to create a gps map of the coral coverage based on model predictions
- infer_pytorch_save_gps.py | 
- [vis-onnx-gps] infer_onnx_save_gps.py | a script that takes an onnx model and performs inference on a folder of images, resulting in a gps track of the model decisions. If there are associated labels for the images i.e. the images are saved in deploy/no-deploy directories, then it will also create a ground truth gps track for comparison

6. Utility Files
utils/
- [] create_video_from_frames.py | creates a .mp4 video from a folder containing sequential image frames
- [] randomise_and_recover_images.py | this script has functions which can rename a folder of images such that they are randomised, and then name them back to their original filenames
- [calc-obs-var] calculate_observer_variability.py | given two folders, each containing sub-directories for the labels for images, this will calculate the agreement between the two label sets
- [] create_patches_from_images.py | a script that takes whole images and creates a grid of x by y, saving each grid cell as a patch
- [] create_patches_from_images_noclasses.py | a script that takes whole images and creates a grid of x by y, saving each grid cell as a patch; but when the images are not labelled


%%%%% Saved Models %%%%%%
outputs/models/

pytorch/
- Model trained on ecologist patches: 
- Model trained on ecologist patches with focal loss: 
- Model trained on ChatGPT patches: 
- Model trained on ChatGPT patches with focal loss: 
- Model trained on CLIP patches: 
- Model trained on CLIP patches with focal loss: 

onnx/
- Best model converted to onnx for deployment on the Jetson: 

%%%%% Saved Maps %%%%%%
outputs/maps/

-GPS tracks of best model:
-GPS tracks of all sites (labelling by Amy):
-GPS tracks of sites by other ecologists:

-Coral coverage estimation map by best model: