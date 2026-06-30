# Import packages
print("Importing necessary packages...")
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class ModifiedModel(nn.Module):
    def __init__(self, base_model, deploy_threshold=0.6):
        super(ModifiedModel, self).__init__()
        self.base_model = base_model
        self.deploy_threshold = deploy_threshold 
    
    def preprocess_image_torch(self, image, grid_shape=(4, 7)):
        """
        image: torch tensor of shape (1, 3, H, W)
        grid_shape: (rows, cols)
        """
        grid_rows, grid_cols = grid_shape

        image = image / 255.0

        # Compute patch size
        patch_h = 760
        patch_w = 758

        target_h = patch_h * grid_rows  # 3032
        target_w = patch_w * grid_cols  # 5320

        # Resize to match grid
        image = F.interpolate(image, size=(target_h, target_w), mode='bicubic', align_corners=False)

        patches = []

        # Loop over the grid and manually slice patches
        for i in range(grid_rows):
            for j in range(grid_cols):
                start_h = i * patch_h
                end_h = (i + 1) * patch_h
                start_w = j * patch_w
                end_w = (j + 1) * patch_w
                patch = image[:, :, start_h:end_h, start_w:end_w]
                patches.append(patch)

        # Stack patches into a single tensor (N, C, H, W) format
        patches = torch.cat(patches, dim=0)  # N * (C, H, W) for all patches

        # Step 6: Resize patches to 256x256
        patches_resized = F.interpolate(patches, size=(256, 256), mode="bicubic", align_corners=False)

        # Step 7: Normalize patches (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)  # Shape: (1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)  # Shape: (1, 3, 1, 1)
        patches_normalized = (patches_resized - mean) / std

        return patches_normalized, patches, patches_resized

    def forward(self, x):
        patches, patches_debug, patches_resize = self.preprocess_image_torch(x)

        outputs = self.base_model(patches)  # Shape: [num_patches, num_classes]
        softmax_output = F.softmax(outputs, dim=1)

        preds = torch.argmax(softmax_output, dim=1).int()

        # Compute deploy ratio
        deploy_class = 2  # Assuming 2 == deploy
        deploy_count = (preds == deploy_class).sum()
        total_count = preds.shape[0]
        ratio = deploy_count.float() / total_count

        deploy = (ratio > self.deploy_threshold).int()

        return preds, ratio, deploy, patches, patches_debug, patches_resize


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ###### Resnet-18 #####
    # model_path = "model-1739752833CKPT.pt"

    # model_load = models.resnet18(pretrained=False)
    # model_load.fc = nn.Sequential(nn.Linear(512, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, 3))


    ###### SMALL RESOLUTION EFFICIENTNET-BO ######
    # model_path = "model-1739772951CKPT.pt"

    # model_load = models.efficientnet_b0(pretrained=False)

    # # Get the number of features in the last layer
    # num_ftrs = model_load.classifier[1].in_features

    # # Replace the last layer (classifier) with a new one for 4 classes
    # model_load.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, 3))

    ##### MOBILENET MODEL #####
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

    # Wrap the original model
    modified_model = ModifiedModel(model_load).to(device)
    modified_model.eval()

    batch_size = 1
    summary(modified_model, input_size=(batch_size, 3, 5312, 3040))

    input_tensor = torch.randn(1,3,5312,3040).to(device) # mobilenet patch inference
    
    softmax_output, class_pred, ratio, patches, patches_debug, patches_resize = modified_model(input_tensor)

    print("Converting to onnx...")
    
    # Export the model
    torch.onnx.export(
        modified_model,            # The model to export
        input_tensor,              # An example input tensor
        "outputs/models/onnx/Mobilenet-1-3-5312-3040.onnx",    # The file path where the model will be saved
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=11,          # ONNX opset version (adjust if needed)
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],     # Input layer name(s)
        output_names=['output'],   # Output layer name(s)
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Allow for dynamic batching
    )

    print("Model has been exported to ONNX format.")
