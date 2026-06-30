# Import packages
print("Importing necessary packages...")
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse

class ModifiedModel(nn.Module):
    def __init__(self, patch_model, new_model, deploy_threshold=0.3):
        super(ModifiedModel, self).__init__()
        self.patch_model = patch_model  # This is your original model
        self.new_model = new_model
        self.deploy_threshold = deploy_threshold

    def forward(self, x):

        # x = self.preprocess_image_torch(x)

        # Pass the input through the original model
        patch_logits = self.patch_model(x)  # shape = [batch_size x num_patches, num_classes]

        # Find output from new_model
        output = self.new_model(torch.unsqueeze(patch_logits, 0)) # shape = [batch_size, 1]

        # Compute sigmoid on the output
        sigmoid_output = F.sigmoid(output)   # shape = [batch_size, 1]

        deploy = (sigmoid_output >= self.deploy_threshold)[0].int()       

        soft_patches = F.softmax(patch_logits, dim=1)
        patch_preds = torch.argmax(soft_patches, dim=1).int()

        return patch_preds, torch.squeeze(sigmoid_output), torch.squeeze(deploy) # Return the additional values


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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPS results.')
    parser.add_argument('--torch_model', type=str, required=True, help='Path to the PyTorch model file')
    parser.add_argument('--mapping_model', type=str, required=True, help='Path to the mapping model weights file')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to the output ONNX model file')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##### MOBILENET MODEL #####
    args = parse_arguments()
    model_path = args.torch_model
    model_load = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_load.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, 3))

    model_load.load_state_dict(torch.load(model_path))


   ######################## LOAD PATCH MODEL ########################
    ###### mobilenet_v3_small #####
    patch_model =models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

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

    ######################## CREATE MAPPING MODEL ########################
    model_path = args.mapping_model

    file_path = Path(model_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    patch_class_dim = patch_model(torch.randn(1, 3, 256, 256).to(device)).shape[1]
    new_model = PatchGridClassifier(patch_class_dim, (7, 4))

    new_model.load_state_dict(torch.load(model_path)) 

    new_model = new_model.to(device)
    new_model.eval()

    # Wrap the original model
    modified_model = ModifiedModel(patch_model, new_model).to(device)
    modified_model.eval()

    input_tensor = torch.randn(28,3,256,256).to(device) # mobilenet patch inference
    
    patch_preds, ratio, deploy = modified_model(input_tensor)

    print("Converting to onnx...")
    
    # Export the model
    torch.onnx.export(
        modified_model,            # The model to export
        input_tensor,              # An example input tensor
        args.onnx_model,    # The file path where the model will be saved
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=11,          # ONNX opset version (adjust if needed)
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],     # Input layer name(s)
        output_names=['output'],   # Output layer name(s)
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Allow for dynamic batching
    )

    print("Model has been exported to ONNX format.")
