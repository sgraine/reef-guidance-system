# Import packages
print("Importing necessary packages...")
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class ModifiedModel(nn.Module):
    def __init__(self, base_model, deploy_threshold=0.8):
        super(ModifiedModel, self).__init__()
        self.base_model = base_model  # This is your original model
        self.deploy_threshold = deploy_threshold

    def forward(self, x):

        # x = self.preprocess_image_torch(x)

        # Pass the input through the original model
        output = self.base_model(x)   # shape = [batch_size (num_patches), num_classes]

        # Compute softmax on the output
        softmax_output = F.softmax(output, dim=1)   # shape = [batch_size (num_patches), num_classes]

        preds = torch.squeeze(torch.argmax(softmax_output, dim=1)).int()
        
        return preds

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPS results.')
    parser.add_argument('--torch_model', type=str, required=True, help='Path to the PyTorch model file')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to the output ONNX model file')

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    ###### Mobilenet_v3_large #####
    # model_path = "outputs/models/pytorch/model-1745449205CKPT.pt"
    # model_load = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")

    # # Get the number of features in the last layer
    # num_ftrs = model_load.classifier[3].in_features

    # # Replace the last layer (classifier) with a new one for 4 classes
    # model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
    #                              nn.ReLU(),
    #                              nn.Dropout(0.15),
    #                              nn.Linear(512, 3))

    ##### MOBILENET MODEL #####
    model_path = args.torch_model
    model_load = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

    # Get the number of features in the last layer
    num_ftrs = model_load.classifier[3].in_features

    # Replace the last layer (classifier) with a new one for 4 classes
    model_load.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, 2))

    model_load.load_state_dict(torch.load(model_path))

    # Wrap the original model
    modified_model = ModifiedModel(model_load).to(device)
    modified_model.eval()

    input_tensor = torch.randn(1,3,1328,760).to(device) # mobilenet patch inference
    
    pred = modified_model(input_tensor)

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
