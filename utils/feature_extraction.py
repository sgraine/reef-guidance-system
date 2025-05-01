import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Wrapper to extract intermediate features
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.features  # All layers before classifier
        self.avgpool = model.avgpool
        self.classifier_part1 = model.classifier[0:3]  # Linear -> ReLU -> Dropout
        self.classifier_part2 = model.classifier[3][3]  # Final classifier Linear(1024, 3)

    def forward(self, x):
        # Backbone feature
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features_backbone = x.clone()

        # Intermediate features
        x = self.classifier_part1(x)
        features_1024 = x.clone()

        return features_backbone, features_1024


# Example image pre-processing
preprocess = transforms.Compose([
    transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).float().unsqueeze(0).to(device)

    with torch.no_grad():
        feat_backbone, feat_512 = feature_extractor(input_tensor)

    return feat_backbone, feat_512

# Example usage
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths and device
    model_path = "outputs/models/pytorch/model-1746052531CKPT.pt" # Model trained on combined and Heron sites only (retain Whitsundays for test)
    
    # Load and modify the model
    model_load = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
    num_ftrs = model_load.classifier[3].in_features
    print(num_ftrs)

    # New classifier
    model_load.classifier[3] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(512, 3)
    )

    # Load weights
    model_load.load_state_dict(torch.load(model_path, map_location=device))
    model_load = model_load.to(device)
    model_load.eval()

    # Instantiate feature extractor
    feature_extractor = FeatureExtractor(model_load).to(device)
    feature_extractor.eval()

    image_path = "../CleanData/Evaluation/Patches/Site 3/Coral/20241001_012855_160_5341_17.png"
    backbone_feat, mid_feat = extract_features(image_path)
    print("Backbone features shape:", backbone_feat.shape)
    print("Intermediate 1024-dim features shape:", mid_feat.shape)
