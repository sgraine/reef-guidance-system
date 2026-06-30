import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import cv2

# --- First (Pillow + Numpy + imgaug) method ---
def img_to_grid(img, row, col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]), row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]), col)]
    grid = [img[j:jj+1, i:ii+1, :] for j, jj in ww for i, ii in hh]
    return grid, len(ww), len(hh)


def cropper(images, width, height):
    seq = iaa.Sequential([iaa.CropToFixedSize(width=width, height=height)])
    return seq.augment_image(images)


def preprocess_numpy(image, grid_shape=(4, 7)):
    # image = Image.open(image_path).convert("RGB")
    # image = np.array(image)

    image_height, image_width = image.shape[:2]
    row, col = grid_shape

    grid, _, _ = img_to_grid(image, row, col)

    patch_w = int(np.floor(image_width / col))
    patch_h = int(np.floor(image_height / row))

    all_patches = []
    for patch in grid:
        patch_crop = cropper(patch, patch_w, patch_h)
        all_patches.append(patch_crop)

    patches_np_tensor = torch.tensor(np.stack(all_patches)).permute(0, 3, 1, 2).float() / 255.0
    return patches_np_tensor


# --- Second (PyTorch) method ---
def preprocess_image_torch(image, grid_shape=(4, 7)):
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
    return patches

# --- Save visual comparison ---
def save_patch_comparison(i, np_patch, torch_patch, diff_patch, output_dir):
    np_img = (np_patch.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    torch_img = (torch_patch.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    diff_img = (diff_patch.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    Image.fromarray(np_img).save(os.path.join(output_dir, f"patch_{i:02d}_numpy.png"))
    Image.fromarray(torch_img).save(os.path.join(output_dir, f"patch_{i:02d}_torch.png"))
    Image.fromarray(diff_img).save(os.path.join(output_dir, f"patch_{i:02d}_diff.png"))


# --- Main comparison ---
def main(image_path):
    grid_shape = (4, 7)
    # output_dir = "outputs/patch_comparison_outputs"
    # os.makedirs(output_dir, exist_ok=True)

    

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image).astype(np.float32)
    image = np.transpose(image, (2, 1, 0))

    numpy_patches = preprocess_numpy(image, grid_shape)
    
    # image = Image.open(image_path).convert("RGB")
    image_torch = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
    torch_patches = preprocess_image_torch(image_torch, grid_shape)

    print(f"Num patches (numpy): {numpy_patches.shape}")
    print(f"Num patches (torch): {torch_patches.shape}")

    for i in range(min(len(numpy_patches), len(torch_patches))):
        np_patch = numpy_patches[i]
        torch_patch = torch_patches[i]
        diff = torch.abs(np_patch - torch_patch)
        mean_diff = diff.mean().item()

        print(f"Patch {i:02d} - Mean Abs Diff: {mean_diff:.6f}")

        if i == 0:
            print("\n--- Example Patch Values (Patch 0) ---")
            print("Numpy patch (scaled to [0,1]):")
            print(np_patch)
            print("\nTorch patch (scaled to [0,1]):")
            print(torch_patch)
            print("\nAbsolute difference:")
            print(diff)

        if i < 5:  # Save only first few comparisons
            save_patch_comparison(i, np_patch, torch_patch, diff, output_dir)

    print(f"Patch comparison images saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compare_preprocessing.py path/to/image.jpg")
    else:
        main(sys.argv[1])
