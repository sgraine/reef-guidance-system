import onnxruntime as ort
import numpy as np
import torch
import cv2, os
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

def video_frames(sigmoid_vals, output, whole_image, image_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(24, 7)
    ax1.axis('off')
    
    # Plot the original image
    ax1.imshow(np.squeeze(whole_image), alpha=1.0) # rows, columns, channels

    # Plot the original image
    ax2.imshow(np.squeeze(whole_image), alpha=1.0) # rows, columns, channels

    scale = np.array([np.shape(whole_image)[1],np.shape(whole_image)[0]])

    # sigmoid_vals = sigmoid_vals.detach().cpu().numpy()d

    sigmoid_vals = np.squeeze(sigmoid_vals)
    if np.shape(sigmoid_vals)[0] == 45:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 9)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 32:
        sigmoid_vals = np.reshape(sigmoid_vals, (4, 8)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 35:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 7)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 40:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 8)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 54:
        sigmoid_vals = np.reshape(sigmoid_vals, (6, 9)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 77:
        sigmoid_vals = np.reshape(sigmoid_vals, (7, 11)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 6:
        sigmoid_vals = np.reshape(sigmoid_vals, (2, 3)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 28:
        sigmoid_vals = np.reshape(sigmoid_vals, (4, 7)) # rows, columns
    else:
        print("Not sure how many rows and columns!", np.shape(sigmoid_vals))

    CMAP = [[255,20,147],[4, 179, 12],[255,165,0],[131,69,63]]
    CMAP = np.asarray(CMAP)
    colour_predictions = CMAP[sigmoid_vals]

    # Plot the heatmap as a transparent overlay
    offs = np.array([scale[0]/sigmoid_vals.shape[1], scale[1]/sigmoid_vals.shape[0]])

    # Add the sigmoid values for each patch as a label
    for pos, val in np.ndenumerate(sigmoid_vals):
        ax2.annotate(val, xy=np.array(pos)[::-1]*offs+offs/2, ha="center", va="center", fontsize=30)

    heatmap = ax2.imshow(np.flipud(colour_predictions), alpha=0.5, aspect="auto", extent=(0,scale[0],0,scale[1])) # alpha -> more transparent as value decreases

    ax2.invert_yaxis()
    ax2.axis('off')

    if output == 1:
        ax2.set_title('DEPLOY')
    else:
        ax2.set_title('NO DEPLOY')

    # Save the final figure as a .png
    plt.tight_layout()
    plt.savefig(image_name+".png", bbox_inches='tight', pad_inches = 0)
    plt.close(fig)


def img_to_grid(img, row, col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
    grid = [img[j:jj+1,i:ii+1,:] for j,jj in ww for i,ii in hh]
    return grid, len(ww), len(hh)

# Define the cropper function using PyTorch operations
def cropper(images, width, height):
    # Get the current size (B, C, H, W)
    C, H, W = images.shape

    # Calculate the cropping coordinates (center crop or simple top-left crop)
    start_x = (W - width) // 2
    start_y = (H - height) // 2

    # Crop the image (C, H, W) -> (C, height, width)
    cropped_images = images[:, start_y:start_y+height, start_x:start_x+width]

    return cropped_images

def preprocess_image(image, input_shape, im_or_patch:str):

    if im_or_patch=='im':
        # convert to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to float32 for precision, as required for normalization
        image = image.astype(np.float32) / 255.0

        # Normalize the image with ImageNet values (mean and std)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # Reshape to (1, 1, 3) for broadcasting
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)   # Reshape to (1, 1, 3) for broadcasting

        # Perform the normalization
        image = (image - mean) / std

        # resize image
        if image.shape[:2] != input_shape:
            image = cv2.resize(image, input_shape)

        # Change the image from HWC to CWH format (channels first)
        image = np.transpose(image, (2, 1, 0))

        # reshape to (1, 3, width, height)
        reshaped = np.expand_dims(image, axis=0)

        return reshaped.astype(np.float32)

    elif im_or_patch=='patch':

        # convert to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Divide the full image into a grid of patches
        grid, _, _ = img_to_grid(image,4,7)  # row, column

        all_patches = []
        for patch in grid:

            
            # Change the image from HWC to CWH format (channels first)
            patch = np.transpose(patch, (2, 0, 1)) 

            # Crop to size from grid
            patch_crop = cropper(patch, 758, 760)

            patch = np.transpose(patch_crop, (1, 2, 0))  # Convert (C, W, H) to (H, W, C)

            # Resize to 256,256
            patch = np.array(Image.fromarray(patch).resize((256,256)))

            # Convert the image to float32 for precision, as required for normalization
            # Rescale values to between 0 and 1
            patch = patch.astype(np.float32) / 255.0

            patch = np.transpose(patch, (2, 0, 1))   # Convert (H, W, C) to (C, W, H)

            print(patch)

            # Normalize the image with ImageNet values (mean and std)
            mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)  # Reshape to (1, 1, 3) for broadcasting  
            std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)   # Reshape to (1, 1, 3) for broadcasting

            # Perform the normalization
            patch = (patch - mean) / std
                
            all_patches.append(patch)

            break

        reshaped = np.asarray(all_patches)

    return reshaped.astype(np.float32)


if __name__ == '__main__':
    
    # Prepare a test image
    # input_folder = '../CleanData/Evaluation/Deployment/Combined/Deploy'
    input_folder = '../Rosbag-images'
    save_folder = 'outputs/'

    
    ################## ONNX model WITHOUT preprocessing ####################
    print("################## ONNX model WITHOUT preprocessing ####################")
    # Load the ONNX model
    onnx_model_path = "outputs/models/onnx/Mobilenet-28-3-256-256.onnx"
    ort_session = ort.InferenceSession(onnx_model_path)

    all_images = [i for i in os.listdir(input_folder)] 

    step = 0
    print("Processing images...")
    for filename in tqdm(all_images):

        image_path = os.path.join(input_folder, filename)

        image = cv2.imread(image_path)
        input_np = preprocess_image(image, (256,256), 'patch')
        # print(np.shape(input_np))
        # print(input_np[0,:,0,0])
        # patches_1 = input_np

        # # Prepare the input in the expected format for ONNX Runtime
        # input_name = ort_session.get_inputs()[0].name
        # output_names = [output.name for output in ort_session.get_outputs()]

        # # Run the ONNX model
        # outputs = ort_session.run(output_names, {input_name: input_np})

        # # Extract outputs (in the order defined in your model)
        # class_pred = outputs[0]     # Softmax probabilities
        # ratio = outputs[1]          # Predicted class (argmax)
        # deploy = outputs[2]         # Ratio of top two softmax values

        # video_frames(class_pred, deploy, image, os.path.join(save_folder, "Mobilenet-28-3-256-256_"+filename[:-4]))

        # # Print the outputs to verify correctness
        # print("Class Predictions (Argmax):", class_pred)
        # print("Ratio of Classifications:", ratio)
        # print("Deployment Decision:", deploy)

        # step += 1
        # if step == 3:
        #     break
        break

    # ################## ONNX model WITH preprocessing ####################
    # print("################## ONNX model WITH preprocessing ####################")
    # # Load the ONNX model
    # onnx_model_path_2 = "outputs/models/onnx/Mobilenet-1-3-5312-3040.onnx"
    # ort_session_2 = ort.InferenceSession(onnx_model_path_2)

    # all_images = [i for i in os.listdir(input_folder)] 

    # step = 0
    # print("Processing images...")
    # for filename in tqdm(all_images):

    #     image_path = os.path.join(input_folder, filename)

    #     image = cv2.imread(image_path)
    #     image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     input_np = np.asarray(image_cv).astype(np.float32)
    #     input_np = np.transpose(input_np, (2, 1, 0)) 
    #     input_np = np.expand_dims(input_np, axis=0)

    #     # Prepare the input in the expected format for ONNX Runtime
    #     input_name = ort_session_2.get_inputs()[0].name
    #     output_names = [output.name for output in ort_session_2.get_outputs()]

    #     # Run the ONNX model
    #     outputs = ort_session_2.run(output_names, {input_name: input_np})

    #     # Extract outputs (in the order defined in your model)
    #     class_pred = outputs[0]     # Softmax probabilities
    #     ratio = outputs[1]          # Predicted class (argmax)
    #     deploy = outputs[2]         # Ratio of top two softmax values
    #     patches = outputs[4]
    #     # patches_2 = patches
    #     # print(np.shape(patches))
    #     print(outputs[4][0])
    #     print("######################################")

    #     print(outputs[5][0])

    #     # video_frames(class_pred, deploy, image, os.path.join(save_folder, "Mobilenet-1-3-5312-3040_"+filename[:-4]))

    #     # # Print the outputs to verify correctness
    #     # print("Class Predictions (Argmax):", class_pred)
    #     # print("Ratio of Classifications:", ratio)
    #     # print("Deployment Decision:", deploy)

    #     # difference = np.abs(patches_1[0] - patches_2[0])
    #     # # print(difference)
    #     # diff_sum = np.sum(difference)
    #     # print(diff_sum)

    #     # diff_gray = np.mean(difference, axis=0)

    #     # plt.imshow(diff_gray, cmap='hot')
    #     # plt.colorbar()
    #     # plt.title("Difference Heatmap")
    #     # plt.savefig('difference_heatmap.png', dpi=300, bbox_inches='tight')
    #     # plt.close()

    #     # step += 1
    #     # if step == 3:
    #     #     break
    #     break