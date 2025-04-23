import cv2
import os

# Folder containing images
image_folder = "Models-Jan2025/heron-inference-model-1740442931-ordered"
video_name = "output_video.mp4"
fps = 5  # Frames per second

# Get all image files sorted by name
images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

# Read the first image to get the frame size
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Add images to video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release resources
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_name}")
