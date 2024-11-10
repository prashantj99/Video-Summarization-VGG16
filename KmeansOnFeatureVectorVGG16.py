import cv2
import numpy as np
import os

# Set base folders
features_matrix_base_folder = 'feature_matrices'
frame_base_folder = 'frames'
rep_frame_base_folder = 'RepresentativeFrames'
predicted_base_folder = 'predicted'

# Get folder containing frames
frames_folder_name = input("ENTER FOLDER NAME CONTAINING FRAMES (e.g., Air_Force_One): ")
frames_folder = os.path.join(frame_base_folder, frames_folder_name)
print(frames_folder)

# Enter output video file name
output_video_file_name = input("Enter output video filename (e.g., video.mp4): ")

# Set path for extracted feature vector
extracted_feature_filename = input("Enter Extracted Feature Vector Filename (e.g., feature.npy): ")
extracted_feature_file_path = os.path.join(features_matrix_base_folder, extracted_feature_filename)

# Load feature matrix
feature_matrix = np.load(extracted_feature_file_path)
print("Shape of the loaded feature matrix:", feature_matrix.shape)

# Determine number of clusters (15% of total frames)
k = int(feature_matrix.shape[0] * 0.15)
print("Value of k:", k)

# Perform K-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centroids = cv2.kmeans(feature_matrix.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Get cluster labels for each image
cluster_labels = labels.flatten()

# Get representative frames 
representative_frame_dict = {}
for frame_index, frame_cluster in enumerate(cluster_labels):
    representative_frame_dict.setdefault(frame_cluster, []).append(frame_index)

# Calculate the median frame index for each cluster
frame_stream = []
for frame_cluster, frame_indices in representative_frame_dict.items():
    if len(frame_indices) > 0:
        median_index = np.median(frame_indices).astype(int)  # Calculate the median index
        frame_stream.append(median_index)

frame_stream.sort()

# Create folder for representative frames if it doesn't exist
representative_frames_folder_path = os.path.join(rep_frame_base_folder, frames_folder_name)
os.makedirs(representative_frames_folder_path, exist_ok=True)

# Process and save representative frames
frames = []
selected_frames = np.zeros(len(cluster_labels))
for frame_index in frame_stream:
    selected_frames[frame_index] = 1  # Mark selected frame
    path = os.path.join(frames_folder, f"img_{frame_index+1:05d}.jpg")
    frame_img = cv2.imread(path)
    output_file = os.path.join(representative_frames_folder_path, f"img_{frame_index+1:05d}.jpg")
    cv2.imwrite(output_file, frame_img)
    frames.append(frame_img)

# Define output video parameters
fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Write representative frames to video
video_writer = cv2.VideoWriter(output_video_file_name, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
for frame in frames:
    video_writer.write(frame)

# Release video writer
video_writer.release()

# Save selected frames to disk
predicted_feature_path = os.path.join(predicted_base_folder, f"{frames_folder_name}.npy")
np.save(predicted_feature_path, selected_frames)

# Success message
print("MP4 video created successfully.")
