import cv2
import numpy as np
import os

# Set base folders
frame_base_folder = '../frames'
predicted_base_folder = 'predicted'

def extract_color_histogram(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate histogram for each channel (R, G, B)
    hist = [np.histogram(frame_rgb[:,:,i], bins=256, range=(0,256))[0] for i in range(3)]

    # Concatenate histograms into a single feature vector
    return np.concatenate(hist)

def process_frames_in_batches(frames_folder, batch_size=100):
    feature_vectors = []
    for filename in os.listdir(frames_folder):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(frames_folder, filename)
            frame = cv2.imread(frame_path)
            feature_vector = extract_color_histogram(frame)
            feature_vectors.append(feature_vector)
            if len(feature_vectors) == batch_size:
                yield np.array(feature_vectors)
                feature_vectors = []
    if feature_vectors:
        yield np.array(feature_vectors)

# Get folder containing frames
frames_folder_name = input("ENTER FOLDER NAME CONTAINING FRAMES (e.g., Air_Force_One): ")
frames_folder = os.path.join(frame_base_folder, frames_folder_name)
print(frames_folder)

# Read all the images from the folder and extract features in batches
feature_matrix_batches = process_frames_in_batches(frames_folder)
feature_matrix = np.concatenate([batch for batch in feature_matrix_batches])
print("Shape of the feature matrix:", feature_matrix.shape)

# Determine number of clusters (15% of total frames)
k = int(feature_matrix.shape[0] * 0.15)
print("Value of k:", k)

# Perform K-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, _ = cv2.kmeans(feature_matrix.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Get cluster labels for each image
cluster_labels = labels.flatten()

# Get representative frames 
representative_frame_dict = {}
for frame_index, frame_cluster in enumerate(cluster_labels):
    representative_frame_dict.setdefault(frame_cluster, []).append(frame_index)

# Calculate the median frame index for each cluster
frame_stream = [int(np.median(frame_indices)) for frame_indices in representative_frame_dict.values() if frame_indices]
frame_stream.sort()

# Process and save representative frames
selected_frames = np.zeros(len(cluster_labels))
selected_frames[frame_stream] = 1

# Save selected frames to disk
predicted_feature_path = os.path.join(predicted_base_folder, f"{frames_folder_name}.npy")
np.save(predicted_feature_path, selected_frames)
