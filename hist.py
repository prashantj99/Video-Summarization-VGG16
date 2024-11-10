import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to extract color histogram features for a frame
def extract_color_histogram(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate histogram for each channel (R, G, B)
    hist_r = np.histogram(frame_rgb[:,:,0], bins=256, range=(0,256))[0]
    hist_g = np.histogram(frame_rgb[:,:,1], bins=256, range=(0,256))[0]
    hist_b = np.histogram(frame_rgb[:,:,2], bins=256, range=(0,256))[0]

    # Concatenate histograms into a single feature vector
    feature_vector = np.concatenate((hist_r, hist_g, hist_b))

    return feature_vector

# Read the input video
video_path = 'vid.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize variables
frames = []
feature_dim = 256

# Extract frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Extract color histogram features for each frame
feature_vectors = [extract_color_histogram(frame) for frame in frames]

# Check the dimensions of the feature vectors
assert all(len(vec) == feature_dim * 3 for vec in feature_vectors), "Dimension mismatch in feature vectors"

# Select a random frame index for displaying histogram
frame_index = np.random.randint(0, len(frames))
selected_frame = frames[frame_index]

# Plot the histogram of the selected frame
plt.figure(figsize=(8, 6))
plt.title('Histogram of Selected Frame')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.hist(selected_frame[:,:,0].ravel(), bins=256, range=(0,256), color='red', alpha=0.5, label='Red')
plt.hist(selected_frame[:,:,1].ravel(), bins=256, range=(0,256), color='green', alpha=0.5, label='Green')
plt.hist(selected_frame[:,:,2].ravel(), bins=256, range=(0,256), color='blue', alpha=0.5, label='Blue')
plt.legend()
plt.grid(True)
plt.show()

# Print the shape of the feature matrix
feature_matrix = np.array(feature_vectors).reshape(len(frames), feature_dim * 3)
print("Shape of the feature matrix:", feature_matrix.shape)
