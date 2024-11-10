import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Folders and file names
features_matrix_base_folder = 'feature_matrices'
frame_base_folder = 'frames'

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Get user inputs for directories and file names
frames_folder_name = input("Enter Frame Folder Name (e.g. Air_Force_One): ")
output_file_name = input("Enter Output Filename (e.g. Air_Force_One.npy):")

frame_folder_path = os.path.join(frame_base_folder, frames_folder_name)
output_file_path = os.path.join(features_matrix_base_folder, output_file_name)

# Prepare a list for features
all_features = []

# Process images in the folder
image_files = [f for f in os.listdir(frame_folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]
batch_size = 64  # Adjust the batch size as needed

for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i + batch_size]
    batch_features = []

    for filename in batch_files:
        img_path = os.path.join(frame_folder_path, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict features and append to batch
        features = model.predict(x)
        features = features.flatten()
        batch_features.append(features)

    # Extend all_features with batch_features
    all_features.extend(batch_features)
    print(f"Processed {i + batch_size}/{len(image_files)} images...")

# Convert to numpy array and save
all_features = np.array(all_features)
print(f"Shape of the feature matrix: {all_features.shape}")
np.save(output_file_path, all_features)
