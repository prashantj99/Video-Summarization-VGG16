import numpy as np

# Specify the path to your .npy file
file_path = input("file path(/folder/file_name): ")

# Load the .npy file
data = np.load(file_path)

# Set print options to display the full data without truncation
np.set_printoptions(threshold=np.inf)

# Print the full data
print(data)
