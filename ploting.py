import numpy as np
import matplotlib.pyplot as plt

# Load binary arrays from .npy files
array1 = np.load('ground_truth/Base jumping.npy')
array2 = np.load('predicted/Base_jumping.npy')

# Create x-axis values (index + 1)
x_values = np.arange(1, len(array1) + 1)

# Create the plot
plt.plot(x_values, array1, 'r-', label='Ground Truth')
plt.plot(x_values, array2, 'g-', label='Predicted')

# Fill between the lines with colors
plt.fill_between(x_values, array1, array2, where=(array1 >= array2), color='red', alpha=0.3, interpolate=True)
plt.fill_between(x_values, array1, array2, where=(array1 < array2), color='green', alpha=0.3, interpolate=True)

# Add labels and legend
plt.xlabel('Frames')
plt.ylabel('Selected')
plt.title('Comparison of Ground Truth and Predicted For Base Jumping Video')
plt.legend()

# Show the plot
plt.show()
