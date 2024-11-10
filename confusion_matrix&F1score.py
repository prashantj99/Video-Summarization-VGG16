from sklearn.metrics import confusion_matrix
import numpy as np

# Function to calculate precision, recall, and F1-score
def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


# Function to print confusion matrix with labels
def print_confusion_matrix(cm, labels):
    tn, fp, fn, tp = cm.ravel()
    print(f"{'Actual\\Predicted':<20}{labels[0]:<10}{labels[1]:<10}")
    print(f"{labels[0]:<20}{tp:<10}{fn:<10}")
    print(f"{labels[1]:<20}{fp:<10}{tn:<10}")

# Prompt the user to enter file names for actual and predicted data
actual_file = input("Enter the file name for actual data (e.g., actual.npy): ")
predicted_file = input("Enter the file name for predicted data (e.g., predicted.npy): ")

# Load actual and predicted data from the specified files
actual = np.load(actual_file)
predicted = np.load(predicted_file)

# Compute confusion matrix
cm = confusion_matrix(actual, predicted)

# Print confusion matrix with labels
print("\nConfusion Matrix:")
print_confusion_matrix(cm, ["Positive", "Negative"])

# Calculate and print precision, recall, and F-score
precision, recall, f_score = calculate_metrics(cm)
print("\nMetrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-score: {f_score:.4f}")
