import cv2
import os

# Directory to save frames
save_dir = input("ENTER DIR NAME TO SAVE FRAMES: ")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Read the input video
video_path = input("ENTER FILE PATH: ")
cap = cv2.VideoCapture(video_path)

# Initialize variables
frame_count = 0

# Process the video
while True:
    # Read a frame
    ret, frame = cap.read()

    # Check if frame is None (end of video)
    if frame is None:
        break

    # Save frame as image
    frame_filename = os.path.join(save_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1


print("NUMBER OF FRAMES EXTRACTED: "+str(frame_count))

# Release the video capture object
cap.release()
