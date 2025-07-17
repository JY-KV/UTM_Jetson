import cv2
import os
import random
import shutil

def capture_frames(class_name, num_frames):
    # Create directory for train images
    train_dir = f"train/{class_name}"
    os.makedirs(train_dir, exist_ok=True)

    # Capture frames from webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if ret:
            # Save frame as image
            img_path = f"{train_dir}/{frame_count}.jpg"
            cv2.imwrite(img_path, frame)
            frame_count += 1
        cv2.waitKey(100)  # Delay of 100ms between frames
    cap.release()

def split_train_validation(class_name):
    # Create directory for validation images
    validation_dir = f"validation/{class_name}"
    os.makedirs(validation_dir, exist_ok=True)

    # Move random images from train to validation
    train_images = os.listdir(f"train/{class_name}")
    num_validation = int(len(train_images) * 0.3)
    validation_images = random.sample(train_images, num_validation)
    for image in validation_images:
        src = f"train/{class_name}/{image}"
        dst = f"validation/{class_name}/{image}"
        shutil.move(src, dst)

def main():
    # Ask for number of classes
    num_classes = int(input("How many classes do you want to capture? (max 4): "))
    if num_classes > 4:
        print("Maximum number of classes is 4.")
        return

    for i in range(num_classes):
        # Ask for class name
        class_name = input(f"Enter the name for class {i+1}: ")

        # Capture frames for 1 minute (10 frames per second)
        num_frames = 60 * 10
        capture_frames(class_name, num_frames)

        # Split train and validation images
        split_train_validation(class_name)

    print("Data collection and splitting completed.")

if __name__ == "__main__":
    main()