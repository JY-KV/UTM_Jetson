import cv2
import os
import random
import shutil

gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "appsink"
    )

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
            # Display the frame with the frame counter and class name
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Webcam', display_frame)

            # Save the frame without any text and resize it to 400x400
            resized_frame = cv2.resize(frame, (400, 400))
            img_path = f"{train_dir}/{frame_count}.jpg"
            cv2.imwrite(img_path, resized_frame)
            frame_count += 1

        # Delay of 100ms between frames to achieve 10 frames per second
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
        num_frames = 6 * 20
        capture_frames(class_name, num_frames)

        # Split train and validation images
        split_train_validation(class_name)

    print("Data collection and splitting completed.")

if __name__ == "__main__":
    main()