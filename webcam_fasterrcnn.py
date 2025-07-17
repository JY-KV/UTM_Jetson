import time
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)

# Define the labels for the objects
classes = [
    '__background__', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish',
    'stingray'
]

# Retrieve the number of input features for the classification head
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the original classification head with a new one tailored to our number of classes
# The new head will have 'n_classes' outputs, corresponding to the number of categories in our dataset
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, len(classes))

model_load_path = 'fasterrcnn_fish_detect_statedict.pth'
model.load_state_dict(torch.load(model_load_path, map_location="cuda"))
model.to("cuda").eval()

gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "appsink"
    )

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the transformation to be applied to the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

# Loop through the frames from the webcam
while True:
    start_time = time.time()

    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to the expected format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame).unsqueeze(0).to("cuda")

    # Perform object detection inference
    with torch.no_grad():
        predictions = model(frame_tensor)

    # Get the predicted bounding boxes, labels, and scores
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Loop through the predictions and draw bounding boxes and labels on the frame
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.6:  # Only show predictions with a confidence score greater than 0.5
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, str(classes[label]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Calculate and display the FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Object Detection', rgb_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()