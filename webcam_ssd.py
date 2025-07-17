import time
import cv2
import numpy as np
import torch

# print(cv2.getBuildInformation())

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

model.to('cuda')
model.eval()

# Define the labels for the objects
classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "appsink"
    )

# Initialize the webcam
# cap = cv2.VideoCapture("nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! nvvidconv ! appsink", cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to open camera")
    exit()

# Loop through the frames from the webcam
while True:

    # Get the height and width of the frame
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape

    start_time = time.time()

    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to the expected format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_input = frame.astype(np.float32) / 255.0
    frame_input = utils.rescale(frame_input, 300, 300)
    frame_input = utils.crop_center(frame_input, 300, 300)
    frame_input = utils.normalize(frame_input)

    # Expand dimensions if necessary
    if len(frame_input.shape) == 3:
        frame_input = np.expand_dims(frame_input, axis=0)

    frame_tensor = utils.prepare_tensor(frame_input)

    # Perform object detection inference
    with torch.no_grad():
        predictions = model(frame_tensor)

    # Inspect the structure of predictions
    # print("Type of predictions:", type(predictions))

    # If predictions is a list or tuple, inspect each element
    # if isinstance(predictions, (list, tuple)):
    #     for i, pred in enumerate(predictions):
    #         print(f"Shape of predictions[{i}]:", pred.shape)
    #         print(f"Type of predictions[{i}]:", type(pred))
    #         print(f"Sample data of predictions[{i}]:", pred)

    # # If predictions is a tensor, inspect its shape and type
    # elif isinstance(predictions, torch.Tensor):
    #     print("Shape of predictions:", predictions.shape)
    #     print("Type of predictions:", predictions.dtype)
    #     print("Sample data of predictions:", predictions)        

    # # pick the most confident prediction
    # exit()
    results_per_input = utils.decode_results(predictions)
    best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
    boxes, labels, scores = best_results_per_input[0]

    # Loop through the predictions and draw bounding boxes and labels on the frame
    for box, label, score in zip(boxes, labels, scores):
        # Convert normalized coordinates to absolute pixel coordinates
        x1, y1, x2, y2 = (box * np.array([300, 300, 300, 300])).astype(int)
        # Scale the coordinates back to the original frame size
        x1 = int(x1 * width / 300)
        y1 = int(y1 * height / 300)
        x2 = int(x2 * width / 300)
        y2 = int(y2 * height / 300)

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