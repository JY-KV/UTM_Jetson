import os
import time
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=True)
model_load_path = os.path.expanduser('~/camera_classification.pth')
model.load_state_dict(torch.load(model_load_path, map_location=device))
model.to('cuda')
model.eval()

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get the height and width of the frame
ret, frame = cap.read()
height, width, _ = frame.shape

# Loop through the frames from the webcam
while True:
    start_time = time.time()

    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to the expected format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    
    input_tensor = preprocess(pil_image)
    frame_tensor = input_tensor.unsqueeze(0).to('cuda')

    # Perform object detection inference
    with torch.no_grad():
        predictions = model(frame_tensor)

    # Get the predicted class
    _, predicted_idx = torch.max(predictions, 1)
    predicted_label = categories[predicted_idx.item()]

    # Display the predicted label on the frame
    cv2.putText(frame, predicted_label, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

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