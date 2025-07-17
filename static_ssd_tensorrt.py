import time
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

BATCH_SIZE = 10
target_dtype = np.float16

f = open("ssd_engine_pytorch.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# need to set input and output precisions to FP16 to fully enable it
output = np.empty([BATCH_SIZE, 2, 8732, max(4, 81)], dtype=target_dtype)

stream = cuda.Stream()

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


# read image
frame = cv2.imread("../images/cat_1.jpg")

# allocate device memory
frame_sample = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_sample = cv2.resize(frame_sample, (300, 300))
input_batch = np.array(np.repeat(np.expand_dims(np.array(frame_sample, dtype=np.float32), axis=0), BATCH_SIZE, axis=0), dtype=np.float32)

d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
context.set_tensor_address(engine.get_tensor_name(1), int(d_output))

def preprocess_image(img):
    '''
    PyTorch has a normalization that it applies by default
    in all of its pretrained vision models.
    We can preprocess our images to match this normalization
    by the following, making sure our final result is in FP16 precision
    '''
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
    return np.array(result, dtype=np.float16)

def predict(batch): # result gets copied into output
    '''
    Prediction Function

    This involves a copy from CPU RAM to GPU VRAM,
    executing the model,
    then copying the results back from GPU VRAM to CPU RAM
    '''
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v3(stream_handle=stream.handle)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output

print("Warming up...")

preprocessed_images = np.array([preprocess_image(image) for image in input_batch])
pred = predict(preprocessed_images)

print("Done warming up!")

# Define the transformation to be applied to the input image
transform = transforms.Compose([
    transforms.ToTensor()
])


# Convert the frame to the expected format
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
original_height, original_width = frame.shape[:2]
resized_frame = cv2.resize(frame, (300, 300))
preprocessed_images = np.array(np.repeat(np.expand_dims(np.array(resized_frame, dtype=np.float32), axis=0), BATCH_SIZE, axis=0), dtype=np.float32)

# Perform object detection inference
with torch.no_grad():
    predictions = predict(preprocessed_images)

# Extract the bounding boxes and class scores from the predictions
boxes = predictions[0]  # Shape: [1, 4, 8732]
scores = predictions[1]  # Shape: [1, 81, 8732]
print("Boxes ", boxes)
print("Scores ", scores)

# Loop through the predictions and draw bounding boxes and labels on the frame
for i in range(boxes.shape[2]):
    box = boxes[0, :, i]
    score = scores[0, :, i]
    label = np.argmax(score)

    if score[label] > 0.9:  # Only show predictions with a confidence score greater than 0.9
        x1, y1, x2, y2 = box
        # Scale the coordinates back to the original frame size
        x1 = int(x1 * original_width / 300)
        y1 = int(y1 * original_height / 300)
        x2 = int(x2 * original_width / 300)
        y2 = int(y2 * original_height / 300)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, str(classes[label]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the frame
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imshow('Object Detection', rgb_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()