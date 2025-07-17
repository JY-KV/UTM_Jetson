import time
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

BATCH_SIZE = 32
target_dtype = np.float16

f = open("resnet50_engine_pytorch.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# need to set input and output precisions to FP16 to fully enable it
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 

with open("../imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# read image
frame = cv2.imread("../images/cat_1.jpg")
height, width, _ = frame.shape

# allocate device memory
frame_sample = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_sample = cv2.resize(frame_sample, (224, 224))
img = np.expand_dims(np.array(frame_sample, dtype=np.float32), axis=0) # Expand image to have a batch dimension
input_batch = np.array(np.repeat(img, BATCH_SIZE, axis=0), dtype=np.float32) # Repeat across the batch dimension

d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)
context.set_tensor_address(engine.get_tensor_name(0), int(d_input)) # input buffer
context.set_tensor_address(engine.get_tensor_name(1), int(d_output)) #output buffer

stream = cuda.Stream()

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
predictions = predict(preprocessed_images)

print("Done warming up!")
start_time = time.time()
pred = predict(preprocessed_images)
elapsed_time = time.time() - start_time

# Get the predicted class
_, predicted_idx = torch.max(predictions, 1)
predicted_label = categories[predicted_idx.item()]

# Display the predicted label on the frame
cv2.putText(frame, predicted_label, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Calculate and display the FPS
fps = 1 / elapsed_time
cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the frame
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imshow('Object Detection', rgb_frame)

cv2.waitKey(0)

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()