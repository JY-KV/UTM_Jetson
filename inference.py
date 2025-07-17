''' Classification inference script '''
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from schema import InferenceResponse

class Inference():
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

    def predict(self, image_buffer):
        """
        Perform prediction on input data
        """
        pil_image = Image.open(BytesIO(image_buffer))

        input_tensor = self.transform(pil_image)
        image_tensor = input_tensor.unsqueeze(0).to('cuda')

         # Perform object detection inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Get the predicted class
        _, predicted_idx = torch.max(predictions, 1)
        predicted_label = self.categories[predicted_idx.item()]

        return InferenceResponse(class_name=predicted_label)
