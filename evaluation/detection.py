import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import argparse

# Define data transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class BadModelClassifier:
    def __init__(self, gpu="cuda:0", backdoor_type="style", model_path="model.pth"):
        self.device = torch.device(gpu)
        self.backdoor_type = backdoor_type
        self.model = self.load_model(model_path)
        self.class_names = ['benign', 'trigger']

    def load_model(self, model_path):
        if self.backdoor_type == 'object':
            model = models.resnet50(pretrained=False)
        else:
            model = models.resnet18(pretrained=False)
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
        
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        model.eval()  # Set model to evaluation mode
        return model
    
    def predict_single_image(self, img_path):
        """Predict the class of a single image."""
        try:
            img = Image.open(img_path).convert('RGB')
            img = data_transforms(img).unsqueeze(0).to(self.device)  # Add batch dimension
            with torch.no_grad():
                outputs = self.model(img)
                _, preds = torch.max(outputs, 1)
            
            if self.backdoor_type == 'pixel' or self.backdoor_type == 'style':
                preds = preds.item()
            else:
                preds = 1 - preds.item()
            
            print(f"Image: {img_path}, Predicted: {self.class_names[preds]}")
            return self.class_names[preds]
        
        except Exception as e:
            print(f"Error: {img_path}, {str(e)}")
            return None

    def compute_trigger_ratio(self, folder_path):
        """Compute the ratio of trigger images in a folder."""
        total_images = 0
        trigger_count = 0

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):  # Process only image files
                    img_path = os.path.join(root, file)
                    pred = self.predict_single_image(img_path)
                    
                    if pred == 'trigger':
                        trigger_count += 1
                    total_images += 1
        
        if total_images == 0:
            print(f"No valid images found in the folder: {folder_path}")
            return 0
        
        trigger_ratio = trigger_count / total_images
        print(f"Trigger ratio in folder '{folder_path}': {trigger_ratio:.4f}")
        return trigger_ratio


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect trigger images using a trained model.")
    parser.add_argument('--gpu', type=str, default="cuda:0", help="Choose GPU device")
    parser.add_argument('--backdoor_type', type=str, default='style', help="Choose backdoor type ('style', 'pixel', 'object')")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model file")
    parser.add_argument('--folder_path', type=str, help="Path to the folder to analyze")
    parser.add_argument('--image_path', type=str, help="Path to a single image to classify")
    
    args = parser.parse_args()
    
    classifier = BadModelClassifier(gpu=args.gpu, backdoor_type=args.backdoor_type, model_path=args.model_path)
    
    if args.image_path:
        # Process a single image if image_path is provided
        classifier.predict_single_image(args.image_path)
    elif args.folder_path:
        # Process a folder if folder_path is provided
        classifier.compute_trigger_ratio(args.folder_path)
    else:
        print("Please provide either an image path or a folder path.")
