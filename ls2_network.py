import torch
from torchvision import transforms # type: ignore
import cv2
from PIL import Image

class LS2Network:

    def __init__(self, net_path, logger=None):
        """
        Initialize the LS2Network with the specified model path.
        """

        self.logger = logger

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.logger:
                self.logger.info("[LS2] CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")
            if self.logger:
                self.logger.info("[LS2] CUDA is not available. Using CPU.")


        self.net_path = net_path
        self.load_net(net_path)
        self.net.to(self.device)
        self.img = None
        self.net.eval()
    

    def load_net(self, net_path):
        """
        Load the neural network model from the specified path.
        """
        if net_path is None:
            return None

        # Define the model architecture
        from torchvision import models
        self.net = models.mobilenet_v2(pretrained=False)
        self.net.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.net.last_channel, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1),
        )

        # Load the state dictionary
        state_dict = torch.load(net_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        if self.logger:
            self.logger.info(f"[LS2] Loaded model from {net_path}")
    
    def crop_to_poi(self, image, poi, size):
        """
        Crop the image to the region of interest (poi).
        """
        x, y = poi
        height, width = image.shape[:2]

        # Calculate the top-left corner of the region to center the poi
        x_start = x - size // 2
        y_start = y - size // 2

        # Ensure the region is within bounds
        x_start = max(0, min(x_start, width - size))
        y_start = max(0, min(y_start, height - size))

        # Crop the image
        cropped_image = image[y_start:y_start + size, x_start:x_start + size]

        # Convert to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        self.img = pil_image
    

    def run_net(self):
        """
        Run the neural network on the cropped image.
        """
        if self.img is None:
            if self.logger:
                self.logger.error("[LS2] No image has been set for processing.")
            return None
        
        # Define transform to match expected input
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transform to the image
        img_tensor = transform(self.img)
        
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(self.device)

        # Run the network
        with torch.no_grad():
            output = self.net(img_tensor)

        if self.logger:
            self.logger.info(f"[LS2] Network output: {output.item()}")
        
        return torch.sigmoid(output).item() # Return the probability score
