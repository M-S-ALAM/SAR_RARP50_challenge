import torch
from torchvision import transforms
from PIL import Image
from model.unet import UNet  # Ensure this import points to where your UNet model is defined
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class UNetInference:
    """
    A class to handle loading a UNet model, preprocessing images, performing inference,
    and visualizing or saving the predicted segmentation masks.
    """

    def __init__(self, model_path, device=None, input_size=(256, 256)):
        """
        Initialize the UNetInference object.

        Args:
            model_path (str): Path to the saved model file.
            device (torch.device, optional): Device for inference. Defaults to GPU if available.
            input_size (tuple, optional): Input size for the model. Defaults to (256, 256).
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.model = self.load_model()

    def load_model(self):
        """
        Load the UNet model from the given path.

        Returns:
            torch.nn.Module: Loaded UNet model.
        """
        model = UNet(n_channels=3, n_classes=10)  # Adjust n_classes if needed
        state_dict = torch.load(self.model_path, map_location=self.device)

        # Remove 'module.' prefix if it exists
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_path):
        """
        Preprocess an input image for inference.

        Args:
            image_path (str): Path to the input image.

        Returns:
            torch.Tensor: Preprocessed image tensor with batch dimension.
        """
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])
        image = transform(image)
        return image.unsqueeze(0)  # Add batch dimension [N, C, H, W]

    def predict(self, image):
        """
        Perform inference on a preprocessed image.

        Args:
            image (torch.Tensor): Preprocessed image tensor.

        Returns:
            torch.Tensor: Model's raw output.
        """
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image)
        return output

    def process_output(self, output):
        """
        Process the raw model output to obtain class labels.

        Args:
            output (torch.Tensor): Raw model output.

        Returns:
            torch.Tensor: Tensor containing class labels for each pixel.
        """
        output = torch.softmax(output.squeeze(0), dim=0)  # Convert to probability distribution
        return torch.argmax(output, dim=0)  # Get class labels

    def visualize_mask(self, mask):
        """
        Visualize the predicted class labels.

        Args:
            mask (torch.Tensor): Tensor containing class labels.
        """
        plt.imshow(mask.cpu().numpy(), cmap='gray')  # Convert tensor to numpy array for visualization
        plt.colorbar()
        plt.title("Predicted Class Labels")
        plt.show()

    def save_class_labels(self, class_labels, filename='predicted_mask.png'):
        """
        Save the predicted class labels as an image.

        Args:
            class_labels (torch.Tensor): Tensor containing class labels.
            filename (str): Path to save the output image.
        """
        save_image(class_labels.unsqueeze(0).float() / class_labels.max(), filename)

    def run_inference(self, image_path, save_path='predicted_mask.png'):
        """
        Perform the full inference pipeline: preprocess image, run inference,
        process the output, visualize the mask, and save the results.

        Args:
            image_path (str): Path to the input image.
            save_path (str, optional): Path to save the predicted mask image. Defaults to 'predicted_mask.png'.
        """
        # Preprocess the input image
        image = self.preprocess_image(image_path)

        # Perform inference
        output = self.predict(image)

        # Process the output to get class labels
        class_labels = self.process_output(output)

        # Visualize the predicted mask
        self.visualize_mask(class_labels)

        # Save the predicted mask
        self.save_class_labels(class_labels, save_path)

        print(f"Inference completed. Predicted mask saved to {save_path}")
        return output


if __name__ == '__main__':
    # Example usage
    model_path = "C:\\Users\\User\\Desktop\\Shahbaz_project\\SAR_RARP50_challenge\\code\\unet_model.pth"
    image_path = "C:\\Users\\User\Desktop\\Shahbaz_project\\SAR_RARP50_challenge\\Database\\Test\\video_41\\frames\\000000480.png"
    save_path = 'predicted_mask.png'

    # Create an instance of the UNetInference class
    unet_inference = UNetInference(model_path=model_path)

    # Run the inference pipeline
    output = unet_inference.run_inference(image_path=image_path, save_path=save_path)
