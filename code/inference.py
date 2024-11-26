import torch
from torchvision import transforms
from PIL import Image
from model import UNet  # Ensure this import points to where your UNet model is defined
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=10)  # Adjust n_classes if your model differs
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the expected input dimension for the model
        transforms.ToTensor(),          # Convert to tensor (and scale to [0, 1])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension [N, C, H, W]
    return image

def predict_image(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    return output  # Maintain batch dimension for further processing

def process_output(output):
    output = torch.softmax(output.squeeze(0), dim=0)  # Apply softmax to output to convert to probability distribution
    class_labels = torch.argmax(output, dim=0)  # Find the class with the highest probability
    return class_labels

def visualize_mask(mask):
    plt.imshow(mask.cpu().numpy(), cmap='gray')  # Convert tensor to numpy array for visualization
    plt.colorbar()
    plt.title("Predicted Class Labels")
    plt.show()

def save_class_labels(class_labels, filename='predicted_mask.png'):
    save_image(class_labels.unsqueeze(0).float() / class_labels.max(), filename)  # Normalize and save image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_path = '/home/shobot/Shahbaz_project/SAR_RARP50_challenge/code/unet_model.pth'
    image_path = '/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/Test/video_41/frames/000000000.png'

    # Load the model
    model = load_model(model_path, device)

    # Preprocess the image
    image = preprocess_image(image_path)

    # Perform inference
    output = predict_image(model, image, device)

    # Process the output to get class labels
    class_labels = process_output(output)
    print(class_labels)

    # Visualize the class labels
    visualize_mask(class_labels)

    # Optionally, save the class labels as an image
    save_class_labels(class_labels, 'predicted_mask.png')

    print("Inference completed and mask visualized/saved.")

if __name__ == '__main__':
    main()
