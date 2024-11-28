import argparse
import numpy as np
import cv2
import random
import pandas as pd
from matplotlib import pyplot as plt
from inference import UNetInference

inference = UNetInference(model_path="C:\\Users\\User\\Desktop\\Shahbaz_project\\SAR_RARP50_challenge\\unet_model.pth")

class SegmentationVisualizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
        self.class_colors = {
            0: (0, 0, 0),      # Background
            1: (255, 0, 0),    # Tool clasper
            2: (0, 255, 0),    # Tool wrist
            3: (0, 0, 255),    # Tool shaft
            4: (255, 255, 0),  # Suturing needle
            5: (255, 0, 255),  # Thread
            6: (0, 255, 255),  # Suction tool
            7: (128, 0, 128),  # Needle Holder
            8: (128, 128, 0),  # Clamps
            9: (0, 128, 128),  # Catheter
        }

    def load_data(self):
        try:
            return pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            return pd.DataFrame()

    def visualize_random_samples(self, num_samples=5):
        if self.data.empty:
            print("Error: No data available for visualization.")
            return

        for _ in range(num_samples):
            random_index = random.randint(0, len(self.data) - 1)
            image_path = self.data.iloc[random_index]['Frame']
            mask_path = self.data.iloc[random_index]['Segmentation']

            output = inference.run_inference(image_path)

            # Read the input image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

            # Read mask as grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            prediction_mask = cv2.imread("C:\\Users\\User\\Desktop\\Shahbaz_project\\SAR_RARP50_challenge\\code\\predicted_mask.png", cv2.IMREAD_GRAYSCALE)

            if mask is None or image is None:
                print(f"Error: Could not read image or mask at index {random_index}.")
                continue

            # Normalize prediction mask (if it contains intensity values)
            prediction_mask = (prediction_mask // (256 // 10))  # Map values to range 0–9

            # Create colorized masks
            color_mask = self.create_color_mask(mask)
            prediction_color_mask = self.create_color_mask(prediction_mask)

            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
            color_mask = cv2.resize(color_mask, (256, 256), interpolation=cv2.INTER_CUBIC)

            # Plot the results
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(color_mask)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            prediction_color_mask = cv2.resize(prediction_color_mask, (256, 256))
            plt.title("Prediction Truth")
            plt.imshow(prediction_color_mask)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    def create_color_mask(self, mask):
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in self.class_colors.items():
            color_mask[mask == cls] = color
        return color_mask


if __name__=='__main__':
    inference_plot_sample = SegmentationVisualizer(data_path="C:\\Users\\User\\Desktop\\Shahbaz_project\\SAR_RARP50_challenge\\Database\\test_data.csv")
    inference_plot_sample.visualize_random_samples()
