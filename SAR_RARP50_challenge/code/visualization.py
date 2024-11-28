import argparse
import numpy as np
import cv2
import random
import pandas as pd
from matplotlib import pyplot as plt

class SegmentationVisualizer:
    """
    A class to handle loading and visualizing segmentation data.
    """

    def __init__(self, data_path):
        """
        Initialize the visualizer with the path to the dataset.

        Args:
            data_path (str): Path to the CSV file containing image and mask paths.
        """
        self.data_path = data_path
        self.data = self.load_data()

        # Define color map
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
        """
        Load the dataset from the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset as a DataFrame.
        """
        try:
            return pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            return pd.DataFrame()

    def visualize_random_samples(self, num_samples=5):
        """
        Visualize a random set of segmentation samples.

        Args:
            num_samples (int): Number of random samples to visualize.
        """
        if self.data.empty:
            print("Error: No data available for visualization.")
            return

        for _ in range(num_samples):
            random_index = random.randint(0, len(self.data) - 1)
            image_path = self.data.iloc[random_index]['Frame']
            mask_path = self.data.iloc[random_index]['Segmentation']

            # Read the input image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

            # Read mask as grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None or image is None:
                print(f"Error: Could not read image or mask at index {random_index}.")
                continue

            # Create colorized mask
            color_mask = self.create_color_mask(mask)

            # Plot the results
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.title("Input Image")
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Ground Truth")
            plt.imshow(color_mask)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    def create_color_mask(self, mask):
        """
        Map grayscale mask values to RGB colors.

        Args:
            mask (np.ndarray): Grayscale mask with class labels.

        Returns:
            np.ndarray: RGB colorized mask.
        """
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in self.class_colors.items():
            color_mask[mask == cls] = color
        return color_mask


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize segmentation samples.")
    parser.add_argument(
        "--data_path", type=str, required=True, default="C:\\Users\\User\\Desktop\\Shahbaz_project\\SAR_RARP50_challenge\\Database\\test_data.csv",
        help="Path to the CSV file containing image and mask paths."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of random samples to visualize (default: 5)."
    )
    return parser.parse_args()


def main():
    #args = parse_args()
    data_path = "C:\\Users\\User\\Desktop\\Shahbaz_project\\SAR_RARP50_challenge\\Database\\test_data.csv"
    visualizer = SegmentationVisualizer(data_path)
    num_samples=1
    visualizer.visualize_random_samples(num_samples=num_samples)


if __name__ == "__main__":
    main()
