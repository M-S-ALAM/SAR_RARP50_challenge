import pandas as pd
import matplotlib.pyplot as plt
import os


class TrainingMetricsVisualizer:
    """
    A class to visualize training metrics and learning rate over epochs.
    """

    def __init__(self, data_directory, save_directory):
        """
        Initialize the visualizer with the path to the data and save directory.

        Args:
            data_directory (str): Path to the CSV file containing training metrics.
            save_directory (str): Path to the directory where plots will be saved.
        """
        self.df = pd.read_csv(data_directory)
        self.save_directory = save_directory

        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)

    def plot_metrics(self):
        """
        Plot and save metrics graphs with annotations for each epoch.

        Plots include:
            - Train and Test Loss
            - Train and Test IoU
            - Train and Test Dice
            - Train and Test mIoU
            - Train and Test mNSD
            - Train and Test Segmentation Score
        """
        # Define the metrics to plot
        plots = [
            ("train_loss", "test_loss", "Loss"),
            ("train_iou", "test_iou", "IoU"),
            ("train_dice", "test_dice", "Dice"),
            ("train_miou", "test_miou", "Mean Intersection over Union"),
            ("train_mnsd", "test_mnsd", "Mean Normalized Surface Dice"),
            ("train_seg_score", "test_seg_score", "Segmentation Score"),
        ]

        # Plot each metric
        for train_metric, test_metric, ylabel in plots:
            plt.figure()

            # Plot training and testing metrics
            plt.plot(
                self.df["epoch"], self.df[train_metric],
                label=f"Train {ylabel}", marker='o', color='blue', linestyle='-'
            )
            plt.plot(
                self.df["epoch"], self.df[test_metric],
                label=f"Test {ylabel}", marker='o', color='green', linestyle='-'
            )

            # Annotate with blue circles
            plt.scatter(self.df["epoch"], self.df[train_metric], color='blue', s=50, label=None)
            plt.scatter(self.df["epoch"], self.df[test_metric], color='green', s=50, label=None)

            # Add labels, title, and legend
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} over Epochs")
            plt.legend()
            plt.grid()

            # Save the plot
            save_path = os.path.join(self.save_directory, f"{ylabel.lower().replace(' ', '_')}_plot.png")
            plt.savefig(save_path)
            plt.show()

    def plot_learning_rate(self):
        """
        Plot and save the learning rate over epochs.
        """
        plt.figure()

        # Plot learning rate
        plt.plot(
            self.df["epoch"], self.df["learning_rate"],
            marker='o', color='blue', linestyle='-'
        )
        plt.scatter(self.df["epoch"], self.df["learning_rate"], color='blue', s=50)

        # Add labels, title, and grid
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate over Epochs")
        plt.grid()

        # Save the plot
        save_path = os.path.join(self.save_directory, "learning_rate_plot.png")
        plt.savefig(save_path)
        plt.show()

    def run(self):
        """
        Execute the full pipeline: Plot all metrics and the learning rate.
        """
        self.plot_metrics()
        self.plot_learning_rate()


if __name__ == '__main__':
    # Paths
    data_file = "C:\\Users\\User\Desktop\Shahbaz_project\SAR_RARP50_challenge\code\\training_metrics.csv"
    save_dir = "C:\\Users\\User\Desktop\Shahbaz_project\SAR_RARP50_challenge\plots"

    # Instantiate and run the visualizer
    visualizer = TrainingMetricsVisualizer(data_file, save_dir)
    visualizer.run()
