import pandas as pd
import matplotlib.pyplot as plt


class TrainingMetricsVisualizer:
    def __init__(self, data_directory):
        self.df = pd.read_csv(data_directory)


    def plot_metrics(self):
        """Plot and save metrics graphs with blue circles for each epoch."""
        plots = [
            ("train_loss", "test_loss", "Loss"),
            ("train_iou", "test_iou", "IOU"),
            ("train_dice", "test_dice", "Dice"),
            ("train_miou", "test_miou", "Mean Intersection over Union"),
            ("train_mnsd", "test_mnsd", " Mean Normalized Surface Dice"),
            ("train_seg_score", "test_seg_score", "Segmentation Score"),
        ]

        for train_metric, test_metric, ylabel in plots:
            plt.figure()
            # Plot lines
            plt.plot(self.df["epoch"], self.df[train_metric], label=f"Train {ylabel}", marker='o', color='blue', linestyle='-')
            plt.plot(self.df["epoch"], self.df[test_metric], label=f"Test {ylabel}", marker='o', color='green', linestyle='-')

            # Annotate blue circles
            plt.scatter(self.df["epoch"], self.df[train_metric], color='blue', s=50, label=None)
            plt.scatter(self.df["epoch"], self.df[test_metric], color='blue', s=50, label=None)

            # Axis labels and title
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} over Epochs")
            plt.legend()
            plt.grid()
            plt.savefig(f"/home/shobot/Shahbaz_project/SAR_RARP50_challenge/plots/{ylabel.lower()}_plot.png")
            plt.show()

    def plot_learning_rate(self):
        """Plot and save the learning rate over epochs."""
        plt.figure()
        plt.plot(self.df["epoch"], self.df["learning_rate"], marker='o', color='blue', linestyle='-')
        plt.scatter(self.df["epoch"], self.df["learning_rate"], color='blue', s=50)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate over Epochs")
        plt.grid()
        plt.savefig("/home/shobot/Shahbaz_project/SAR_RARP50_challenge/plots/learning_rate_plot.png")
        plt.show()

    def run(self):
        """Execute the full pipeline."""
        self.plot_metrics()
        self.plot_learning_rate()

if __name__=='__main__':
    # Data
    data = '/home/shobot/Shahbaz_project/SAR_RARP50_challenge/code/training_metrics.csv'
    # Instantiate and run the visualizer
    visualizer = TrainingMetricsVisualizer(data)
    visualizer.run()
