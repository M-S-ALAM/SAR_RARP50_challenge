import os
import pandas as pd
from tqdm import tqdm


class DataPreprocessor:
    """
    A class to process directories containing frames and segmentations and save them into a CSV file.

    Attributes:
        base_dir (str): The base directory containing training data.
        output_csv (str): The path where the output CSV will be saved.
    """

    def __init__(self, base_dir, output_csv):
        """
        Initialize the DataPreprocessor with the base directory and output CSV path.

        Args:
            base_dir (str): Path to the base directory containing training data.
            output_csv (str): Path to save the generated CSV file.
        """
        self.base_dir = base_dir
        self.output_csv = output_csv
        self.frames = []
        self.segmentations = []

    def process_directories(self):
        """
        Process the directories under the base directory to extract frame and segmentation paths.
        """
        # Iterate through all subdirectories in the base directory
        for video_dir in tqdm(os.listdir(self.base_dir), desc="Processing video directories"):
            # Process only directories with names starting with 'video'
            if video_dir.split('_')[0] == 'video':
                video_path = os.path.join(self.base_dir, video_dir)

                # Paths for frames and segmentations within the video directory
                output_frames = os.path.join(video_path, 'frames')
                output_segmentation = os.path.join(video_path, 'segmentation')

                # Collect all frame file paths
                self._process_files(output_frames, self.frames)

                # Collect all segmentation file paths
                self._process_files(output_segmentation, self.segmentations)

    def _process_files(self, directory, storage_list):
        """
        Collect file paths from a directory and store them in a provided list.

        Args:
            directory (str): Path to the directory containing files.
            storage_list (list): List to store the file paths.
        """
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            storage_list.append(file_path)

    def save_to_csv(self):
        """
        Save the collected frame and segmentation paths into a CSV file.
        """
        # Create a DataFrame from the collected data
        data = pd.DataFrame({
            'Frame': self.frames,
            'Segmentation': self.segmentations
        })

        # Save the DataFrame to a CSV file
        data.to_csv(self.output_csv, index=False)
        print(f"Data successfully saved to {self.output_csv}")

    def run(self):
        """
        Run the entire data preprocessing pipeline: process directories and save to CSV.
        """
        self.process_directories()
        self.save_to_csv()


# Example usage
if __name__ == '__main__':
    base_directory = "C:\\Users\\User\Desktop\Shahbaz_project\SAR_RARP50_challenge\Database\Test"
    output_csv_path = "C:\\Users\\User\Desktop\Shahbaz_project\SAR_RARP50_challenge\Database/test_data.csv"
    preprocessor = DataPreprocessor(base_directory, output_csv_path)
    preprocessor.run()
