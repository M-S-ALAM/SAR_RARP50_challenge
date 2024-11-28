import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from imageio import imread


class VideoProcessor:
    """
    A class to handle video processing tasks such as frame extraction and annotation loading.
    """

    def __init__(self, base_dir):
        """
        Initialize the VideoProcessor with the base directory.

        Args:
            base_dir (str): Path to the base directory containing training videos and annotations.
        """
        self.base_dir = base_dir

    def extract_frames(self, video_path, output_folder, frequency):
        """
        Extract frames from a video file at a specified frequency.

        Args:
            video_path (str): Path to the video file.
            output_folder (str): Folder where extracted frames will be saved.
            frequency (int): Frequency in Hz to extract frames.

        Notes:
            If the video's FPS is lower than the specified frequency, every frame is extracted.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        print(f"Detected FPS: {fps}")  # Debug: Print the detected FPS

        if not fps or fps < frequency:
            print(f"Warning: Detected FPS ({fps}) is less than the extraction frequency ({frequency}). Extracting every frame instead.")
            frequency = fps  # Adjust frequency to extract every frame

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total number of frames: {frame_count}")
        frame_indices = set(np.arange(0, frame_count, max(1, int(fps // frequency))))

        frame_id = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_id in frame_indices:
                frame_path = os.path.join(output_folder, f'{frame_id:09d}.png')
                cv2.imwrite(frame_path, frame)
            frame_id += 1

        video.release()
        print(f"Frames extracted and saved to {output_folder}")

    @staticmethod
    def read_annotations(action_file):
        """
        Read surgical action annotations from a CSV file.

        Args:
            action_file (str): Path to the CSV file containing action annotations.

        Returns:
            pd.DataFrame: DataFrame with action annotations.
        """
        return pd.read_csv(action_file, header=None, names=['Start', 'End', 'Action'])

    @staticmethod
    def load_segmentation_mask(mask_folder, frame_number):
        """
        Load a segmentation mask by frame number.

        Args:
            mask_folder (str): Path to the folder containing segmentation masks.
            frame_number (int): Frame number to load the corresponding mask for.

        Returns:
            np.ndarray: Segmentation mask as a numpy array.
        """
        mask_path = os.path.join(mask_folder, f'{frame_number:09d}.png')
        return imread(mask_path)

    def process_videos(self, frequency=10):
        """
        Process all videos in the base directory, extracting frames at the specified frequency.

        Args:
            frequency (int): Frequency in Hz to extract frames.
        """
        for video_dir in tqdm(os.listdir(self.base_dir), desc="Processing video directories"):
            if video_dir.split('_')[0] == 'video':
                video_path = os.path.join(self.base_dir, video_dir, 'video_left.avi')
                output_folder = os.path.join(self.base_dir, video_dir, 'frames')
                self.extract_frames(video_path, output_folder, frequency)


if __name__ == '__main__':
    # Define the base directory
    base_directory = '/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/Train/'

    # Create an instance of VideoProcessor
    video_processor = VideoProcessor(base_directory)

    # Process all videos and extract frames at 10Hz
    video_processor.process_videos(frequency=10)
