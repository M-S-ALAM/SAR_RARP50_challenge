import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from imageio import imread
def extract_frames(video_path, output_folder, frequency):
    """
    Extracts frames from a video file at a specified frequency, adjusting if the video's fps is lower.
    :param video_path: Path to the video file.
    :param output_folder: Folder where extracted frames will be saved.
    :param frequency: Frequency in Hz to extract frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Detected FPS: {fps}")  # Debug: Print the detected fps

    if not fps or fps < frequency:
        print(f"Warning: Detected FPS ({fps}) is less than the extraction frequency ({frequency}). Extracting every frame instead.")
        frequency = fps  # Adjust frequency to extract every frames

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frame: {frame_count}")
    frame_indices = set(np.arange(0, frame_count, int(fps // frequency)*10))
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

def read_annotations(action_file):
    """
    Reads surgical action annotations from a CSV file.
    :param action_file: Path to the CSV file containing action annotations.
    :return: DataFrame with action annotations.
    """
    return pd.read_csv(action_file, header=None, names=['Start', 'End', 'Action'])

def load_segmentation_mask(mask_folder, frame_number):
    """
    Load a segmentation mask by frame number.
    :param mask_folder: Path to the folder containing segmentation masks.
    :param frame_number: Frame number to load the corresponding mask for.
    :return: Segmentation mask as a numpy array.
    """
    mask_path = os.path.join(mask_folder, f'{frame_number:09d}.png')
    return imread(mask_path)


def main():
    for base_dire in tqdm(os.listdir('/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/Train/')):
        if base_dire.split('_')[0]=='video':
            directory = os.path.join('/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/Train/', base_dire)
            video_path = os.path.join(directory, 'video_left.avi')
            output_folder = os.path.join(directory, 'frames')
            extract_frames(video_path, output_folder, 10)  # Extract frames at 10Hz


if __name__=='__main__':
    main()