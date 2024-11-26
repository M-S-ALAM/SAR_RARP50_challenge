from operator import index

import pandas as pd
from tqdm import tqdm
import os
print()

def main():
    frames = []
    segemntation = []
    data = pd.DataFrame(columns=['Frame', 'Segmentation'])
    for base_dire in tqdm(os.listdir('/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/Train/')):
        if base_dire.split('_')[0] == 'video':
            directory = os.path.join('/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/Train/', base_dire)
            output_frames = os.path.join(directory, 'frames')
            output_segmentation = os.path.join(directory, 'segmentation')
            for dir in os.listdir(output_frames):
                frame_dir = os.path.join(output_frames ,dir)
                frames.append(frame_dir)
            for dir in os.listdir(output_segmentation):
                segmenation_dir = os.path.join(output_segmentation, dir)
                segemntation.append(segmenation_dir)
    data['Frame'] = frames
    data['Segmentation'] = segemntation
    data.to_csv('/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/train_data.csv', index=False)

if __name__=='__main__':
    main()