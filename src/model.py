import pandas as pd
import numpy as np
import os
import librosa

from pathlib import Path
from typing import List

from helper_funcs import visualize_mfcc, extract_features, extract_centroid_from_audio, \
    extract_mfcc_from_audio, extract_rmse_from_audio,\
    visualize_audio, visualize_centroid, visualize_power_and_spectral_contrast,\
    visualize_mfcc_and_mfcc_deltas, extract_melspectrogram_from_audio

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)

base_path = Path('../data')
room_path = base_path / 'room'
shops_path = base_path / 'shops'
test_audio_path = shops_path / 'lewiatan_001.wav'
#test_audio_path = room_path / '1, room 002.wav'


hop_length = 128

features = []

path_list = room_path.glob('**/*.wav')
#path_list = shops_path.glob('**/*.wav')
for path in path_list:
    audio, sr = librosa.load(path, duration=4)
    data = extract_features(path)
    class_name = path.parts[-1][0]
    print(class_name)


# Convert into a Panda dataframe
df_features = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(df_features), ' files')
print(df_features)

df_features["class_label"] = df_features["class_label"].apply(lambda x: 0 if x == '0' else 1)  # change to binary
y = np.array(df_features["class_label"].tolist())