import pandas as pd
import numpy as np
import os
import librosa
import librosa.display as display
import matplotlib.pyplot as plt

from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

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
#
# # audio, sr = librosa.load(test_audio_path, duration=4)
# # print(type(audio))
#
# # Loading longer file
# sr = librosa.get_samplerate(test_audio_path)
#
#
# # Set the frame parameters to be equivalent to the librosa defaults
# # in the file's native sampling rate
# frame_length_load = sr // 100
# hop_length_load = frame_length_load
# four_sec_block = int(4 * sr / frame_length_load)
# print(sr)
# print(frame_length_load)
# print(four_sec_block)
#
#
#
# # Stream the data, working on 128 frames at a time
# stream = librosa.stream(test_audio_path,
#                         offset=0.5,
#                         block_length=four_sec_block,
#                         frame_length=frame_length_load,
#                         hop_length=hop_length_load,
#                         fill_value=0.0)
# sr_original = sr
# sr_target = 22050
# for i, block in enumerate(stream):
#     print(i)
#     # Resample to 22050
#     block = librosa.resample(block, sr_original, sr_target)
#     print(len(block))
#     sr = sr_target
#
#     # RMSE
#     #rmse = extract_rmse_from_audio(block)
#
#     # Centroid
#     centroid, centroid_times = extract_centroid_from_audio(block, sr, hop_length)
#     print('centroid')
#     print(centroid.shape)
#
#     # MFCC
#     mfcc, mfcc_delta, mfcc_delta2 = extract_mfcc_from_audio(block, sr)
#     print('MFCC')
#     print(mfcc.shape)
#
#     # Spectral contrast
#     S = np.abs(librosa.stft(block, n_fft=512))
#     contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=512, center=False, n_bands=4, quantile=0.01)
#     print('absolute spectrum')
#     print(S.shape)
#     print('contrast')
#     print(contrast.shape)
#
#     # Visuals
#     visualize_audio(block, sr)
#     visualize_centroid(centroid, centroid_times)
#     #visualize_mfcc(mfcc, 'MFCCS')
#     visualize_mfcc_and_mfcc_deltas(mfcc, mfcc_delta, mfcc_delta2)
#     visualize_power_and_spectral_contrast(S, contrast)
#     continue
#
#
# # Centroid
# centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=512, hop_length=hop_length)
# centroid_frames = np.indices((centroid.shape[-1],))
# centroid_times = librosa.frames_to_time(centroid_frames, sr=sr, hop_length=hop_length)
# print(centroid.shape)
# print(centroid_times.shape)
#
# # MFCC
# sample_mfcc = extract_features(test_audio_path)
#
# # Visuals
# visualize_audio(audio, sr)
# visualize_centroid(centroid, centroid_times)
# visualize_mfcc(sample_mfcc, 'MFCCS')
#
# # plt.figure(figsize=(10, 4))
# # librosa.display.specshow(centroid, x_axis='time', sr=sr, hop_length=hop_length)
# # plt.colorbar()
# # plt.title("centroid")
# # plt.tight_layout()
# # plt.show()
#
#

####################### Extract features ###########################

features = []

path_list = room_path.glob('**/*.wav')
#path_list = shops_path.glob('**/*.wav')
for path in path_list:
    audio, sr = librosa.load(path, duration=4)
    data = extract_features(path)
    class_name = path.parts[-1][0]
    print(class_name)
    features.append([data, class_name])

    # Centroid
    centroid, centroid_times = extract_centroid_from_audio(audio, sr, hop_length)
    print('centroid')
    print(centroid.shape)

    # MFCC
    mfcc, mfcc_delta, mfcc_delta2 = extract_mfcc_from_audio(audio, sr)
    print('MFCC')
    print(mfcc.shape)

    # Mel spectrogram
    mspec, mspec_delta, mscpec_delta2 = extract_melspectrogram_from_audio(audio, sr)
    print('mspec')
    print(mspec.shape)

    # Spectral contrast
    S = np.abs(librosa.stft(audio, n_fft=2048))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=2048, center=True, n_bands=4, quantile=0.01)
    print('absolute spectrum')
    print(S.shape)
    print('contrast')
    print(contrast.shape)

    # Visuals
    visualize_audio(audio, sr)
    visualize_centroid(centroid, centroid_times)
    # visualize_mfcc(mfcc, 'MFCCS')
    visualize_mfcc_and_mfcc_deltas(mfcc, mfcc_delta, mfcc_delta2, feat_type='MFCC')
    visualize_mfcc_and_mfcc_deltas(mspec, mspec_delta, mscpec_delta2, feat_type='mspec')
    visualize_power_and_spectral_contrast(S, contrast)
    continue

# Convert into a Panda dataframe
df_features = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(df_features), ' files')
print(df_features)


#X = np.array(df_features["feature"].tolist())
df_features["class_label"] = df_features["class_label"].apply(lambda x: 0 if x == '0' else 1)  # change to binary
y = np.array(df_features["class_label"].tolist())

# label_encoder = LabelEncoder()
#one_hot_encoder = OneHotEncoder(sparse=False)
# y_int = label_encoder.fit_transform(y)
#y_categorical = one_hot_encoder.fit_transform(y.reshape((-1, 1)))

# # print(y)
# # print(y_int)
# # print(y_categorical)
# #
# # inverted = label_encoder.inverse_transform([argmax(y_categorical[0, :])])
# # print(inverted)
#
# #x_train, x_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)
#
#



# for i, example in df_features.iterrows():
#     if i == 15:
#         visualize_mfcc(example["feature"], example["class_label"])
