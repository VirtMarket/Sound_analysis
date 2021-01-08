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

desired_width=320
pd.set_option('display.width', desired_width)
#np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',10)

base_path = Path('../data/UrbanSound8K/audio')
audio_fold1_path = base_path / 'fold1'
metadata_path = Path('../data/urbanSound8K/metadata/UrbanSound8K.csv')
# audio_fold1_path = '../data/UrbanSound8K/audio/fold1'
# metadata_path = '../data/urbanSound8K/metadata/UrbanSound8K.csv'
test_audio_path = audio_fold1_path / '7383-3-0-0.wav'

audio, sr = librosa.load(test_audio_path)

print(len(audio)/sr)

plt.figure()
display.waveplot(audio, sr=sr)
plt.show()

# Metadata
metadata = pd.read_csv(metadata_path)
print(metadata.head())
print(metadata.describe())


####################### Extract features ###########################


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs


features = []

for index, row in metadata.iterrows():
    if index == 10:
        break
    file_name = base_path / ('fold' + str(row["fold"])) / str(row["slice_file_name"])
    class_label = row["class"]
    data = extract_features(file_name)

    features.append([data, class_label])
    print(index)


# Convert into a Panda dataframe
df_features = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(df_features), ' files')
print(df_features)


#X = np.array(df_features["feature"].tolist())
y = np.array(df_features["class_label"].tolist())

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False)
y_int = label_encoder.fit_transform(y)
y_categorical = one_hot_encoder.fit_transform(y_int.reshape((-1, 1)))

# print(y)
# print(y_int)
# print(y_categorical)
#
# inverted = label_encoder.inverse_transform([argmax(y_categorical[0, :])])
# print(inverted)

#x_train, x_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)


def visualize_mfcc(mfccs: List[float], label: str) -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(label)
    plt.tight_layout()
    plt.show()


for _, example in df_features.iterrows():
    visualize_mfcc(example["feature"], example["class_label"])
