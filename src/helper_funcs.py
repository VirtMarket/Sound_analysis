import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as display

from typing import List, Tuple
from pathlib import Path


def extract_features(file_name: Path) -> np.ndarray:
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=4)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)#, f_min=0.0, f_max=sr/2)
        mfccs = mfccs[:12, :] # only keep the first 12 coefficients corresponding to lower frequencies
        mfccs -= (np.mean(mfccs, axis=0) + 1e-8)
        mfccsscaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return mfccs


def extract_rmse_from_audio(audio: np.ndarray):
    rms = librosa.feature.rmse(y=audio)
    return rms


def extract_centroid_from_audio(audio: np.ndarray, sr: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=512, hop_length=hop_length)
    centroid_frames = np.indices((centroid.shape[-1],))
    centroid_times = librosa.frames_to_time(centroid_frames, sr=sr, hop_length=hop_length)
    return centroid, centroid_times


def extract_melspectrogram_from_audio(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mspec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mspec -= (np.mean(mspec, axis=0) + 1e-8)
    mspec_delta = librosa.feature.delta(mspec)
    mspec_delta2 = librosa.feature.delta(mspec, order=2)
    return mspec, mspec_delta, mspec_delta2


def extract_mfcc_from_audio(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # , f_min=0.0, f_max=sr/2)
    mfccs = mfccs[:12, :]  # only keep the first 12 coefficients corresponding to lower frequencies
    mfccs -= (np.mean(mfccs, axis=0) + 1e-8)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    return mfccs, mfccs_delta, mfccs_delta2


def visualize_audio(audio: np.ndarray, sr: int) -> None:
    plt.figure()
    display.waveplot(audio, sr=sr)
    plt.show()


def visualize_mfcc(mfccs: List[float], label: str) -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(label)
    plt.tight_layout()
    plt.show()


def visualize_mfcc_and_mfcc_deltas(mfcc: List[float], mfcc_delta: List[float], mfcc_delta2: List[float], feat_type: str) -> None:
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfcc)
    plt.title(feat_type)
    plt.colorbar()
    plt.subplot(3, 1, 2)
    librosa.display.specshow(mfcc_delta)
    #plt.title(r'MFCC-$\Delta$')
    plt.title(feat_type + r'-$\Delta$')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(mfcc_delta2, x_axis='time')
    plt.title(feat_type + r'-$\Delta^2$')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def visualize_centroid(centroid: np.ndarray, centroid_times: np.ndarray) -> None:
    plt.figure()
    # plt.semilogy(centroid.T, label='Spectral centroid')
    plt.semilogy(centroid_times[0], centroid[0], label='Spectral centroid')
    plt.ylabel('Hz')
    plt.xlim([0, centroid_times[0][-1]])
    plt.xlabel('time, s')
    plt.legend()
    plt.show()


def visualize_power_and_spectral_contrast(S: np.ndarray, contrast: np.ndarray) -> None:
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(contrast, x_axis='time')
    plt.colorbar()
    plt.ylabel('Frequency bands')
    plt.title('Spectral contrast')
    plt.tight_layout()
    plt.show()