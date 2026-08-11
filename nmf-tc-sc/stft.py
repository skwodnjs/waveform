import librosa
import numpy as np


"""
Hanning window, short-time Fourier transform, 50% overlap
"""


def get_complex_matrix(file_path, window_length):
    """
    Input:
        file_path: Path to the audio file.
        window_length: Window length in seconds.

    Output:
        X: Observation matrix with shape (frequency, time).
    """


    print("[stft.py] get_complex_matrix")


    # ============================================================
    # 1. MP3 파일 읽기
    # ============================================================

    audio, sr = librosa.load(file_path, sr=None, mono=True)

    print("Sample rate:", sr, "Hz")
    print("Number of samples:", len(audio))
    print(f"Duration: {len(audio) / sr:.2f} seconds")


    # ============================================================
    # 2. Setting
    # ============================================================

    frame_length = int(sr * window_length)

    # 50% overlap
    hop_length = frame_length // 2

    window = np.hanning(frame_length)

    num_frames = 1 + (len(audio) - frame_length) // hop_length

    print("Frame length:", frame_length)
    print("Hop length:", hop_length)
    print("Number of frames:", num_frames)


    # ============================================================
    # 3. 각 frame의 magnitude spectrum 계산
    # ============================================================

    stft_matrix = []

    for i in range(num_frames):

        start = i * hop_length
        end = start + frame_length

        frame = audio[start:end]

        windowed_frame = frame * window

        dft = np.fft.rfft(windowed_frame)

        stft_matrix.append(dft)

    stft_matrix = np.array(stft_matrix).T

    return stft_matrix, sr


def inverse_stft(X, frame_length, hop_length):
    """
    Input:
        X: Complex STFT matrix with shape = (frequency, time).

    Output:
        audio: Reconstructed waveform.
    """

    print("[stft.py] inverse_stft")

    num_frames = X.shape[1]

    output_length = frame_length + (num_frames - 1) * hop_length

    audio = np.zeros(output_length)
    normalization = np.zeros(output_length)

    window = np.hanning(frame_length)

    for i in range(num_frames):

        frame = np.fft.irfft(X[:, i], n=frame_length)

        start = i * hop_length
        end = start + frame_length

        audio[start:end] += frame * window
        normalization[start:end] += window ** 2

    mask = normalization > 1e-10
    audio[mask] /= normalization[mask]

    return audio