import numpy as np
from sklearn.decomposition import NMF
import soundfile as sf

import stft


# ============================================================
# 1. Short time Fourier transformation
# ============================================================

window_length = 0.040   # 40 ms
stft_matrix, sr = stft.get_complex_matrix("music.mp3", window_length)
magnitudes = np.abs(stft_matrix)


# ============================================================
# 2. Nonnegative Matrix Factorization
# ============================================================

J = 4

model = NMF(
    n_components=J,
    init="nndsvda",
    solver="mu",                    # NMF-DIV
    beta_loss="kullback-leibler",   # NMF-DIV
    max_iter=1000,
    random_state=0
)

W = model.fit_transform(magnitudes)
H = model.components_

print("X:", magnitudes.shape)
print("W:", W.shape)
print("H:", H.shape)


# ============================================================
# 3. Component spectrograms
# ============================================================

components = []

for j in range(J):
    V_j = np.outer(W[:, j], H[j, :])
    components.append(V_j)

components = np.array(components)

print("components:", components.shape)


# ============================================================
# 4. Ratio masks
# ============================================================

V_sum = np.sum(components, axis=0) + 1e-10

component_stfts = []

for j in range(J):
    mask = components[j] / V_sum

    X_j = mask * stft_matrix

    component_stfts.append(X_j)

component_stfts = np.array(component_stfts)

print("component STFTs:", component_stfts.shape)


# ============================================================
# 5. Sound file
# ============================================================

frame_length = int(sr * window_length)
hop_length = frame_length // 2

for j in range(J):
    audio_j = stft.inverse_stft(component_stfts[j], frame_length, hop_length)
    sf.write(f"component_{j + 1}.wav", audio_j, sr)