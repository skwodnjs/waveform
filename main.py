import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ============================================================
# 1. MP3 파일 읽기
# ============================================================

file_path = "music.mp3"

audio, sr = librosa.load(
    file_path,
    sr=None,
    mono=True
)

print("Sample rate:", sr, "Hz")
print("Number of samples:", len(audio))
print(f"Duration: {len(audio) / sr:.4f}seconds")


# ============================================================
# 2. STFT 설정
# ============================================================

frame_duration = 0.040  # 40 ms

frame_length = int(sr * frame_duration)

# 50% overlap
hop_length = frame_length // 2

window = np.hanning(frame_length)

frequencies = np.fft.rfftfreq(
    frame_length,
    d=1 / sr
)

num_frames = 1 + (len(audio) - frame_length) // hop_length

print("Frame length:", frame_length)
print("Hop length:", hop_length)
print("Number of frames:", num_frames)


# ============================================================
# 3. 모든 frame의 magnitude spectrum 계산
# ============================================================

magnitudes = []

for i in range(num_frames):

    start = i * hop_length
    end = start + frame_length

    frame = audio[start:end]

    windowed_frame = frame * window

    dft = np.fft.rfft(windowed_frame)

    magnitude = np.abs(dft)

    magnitudes.append(magnitude)

magnitudes = np.array(magnitudes)


# ============================================================
# 4. 영상에 표시할 주파수 범위
# ============================================================

max_frequency = 7500  # Hz

mask = frequencies <= max_frequency

display_frequencies = frequencies[mask]
display_magnitudes = magnitudes[:, mask]

# 모든 frame에서 같은 y축을 사용
max_magnitude = np.max(display_magnitudes)


# ============================================================
# 5. 막대그래프 생성
# ============================================================

fig, ax = plt.subplots(figsize=(16, 9))

# DFT frequency bin 사이의 간격
frequency_resolution = sr / frame_length

bars = ax.bar(
    display_frequencies,
    display_magnitudes[0],
    width=frequency_resolution * 0.8
)

ax.set_xlim(0, max_frequency)
ax.set_ylim(0, max_magnitude * 1.05)

ax.set_xlabel("Frequency (Hz)", fontsize=14)
ax.set_ylabel("Magnitude", fontsize=14)
title = ax.set_title("Magnitude Spectrum", fontsize=18)
time_text = ax.text(0.98, 0.95, "", transform=ax.transAxes, horizontalalignment="right", verticalalignment="top", fontsize=16)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()


# ============================================================
# 6. Animation update
# ============================================================

def update(frame_idx):

    magnitude = display_magnitudes[frame_idx]

    for bar, height in zip(bars, magnitude):
        bar.set_height(height)

    current_time = frame_idx * hop_length / sr

    title.set_text(
        "Magnitude Spectrum"
    )

    time_text.set_text(
        f"{current_time:.2f} s"
    )

    return (*bars, title, time_text)


# ============================================================
# 7. Animation 생성
# ============================================================

fps = sr / hop_length

print("Animation FPS:", fps)

animation = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=1000 / fps,
    blit=False
)


# ============================================================
# 8. MP4 저장 + 실시간 미리보기 + 진행도
# ============================================================

writer = FFMpegWriter(
    fps=fps,
    codec="libx264",
    bitrate=8000
)

output_path = "video.mp4"


def format_time(seconds):
    seconds = max(0, int(seconds))

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def progress_callback(current_frame, total_frames):

    # --------------------------------------------------------
    # 진행도
    # --------------------------------------------------------

    completed = current_frame + 1
    percent = completed / total_frames * 100

    print(
        f"\rSaving: {completed}/{total_frames} "
        f"({percent:6.2f}%) ",
        end="",
        flush=True
    )

    # --------------------------------------------------------
    # 미리보기 창 업데이트
    # --------------------------------------------------------

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# interactive mode
plt.ion()

# 창을 먼저 표시
plt.show(block=False)

animation.save(
    output_path,
    writer=writer,
    dpi=120,
    progress_callback=progress_callback
)

plt.ioff()

print()
print("Saved:", output_path)

plt.show()