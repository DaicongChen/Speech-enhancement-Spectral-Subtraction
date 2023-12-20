import numpy as np
import librosa
import scipy.signal.windows
import scipy.fft
import soundfile as sf
import matplotlib.pyplot as plt

def spectral_subtraction(signal, noise_spectrum, frame_size, overlap):
    hop_size = frame_size - overlap
    n_frames = 1 + int((len(signal) - frame_size) / hop_size)
    processed_signal = np.zeros(len(signal))

    for i in range(n_frames):
        start_idx = i * hop_size
        end_idx = start_idx + frame_size
        frame = signal[start_idx:end_idx]

        window = scipy.signal.windows.hamming(frame_size)
        windowed_frame = frame * window

        frame_spectrum = scipy.fft.fft(windowed_frame)

        subtracted_spectrum = np.abs(frame_spectrum) - noise_spectrum
        subtracted_spectrum = np.maximum(subtracted_spectrum, 0)

        processed_frame = scipy.fft.ifft(subtracted_spectrum * np.exp(1j * np.angle(frame_spectrum)))
        processed_frame = np.real(processed_frame)

        processed_signal[start_idx:end_idx] += processed_frame * window
    return processed_signal
def modified_spectral_subtraction(signal, noise_spectrum, frame_size, overlap,belta,alpha):
    hop_size = frame_size - overlap
    n_frames = 1 + int((len(signal) - frame_size) / hop_size)
    processed_signal = np.zeros(len(signal))

    for i in range(n_frames):
        start_idx = i * hop_size
        end_idx = start_idx + frame_size
        frame = signal[start_idx:end_idx]

        window = scipy.signal.windows.hamming(frame_size)
        windowed_frame = frame * window

        frame_spectrum = scipy.fft.fft(windowed_frame)
        subtracted_spectrum = np.abs(frame_spectrum) - alpha*noise_spectrum
        subtracted_spectrum = np.maximum(subtracted_spectrum, belta*np.abs(frame_spectrum))

        processed_frame = scipy.fft.ifft(subtracted_spectrum * np.exp(1j * np.angle(frame_spectrum)))
        processed_frame = np.real(processed_frame)

        processed_signal[start_idx:end_idx] += processed_frame * window
    return processed_signal

def calculate_snr(signal, noise):
    # 防止除以零
    noise_power = np.sum(noise**2)
    if noise_power == 0:
        return np.inf
    snr = 10 * np.log10(np.sum(signal**2) / noise_power)
    return snr
# 读取输入音频文件
input_wav = r"D:\download\noisy.wav"
signal, sr = librosa.load(input_wav, sr=None)
input_pure=r"D:\download\speech_clean.wav" #Using Clean voice for compare使用纯净语音做测试对比
siganl_pure, sr2= librosa.load(input_pure, sr=None)
# 假设前30帧为噪声
frame_size = 1024
overlap = 102
n_noise_frames = 30
noise_frames = signal[:n_noise_frames * (frame_size - overlap)+overlap]
#
# 确保所有帧具有相同的大小
noise_frames = np.array([noise_frames[i:i + frame_size] for i in range(0, len(noise_frames), frame_size - overlap) if len(noise_frames[i:i + frame_size]) == frame_size])

# 估计噪声频谱
noise_spectrum = np.mean([np.abs(scipy.fft.fft(frame)) for frame in noise_frames], axis=0)

# 应用谱减法
denoised_signal = spectral_subtraction(signal, noise_spectrum, frame_size, overlap)
frame_snrs = []
hop_size = frame_size - overlap
for i in range(0, len(signal) - frame_size, hop_size):
    frame_signal = signal[i:i + frame_size]
    frame_noise = signal[i:i + frame_size] - denoised_signal[i:i + frame_size]
    frame_snr = calculate_snr(frame_signal, frame_noise)
    frame_snrs.append(frame_snr)
average_snr = np.mean(frame_snrs)
alpha=4.0-average_snr*3/20
if(average_snr<0):
    for i in range(0, 4, 1):
        belta = 0.02 + 0.01 * i
        denoised_signal2 = modified_spectral_subtraction(signal, noise_spectrum, frame_size, overlap, belta, alpha)
        output_wav2 = f"D:\\new\\belta{belta:.3f}.wav"
        sf.write(output_wav2, denoised_signal2, sr)
        print(f'belta={belta},alpha={alpha}')
else:
    for i in range(0,4,1):
        belta=0.2-0.065*i
        denoised_signal2 = modified_spectral_subtraction(signal, noise_spectrum, frame_size, overlap, belta, alpha)
        output_wav2 = f"D:\\new\\belta{belta:.3f}.wav"
        sf.write(output_wav2, denoised_signal2, sr)
        print(f'belta={belta},alpha={alpha}')


print(f'The average SNR is: {average_snr:.2f} dB,belta={belta},alpha={alpha}')
#final_sigal=signal-denoised_signal
# 保存输出音频文件


output_wav = r"D:\download\input2.wav"
sf.write(output_wav, denoised_signal, sr)

# 去噪前
plt.figure(figsize=(14, 5))  # 调整以匹配示例图的尺寸

# 去噪前的波形
plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('Original Waveform')

# 去噪后的波形
plt.subplot(2, 1, 2)
plt.plot(denoised_signal)
plt.title('Denoised Waveform')
plt.show()

plt.subplot(2, 1, 1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Spectrogram')

# 去噪后
plt.subplot(2, 1, 2)
D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_signal)), ref=np.max)
librosa.display.specshow(D_denoised, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Denoised Spectrogram')
plt.show()

plt.subplot(3, 1, 1)
D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_signal)), ref=np.max)
librosa.display.specshow(D_denoised, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Denoised Spectrogram')


plt.subplot(3, 1, 2)
D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_signal2)), ref=np.max)
librosa.display.specshow(D_denoised, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Modified Denoised Spectrogram')


plt.subplot(3, 1, 3)
D_Pure = librosa.amplitude_to_db(np.abs(librosa.stft(siganl_pure)), ref=np.max)
librosa.display.specshow(D_Pure, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Pure Spectrogram')
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(frame_snrs)
plt.title('Frame-wise SNR')
plt.xlabel('Frame')
plt.ylabel('SNR (dB)')
plt.show()
print('DONE')