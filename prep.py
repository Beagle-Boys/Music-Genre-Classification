import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050) # sr * T -> 22050 * 30

#librosa.display.waveplot(signal, sr=sr)

#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:len(frequency)//2]
left_magnitude = magnitude[:len(magnitude)//2]

#plt.plot(left_frequency, left_magnitude)
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Magnitude")
#plt.show()

# stft -> spectrogram

n_fft = 2048 # number of samples per fft
hop_length = 512

stft = librosa.core.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectogram = np.abs(stft)

log_spectogram = librosa.amplitude_to_db(spectogram)

#librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
#plt.xlabel("Time (s)")
#plt.ylabel("Frequency")
#plt.colorbar()
#plt.show()

# MFCCs
MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time (s)")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()