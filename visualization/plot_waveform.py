import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(signal, sr, title = "Waveform"):
    time = np.linspace(0, len(signal) / sr, num=len(signal))

    plt.figure(figsize=(15, 5))
    plt.plot(time, signal, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()