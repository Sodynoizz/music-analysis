"""
MIT License

Copyright (c) 2024 Chorasit Apilardmongkol

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Optional
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt


class AudioAnalyzer:
    def __init__(self, file_path: str):
        """
        Initialize the AudioAnalyzer with the path to an audio file.

        Parameters:
        file_path (str): Path to the audio file.
        """
        self.file_path: str = file_path
        self.sample_rate: Optional[int] = None
        self.audio_data: Optional[np.ndarray] = None
        self.frequencies: Optional[np.ndarray] = None
        self.times: Optional[np.ndarray] = None
        self.Zxx: Optional[np.ndarray] = None
        self._spectral_flux: Optional[np.ndarray] = None
        self._moving_avg: Optional[np.ndarray] = None
        self._times_moving_avg: Optional[np.ndarray] = None

    @classmethod
    def convert_mp3_to_wav(
        cls, file_path: str, output_path: str = "output/output.wav"
    ) -> "AudioAnalyzer":
        """
        Convert an MP3 file to WAV format.

        Parameters:
        file_path (str): Path to the MP3 file.
        output_path (str): Path to save the converted WAV file.

        Returns:
        AudioAnalyzer: An instance of AudioAnalyzer with the WAV file path.
        """
        audio = AudioSegment.from_mp3(file_path)
        audio.export(output_path, format="wav")
        return cls(output_path)

    @classmethod
    def load_wav_file(cls, file_path: str) -> "AudioAnalyzer":
        """
        Load a WAV file and initialize an AudioAnalyzer instance.

        Parameters:
        file_path (str): Path to the WAV file.

        Returns:
        AudioAnalyzer: An instance of AudioAnalyzer with the loaded WAV data.
        """
        sample_rate, audio_data = wavfile.read(file_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        instance = cls(file_path)
        instance.sample_rate = sample_rate
        instance.audio_data = audio_data
        return instance

    def compute_stft(self, nperseg: int = 1024) -> None:
        """
        Compute the Short-Time Fourier Transform (STFT) of the audio data.

        Parameters:
        nperseg (int): Length of each segment for STFT.

        Returns:
        None
        """
        if self.audio_data is None or self.sample_rate is None:
            raise ValueError("Audio data or sample rate not loaded.")
        self.frequencies, self.times, self.Zxx = stft(
            self.audio_data, self.sample_rate, nperseg=nperseg
        )

    def calculate_spectral_flux(self) -> None:
        """
        Calculate the spectral flux from the STFT of the audio data.

        Returns:
        None
        """
        if self.Zxx is None:
            raise ValueError("STFT not computed.")
        magnitude = np.abs(self.Zxx)
        diff_magnitude = np.diff(magnitude, axis=1)
        sum_diff_magnitude = np.sum(diff_magnitude**2, axis=0)
        self._spectral_flux = np.sqrt(sum_diff_magnitude)
        self._spectral_flux = self._spectral_flux / np.amax(self._spectral_flux)

    @staticmethod
    def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute the moving average of a data array.

        Parameters:
        data (np.ndarray): The data to compute the moving average on.
        window_size (int): The window size for the moving average.

        Returns:
        np.ndarray: The moving average of the data.
        """
        return np.convolve(data, np.ones(window_size), "valid") / window_size

    def compute_moving_average(self, window_size: int = 25) -> None:
        """
        Compute the moving average of the spectral flux.

        Parameters:
        window_size (int): The window size for the moving average.

        Returns:
        None
        """
        if self.spectral_flux is None:
            self.calculate_spectral_flux()
        self._moving_avg = self.moving_average(self.spectral_flux, window_size)
        self._times_moving_avg = self.times[
            1 + window_size // 2 : -window_size // 2 + 1
        ]

    @property
    def spectral_flux(self) -> np.ndarray:
        """
        Get the spectral flux of the audio data.

        Returns:
        np.ndarray: The spectral flux of the audio data.
        """
        if self._spectral_flux is None:
            self.calculate_spectral_flux()
        return self._spectral_flux

    @property
    def moving_avg(self) -> np.ndarray:
        """
        Get the moving average of the spectral flux.

        Returns:
        np.ndarray: The moving average of the spectral flux.
        """
        if self._moving_avg is None:
            self.compute_moving_average()
        return self._moving_avg

    @property
    def times_moving_avg(self) -> np.ndarray:
        """
        Get the times corresponding to the moving average of the spectral flux.

        Returns:
        np.ndarray: The times for the moving average of the spectral flux.
        """
        if self._times_moving_avg is None:
            self.compute_moving_average()
        return self._times_moving_avg

    def plot_spectral_flux(self) -> None:
        """
        Plot the spectral flux over time, including the moving average.

        Returns:
        None
        """
        plt.figure(figsize=(12, 8))
        plt.plot(self.times[1:], self.spectral_flux, label="Spectral Flux")
        if self._moving_avg is not None:
            plt.plot(
                self.times_moving_avg,
                self.moving_avg,
                label="Moving Average",
                color="g",
            )
        plt.xlabel("Time (s)")
        plt.ylabel("Spectral Flux")
        plt.title("Spectral Flux over Time")
        plt.legend()
        plt.show()


# Usage example:
# Converting MP3 to WAV and loading
analyzer = AudioAnalyzer.convert_mp3_to_wav("music/music.mp3")
analyzer = AudioAnalyzer.load_wav_file("output/output.wav")

# Continuing with the analysis
analyzer.compute_stft()

# Accessing properties which will compute values if not already computed
print(analyzer.spectral_flux)
print(analyzer.moving_avg)

# Plotting
analyzer.plot_spectral_flux()
