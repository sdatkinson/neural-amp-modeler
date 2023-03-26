"""
Input and output WAV file alignment script.
"""

from argparse import ArgumentParser, Namespace
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
import wavio


def read_wav(filename: str) -> Tuple[np.ndarray, int, int]:
    """
    Reads a WAV file using the wavio library.

    Args:
        filename (str): Path to the WAV file.

    Returns:
        numpy.ndarray: A 2D NumPy array containing the audio data and the sampling rate.
    """
    # Load the WAV file using wavio
    wav = wavio.read(filename)

    # Extract the audio data and sampling rate from the wav object
    audio_data = wav.data
    sample_rate = wav.rate
    bitdepth = wav.sampwidth * 8

    # Calculate the maximum amplitude of the audio data
    max_amplitude = 2**(bitdepth - 1)

    # Normalize the audio data to the range [-1, 1]
    audio_data = audio_data.astype(np.float32) / max_amplitude

    # Return the audio data and sampling rate as a 2D NumPy array
    return audio_data, sample_rate, bitdepth


def write_wav(filename: str, audio_data: np.ndarray, sample_rate: int, bitdepth: int = 16):
    """
    Writes audio data to a WAV file using the wavio library.

    Args:
        filename (str): Path to the output WAV file.
        audio_data (numpy.ndarray): A 1D or 2D NumPy array containing the audio data.
        sample_rate (int): The sampling rate of the audio data in Hz.
        bitdepth (int): The bit depth of the audio data. Default is 16.

    Raises:
        ValueError: If the bit depth is not 8, 16, 24, or 32.

    """

    # Ensure that the bit depth is valid
    if bitdepth not in [8, 16, 24, 32]:
        raise ValueError("Invalid bit depth. Must be 8, 16, 24, or 32.")

    # If the audio data is a 1D array, convert it to a 2D array with a single channel
    if audio_data.ndim == 1:
        audio_data = audio_data[:, np.newaxis]

    # Write the audio data to the WAV file using wavio
    wavio.write(filename, audio_data, rate=sample_rate, sampwidth=bitdepth // 8)


def align_wav(input_wav: np.ndarray, output_wav: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Aligns the output waveform to the input waveform

    Args:
        input_wav (numpy.ndarray): A 1D NumPy array containing the input waveform.
        output_wav (numpy.ndarray): A 1D NumPy array containing the output waveform

    Returns:
        numpy.ndarray: A 1D NumPy array containing the aligned output waveform.
        int: The number of samples that the output waveform was shifted.
    """

    # Calculate the cross-correlation
    corr = correlate(output_wav, input_wav, mode='full')

    # Find the index of the maximum correlation value
    shift = np.argmax(corr) - len(input_wav) + 1

    # Shift the output waveform to align with the input waveform
    aligned_output_wav = np.roll(output_wav, -shift)

    return aligned_output_wav, shift


# Main function
def main(args: Namespace):
    """
    Main function for aligning the output WAV file to the input WAV file.
    """

    # Get the input and output WAV file paths
    input_wav_path = args.input_wav_path
    output_wav_path = args.output_wav_path

    # Read the input and output WAV files
    input_wav_data, _, _ = read_wav(input_wav_path)
    output_wav_data, output_sample_rate, output_bitdepth = read_wav(output_wav_path)

    # Print the shape of the input and output WAV files
    print(input_wav_data.shape)
    print(output_wav_data.shape)

    # Print the maximum and minimum values of the input and output WAV files
    print(input_wav_data.max(), input_wav_data.min())
    print(output_wav_data.max(), output_wav_data.min())

    # Shift the output to test the alignment
    shift = 3000
    output_wav_data = np.roll(output_wav_data, shift)
    aligned_output_wav_data, align_shift = align_wav(input_wav_data, output_wav_data)
    print('Shift: ', align_shift)

    # PLot before and after alignment
    plt.figure(figsize=(12, 4))
    plt.plot(input_wav_data)
    plt.plot(output_wav_data)
    plt.title('Output')

    plt.figure(figsize=(12, 4))
    plt.plot(input_wav_data)
    plt.plot(aligned_output_wav_data)
    plt.title('Aligned Output')

    plt.figure(figsize=(12, 4))
    plt.plot(input_wav_data[:50000])
    plt.plot(aligned_output_wav_data[:50000])
    plt.title('Aligned Output Zoomed In')

    plt.show()

    # Write the aligned output wav file
    aligned_output_wav_path = output_wav_path.replace('.wav', '_aligned.wav')
    print('Writing aligned output wav file to: ', aligned_output_wav_path)
    write_wav(aligned_output_wav_path, aligned_output_wav_data, output_sample_rate, output_bitdepth)


if __name__ == '__main__':
    # Read the input and output path from the command line
    parser = ArgumentParser()
    parser.add_argument('--input_wav_path', type=str, required=True)
    parser.add_argument('--output_wav_path', type=str, required=True)
    main(parser.parse_args())
