import math
import torchaudio
import torch

# from audiocraft.utils.notebook import display_audio

import os
from pydub import AudioSegment
from tqdm import tqdm

import matplotlib.pyplot as plt


# Audio Manipulation
def get_bip_bip(
    bip_duration=0.125, frequency=440, duration=0.5, sample_rate=32000, device="cuda"
):
    """Generates a series of bip bip at the given frequency."""
    t = (
        torch.arange(int(duration * sample_rate), device="cuda", dtype=torch.float)
        / sample_rate
    )
    wav = torch.cos(2 * math.pi * 440 * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope


def split_audio_and_save(
    chunk_size: int, songs_path: str, description: str, output_dir: str = "output"
) -> list[AudioSegment]:
    """split audio into appropriate chunk sizes and save from folder of audio files.

    Note: the final chunk for each song will likely be less than the desired chunk size if not exactly divisible by song length.

    chunk_size: the size of the chunks in seconds
    songs_path: directory of songs
    description: description to save
    output_dir: directory name for where to save. will be saved within the songs_path

    Usage:
        # run splitting function
        split_audio_and_save(30, audio_path, "chill bollywood beats with vocals, a slow BPM, reverb and a vinyl crackle")

        # Outputs will look something like this 000_000.wav, 000_000.txt, etc..
    """
    input_songs_list = os.listdir(songs_path)
    output_path = os.path.join(songs_path, output_dir)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, input_song in enumerate(tqdm(input_songs_list)):
        if ".mp3" not in input_song:
            continue
        input_song_path = os.path.join(songs_path, input_song)
        try:
            input_song_obj = AudioSegment.from_mp3(input_song_path)
        except:
            print('skipping "%s" since it cannot be decoded by ffmpeg' % input_song)
            continue
        input_song_length = len(input_song_obj)
        for j, start_ind in enumerate(range(0, input_song_length, chunk_size * 1000)):
            end_ind = start_ind + chunk_size * 1000
            if end_ind > input_song_length:
                end_ind = input_song_length
            chunk_name = format(i, "03d") + "_" + format(j, "03d")
            # save the audio chunk as .wav
            input_song_obj[start_ind:end_ind].export(
                os.path.join(output_path, chunk_name + ".wav"), format="wav"
            )
            # save the .txt file with description
            with open(os.path.join(output_path, chunk_name + ".txt"), "w") as f:
                f.write(description)


# visualize audio files
# waveform, sample_rate = torchaudio.load(SAMPLE_WAV)


def plot_waveform(waveform, sample_rate):
    """Plots the waveform.

    Args:
      waveform (torch.Tensor): Tensor of shape (channels, num_frames)
      sample_rate (int): Sample rate of the waveform

    Returns:
      None

    Usage:
    > plot_waveform(waveform, sample_rate)
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)


def plot_specgram_from_waveform(waveform, sample_rate, title="Spectrogram") -> None:
    """Plots the spectrogram from the original waveform.

    Args:
      waveform (torch.Tensor): Tensor of shape (channels, num_frames)
      sample_rate (int): Sample rate of the waveform
      title (str, optional): Title of the plot. Defaults to "Spectrogram".

    Returns:
      None

    Usage:
    > plot_specgram(waveform, sample_rate)
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(
    spec: T.tensor, title=None, ylabel="freq_bin", aspect="auto", xmax=None
) -> None:
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


# Audio Handling via Torch: https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html
# Audio Handling via PyDub: https://github.com/jiaaro/pydub/blob/master/API.markdown
# Audio Handling via Librosa: https://librosa.org/doc/latest/index.html


def audio_to_midi(self, audio_filepath: str) -> torch.Tensor:
    """Converts an audio clip to a MIDI clip.

    Args:
    audio (torch.Tensor): Audio clip in tensor fp32 format (1 x T).
    fs (int, optional): Sampling frequency in Hertz. Defaults to 100.

    Returns:
    torch.Tensor: MIDI clip in tensor fp32 format (128 x T)
    """
    # Use Basic Pitch library to enable this -> can convert AudioClip -> MidiClip
    # https://github.com/spotify/basic-pitch
    pass
