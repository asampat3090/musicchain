# test_data.py
import numpy as np
import torch
import pretty_midi
from songchain.data import MIDIClip, AudioClip


class TestMIDIClip:
    def setup_method(self):
        self.midi_obj = MIDIClip("roja-ar-rahman-melody.mid")

    def test_setup(self):
        assert self.midi_obj.filepath == "roja-ar-rahman-melody.mid"
        assert isinstance(self.midi_obj.midi, pretty_midi.PrettyMIDI)
        assert self.midi_obj.duration == 44.21053

    def test_pianoroll_binary(self):
        # test binary pianoroll (default parameters)
        result = self.midi_obj.pianoroll()
        assert result.shape == (128, 1680)  # default 24 ticks per beat
        assert result.max() == 1
        assert result.min() == 0
        assert torch.is_tensor(result)  # torch Tensor
        assert result.dtype == torch.int64  # default is int64

    def test_pianoroll_velocity(self):
        # test velocity pianoroll (default parameters)
        result = self.midi_obj.pianoroll(type="velocity")
        assert result.shape == (128, 1680)  # default 24 ticks per beat
        assert result.max() == 80
        assert result.min() == 0
        assert torch.is_tensor(result)  # torch Tensor
        assert result.dtype == torch.int64  # default is int64

    def test_notes_index(self):
        # test index notes (default parameters)
        result = self.midi_obj.notes(type="index")
        assert result.shape == (1, 1680)  # default 24 ticks per beat
        assert result[0, 0] == 64
        assert isinstance(result, np.ndarray)  # default is np.ndarray

    def test_notes_note(self):
        result = self.midi_obj.notes(type="note")
        assert result.shape == (1, 1680)
        assert result[0, 0] == "E4"
        assert isinstance(result, np.ndarray)  # default is np.ndarray


# def setup_method(self):
#   self.text = "hello world"

# def test_upper(self):
#   assert self.text.upper() == "HELLO WORLD"

# def test_lower(self):
#   assert self.text.lower() == "hello world"


class TestAudioClip:
    def setup_method(self):
        self.audio_obj = AudioClip("002_001.wav")

    def test_setup(self):
        assert self.audio_obj.filepath == "002_001.wav"
        assert torch.is_tensor(self.audio_obj.tensor)
        assert isinstance(self.audio_obj.sample_rate, int)
        assert self.audio_obj.duration == 30.0

    def test_waveform(self):
        assert torch.is_tensor(self.audio_obj.waveform)
        assert self.audio_obj.sample_rate == 44100
        assert self.audio_obj.mono == False  # stereo
        assert self.audio_obj.waveform.shape == (2, 1323000)  # 30 * 44100

    def test_spectrogram(self):
        assert torch.is_tensor(self.audio_obj.spectrogram)
        assert self.audio_obj.spectrogram.shape == (2, 1025, 2584)

    def test_encodec(self):
        assert self.audio_obj.encodec.shape == (1, 4, 1500)  # 30 * 44100 / 882?
