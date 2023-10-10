# test_data.py
import numpy as np
import torch
import pretty_midi
from audiochain.data import MIDIClip, AudioClip


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
