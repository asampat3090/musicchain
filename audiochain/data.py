# TODO: cleanup import statements
import numpy as np
import torch
import torchaudio
import pydub
import pretty_midi
import pypianoroll


class AudioClip:
    # def __init__(self, sample_rate=32000, n_fft, hop_length, n_mels, f_min, f_max, duration, mono):
    #     self.sample_rate = sample_rate
    #     self.n_fft = n_fft
    #     self.hop_length = hop_length
    #     self.n_mels = n_mels
    #     self.f_min = f_min
    #     self.f_max = f_max
    #     self.duration = duration
    #     self.mono = mono

    # def __call__(self, data):
    #     data = librosa.resample(data, self.sample_rate, self.sample_rate)
    #     data = librosa.to_mono(data)
    #     data = librosa.util.fix_length(data, self.duration * self.sample_rate)
    #     data = librosa.feature.melspectrogram(
    #         data,
    #         sr=self.sample_rate,
    #         n_fft=self.n_fft,
    #         hop_length=self.hop_length,
    #         n_mels=self.n_mels,
    #         fmin=self.f_min,
    #         fmax=self.f_max,
    #     )
    #     data = librosa.power_to_db(data, ref=np.max)
    #     data = data.astype(np.float32)
    #     return data

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.tensor, self.sample_rate = torchaudio.load(self.filepath)
        self.duration = self.tensor.shape[1] / self.sample_rate
        self.mono = True if self.tensor.shape[0] == 1 else False
        # spectrogram params
        self.n_fft = 2048  # frequency bins
        self.hop_length = 512
        # mel spectrogram params
        self.n_mels = 128
        self.f_min = 0
        self.f_max = 8000

    @property
    def waveform(self) -> torch.Tensor:
        return self.tensor

    @property
    def spectrogram(self) -> torch.Tensor:
        # https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html
        transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return transform(self.tensor)

    @property
    def mel_spectrogram(self) -> torch.Tensor:
        pass

    # When properties change all representations of the AudioClip automagically change
    # This saves having to recompute representations as you experiment with audio files

    # @name.setter   #property-name.setter decorator
    # def name(self, value):
    #     self.__name = value


class MIDIClip:
    """MIDI clip object which reads from a MIDI file andi"""

    def __init__(self, filepath: str) -> None:
        """
        filepath (str): Path to the MIDI file (.mid).
        midi (pretty_midi.PrettyMIDI): PrettyMIDI object.
        duration (float): Duration of the MIDI file in seconds.
        """
        self.filepath = filepath
        self.midi = pretty_midi.PrettyMIDI(midi_file=self.filepath)
        self.duration = self.midi.get_end_time()  # in seconds

    def pianoroll(
        self,
        resolution: int = 24,
        type: str = "binary",
        tensor_type: torch.Type = torch.int64,
    ) -> torch.Tensor:
        # TODO handle multiple tracks (e.g. piano and drums)
        # TODO clean up examples of models
        """Converts a MIDI file to a piano roll representation (CxHxW)

        The piano roll representation is a common way to represent music data.
        It consists of a two-dimensional matrix where the rows represent time steps
        and the columns represent pitches (and special symbols in certain variations). The time steps also depend on the model and level of granularity desired. For coherence, many ML models filter inputs to 4/4 time signatures and use sixteenth notes as a single time step (16 steps / bar).
        The entries of the matrix are considered as non-negative. In summary we have 3 different dimensions which are the same size in both formats.

        C - channels or tracks which typically will be different instruments
        H - height is the number of pitches
        W - width or the number of time steps (e.g. 16th notes)

        The piano roll representation for a single track or channel can be in two forms:

        1) Binary: One example is when the entries are binary, indicating whether a note is played at a given time step and pitch.

        X ∈ {0, 1}^(H×W)

        *NOTE: velocity and complex time signature differences are IGNORED in this version!*

        2) Velocity: Another variation on this form also considers the velocity of the note. In this case, the entries are integers in the range of 0-127 indicating the velocity of the note played at a given time step and pitch. In this first section, we will talk about the models that look at only the binary piano roll representation, but will revisit the velocity in the next section.

        X ∈ [0,127]^(H×W)

        *NOTE: The number of pitches in a MIDI file is 128, however the dimension of `h` may be reduced to narrow the range of notes to a specific instrument (e.g. 88 keys in a piano) or can be expanded to include silence or rests (e.g. 129/130 or 89/90).

        Args:
            resolution (int, optional): Number of ticks per beat (default: 24).
            type (str, optional): type of pianoroll: binary - 0 or 1 , velocity - 0-127 (default: "binary")
            tensor_type (torch.Type, optional): Tensor type can be updated for different hardware configs (default: torch.float32)

        Returns:
            torch.Tensor: Piano roll representation of the MIDI file in the tensor format specified (Cx128xW)

        Usage:
        >>> midi_obj = MIDIClip(midi_path)
        >>> midi_obj.piano_roll(type="binary")
        >>> [[1,..,0,...,0],
            ...,
            [0,...,1,...,0],
            ...,
            [0,...,0,...,1]]

        Examples of Models:
            * **MidiNet** (binary): piano roll representation in 4/4 time signature with 16th notes as a single time step. Each track is a new channel (e.g. CxHxW). Each sample is 1 bar and each value can be one of 128 MIDI pitches or silence resulting in a Cx129x16 matrix for each bar.
            * **MuseGAN** (binary): piano roll representation in 4/4 time signature with 16th notes as 6 time steps. That is each bar has 96 time steps. Each track is a new channel and each sample is 1 bar (e.g. CxHxW). Each value can be one of 128 MIDI pitches resulting in Cx128x96 matrix for each bar. The dimensions are rearranged to be 96x128xC.
            * **Music Transformer (Google)**: Multiple formats for different datasets and experiments
                - JS Chorales Bach: (similar to DeepBach) uses the following sequence of 4 voices (soprano, alto, tenor, bass) in 4/4 time in sixteenth note increments
                    - ![bach chorales](images/music-transformer-bach-chorales.png)
                    - Inputs are serialized $S_1A_1T_1B_1S_2A_2T_2B_2$ and each token is represented as a one-hot vector → 128 x (WxC) where W is number of sixteenth notes and C is the number of channels - in this case 4 for each voice
                - Piano E Competition Dataset: [Ref: https://arxiv.org/pdf/1808.03715.pdf] → Use a set of sequence events like the following.
                    - ![piano competition](images/music-transformer-sequence.png)
                    - Overall the total number of sequences are eventually represented as a sequence of 1-hot vectors representing each of the 388 possible events -> 388 x T where T is the the number of 10ms increments in the sample. This representation for the Piano competition dataset is discussed in section 6.1 of [this paper](https://link.springer.com/article/10.1007/s00521-018-3758-9).
                    - 128 `NOTE-ON` events: one for each of the 128 MIDI pitches. Each one starts a new note.
                    - 128 `NOTE-OFF` events: one for each of the 128 MIDI pitches. Each one releases a note.
                    - 100 `TIME-SHIFT` events: each one moves the time step forward by increments of 10 ms up to 1 s.
                    - 32 `VELOCITY` events: each one changes the velocity applied to all subsequent notes (until the next velocity event).

        """
        assert type in ["binary", "velocity"], "type must be either binary or velocity"
        ppr_midi = pypianoroll.read(self.filepath, resolution=resolution)
        if type == "binary":
            ppr_midi.binarize()
        piano_roll = ppr_midi.tracks[0].pianoroll.astype(int)
        piano_roll = np.swapaxes(piano_roll, 0, 1)
        piano_roll = torch.from_numpy(piano_roll)
        piano_roll = piano_roll.to(dtype=tensor_type)
        return piano_roll

    def notes(
        self,
        resolution: int = 24,
        type: str = "index",
    ) -> torch.Tensor:
        # TODO: clean up example of models
        """Returns a list of note objects with maximum one note per timestep (CxW).

        The note can be represented either as an index or a note value (e.g. 64 or `E4`). Here we have only one space for a note and no option for a multi-hot representation. This is useful for models that only consider one note at a time (e.g. DeepBach).

        C - channels or tracks which typically will be different instruments
        W - width or the number of time steps (e.g. 16th notes). The notes representation for a single track can either an index or a note value (e.g. 64 or `E4`).

        1) Index: The note is represented as an integer index from 0-127.

        X ∈ [0-127]^W

        2) Note: The note is represented as a string (e.g. `E4`).

        X ∈ {'C-1',....,G9}^W

        Args:
            resolution (int, optional): Number of ticks per beat (default: 24).
            type (str, optional): "index" (e.g. 64) or "note" (e.g. 'E4') (default: "index")

        Returns:
            np.Array: notes from the MIDI file in either integer or  (CxW)

        Usage:
        >>> midi_obj = MIDIClip(midi_path)
        >>> midi_obj.notes(type="index")
        >>> [[64,...,75,...,71]]

        Examples of Models:
            * **MelodyRNN** (index): "[60, -2, 60, -2, 67, -2, 67, -2]” (-2 = no event, -1=note-off event, 0-127 = note-on event for that MIDI pitch) for each track. 4/4 time signature with 16th notes as a single time step. Each bar has 16 time steps. Samples can be 2 bars or 16 bars.

            * **DeepBach** (note): 4 tracks/rows (soprano, alto, tenor, bass) with 16 time steps per bar (16th notes) represented by strings for the 128 pitches `C1` to `G9` and a hold `__` (i.e. 129 total pitches), two additional rows are added with `0` or `1` to indicate fermata and the beat count (e.g. `1,2,3,4`)

            * **MusicVAE** (index): "[60, -2, 60, -2, 67, -2, 67, -2]” (-2 = no event, -1=note-off event, 0-127 = note-on event for that MIDI pitch) for each track. 4/4 time signature with 16th notes as a single time step. Each bar has 16 time steps. Samples can be 2 bars or 16 bars.
        """
        assert type in ["index", "note"], "type must be either index or note"
        ppr_mid = pypianoroll.read(self.filepath, resolution=resolution)
        num_tracks = len(ppr_mid.tracks)
        # assume all tracks have the same # of time steps
        time_steps = ppr_mid.tracks[0].pianoroll.shape[0]
        result = []

        for i in range(len(ppr_mid.tracks)):
            track_vals = []
            for j in range(ppr_mid.tracks[i].pianoroll.shape[0]):
                note_ind = np.argmax(ppr_mid.tracks[i].pianoroll[j, :])
                if type == "index":
                    track_vals.append(note_ind)
                elif type == "note":
                    track_vals.append(pretty_midi.note_number_to_name(note_ind))
            result.append(track_vals)
        return np.array(result)

    def plot(self, type: str = "static") -> None:
        pass
        # dynamic plot via bokeh using https://github.com/dubreuia/visual_midi
        # statis using pypianoroll: https://salu133445.github.io/pypianoroll/visualization.html
