# 🎵 SongChain 🔗 (WIP) 

[LangChain](https://python.langchain.com/docs/get_started/introduction.html) has garnered significant popularity as a means to string together models and their requisite data processing for generative language models. 

SongChain is doing this for ML models used for music including voice generation, audio and MIDI-based music generation and others. In particular, we aim to create a framework for music models that is as easy to use as LangChain is for language models.

We want to enable the following use cases:
* Simple Input / Output for musicians used to working with DAWs
* Prompt Engineering and Prompt Tuning (same as LangChain)
* Enable Simple Fine-Tuning of ML models for music using your own musical repertoire


Key Differences (WIP)
* Focus on audio rather than language: musicians work with audio and MIDI clips rather than language and work off of those variations to craft the perfect song
* Audio and MIDI data require different types of encodings and processing than language data - we make it simple to work with audio data 
* Audio and MIDI chaining is more about chaining audio and MIDI outputs temporally to craft larger audio files 

## Installation 
```bash
pip install songchain
```

### Dependencies (WIP)

* ffmpeg 
* pydub 
* pypianoroll
* librosa 
* torchaudio
* pretty_midi
* mido

## Token to Time Conversion

SongChain uses standard time conversion for manipulating audio files. This means that you can specify a time in seconds and the model will aim to use that for processing and training of models. 

SongChain automatically converts the token size inputs of a model (for inference or training time) to appropriate size audio segments. 

Using ffmpeg, the tool automagically loads in your files, extracts the sampling rates and resamples based on the needs of the model specified for training or inference and then will tokenize it based on the window size of the model. 

For example,......

## Model Selection 

AudioChain is designed to be modular and flexible. It is designed to be able to work with any model that can be trained on audio or MIDI data. Our initial goal is to help musicians (i.e. audiophiles) to create variations of their own melodies and vocals. 

1) Vocal Generator (WIP)
2) Music Generator (WIP)
3) Speech Generator (WIP)

## Installation 

We will specify the installation either via CPU or GPU. Many of the audio manipulation libraries use ffmpeg or other binaries which also need to be installed. To leverage the full power of SongChain, we recommend using a GPU so you can run and fine-tune models faster. 