# AudChain or AudioChain

[LangChain](https://python.langchain.com/docs/get_started/introduction.html) has garnered significant popularity as a means to string together models and their requisite data processing for generative language models. 

AudioChain is doing this for audio models including voice generation, music generation and others. In particular, we aim to create a framework for audio models that is as easy to use as LangChain is for language models.

We want to enable the following use cases:
* Simple Input / Output for musicians used to working with DAWs
* Prompt Engineering and Prompt Tuning (same as LangChain)
* Enable Simple Fine-Tuning of models using your own musical repertoire


Key Differences (TBD)
* Focus on audio rather than language: musicians work with audio clips rather than language and work off of those variations to craft the perfect song
* Audio data requires different types of encodings and processing than language data - we make it simple to work with audio data 
* Audio chaining is more about chaining audio outputs temporally to craft larger audio files 

## Installation 
```bash
pip install audiochain
```

### Dependencies 

* ffmpeg 
* pydub 

## Token to Time Conversion

AudioChain uses standard time conversion for manipulating audio files. This means that you can specify a time in seconds and the model will aim to use that for processing and training of models. 

AudioChain automatically converts the token size inputs of a model (for inference or training time) to appropriate size audio segments. 

Using ffmpeg, the tool automagically loads in your files, extracts the sampling rates and resamples based on the needs of the model specified for training or inference and then will tokenize it based on the window size of the model. 

For example,......

## Model Selection 

AudioChain is designed to be modular and flexible. It is designed to be able to work with any model that can be trained on audio data. Our initial goal is to help musicians (i.e. audiophiles) to create variations of their own melodies and vocals. 

1) Vocal Generator (WIP)
2) Music Generator (WIP)
3) Speech Generator (WIP)

## Installation 

We will specify the installation either via CPU or GPU. Many of the audio manipulation libraries use ffmpeg or other binaries which also need to be installed. To leverage the full power of AudioChain, we recommend using a GPU so you can run and fine-tune models faster. 