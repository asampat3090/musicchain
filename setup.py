from setuptools import setup, find_packages

setup(
    name="audiochain",
    version="0.1",
    packages=find_packages(),
    description="A package to chain together multiple audio based ML models",
    author="anand sampat",
    author_email="anands@cs.stanford.edu",
    url="https://github.com/asampat3090/audiochain",
    install_requires=[
        # torch,
        # torchaudio,
        # audiocraft,
        # pydub,
        # tqdm,
        # pretty_midi,
        # pypianoroll
    ],
)
