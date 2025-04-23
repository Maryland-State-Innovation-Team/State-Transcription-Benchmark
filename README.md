# State-Transcription-Benchmark
A pipeline to construct state-specific audio transcription benchmarks


## Setup
1. Register for a US Census API key and enter CENSUS_API_KEY in .env file
2. Register for Huggingface account and enter HF_TOKEN in .env file
3. Visit [Mozilla's Common Voice Corpus 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) and accept the terms of use

## Installation

```bash
python -m virtualenv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Use as an import
```python
from benchmark_dataset import construct_dataset

# Construct a Maryland-weighted voice dataset with no fewer than 10,000 samples
my_state_dataset = construct_dataset('MD', 10000)
```

## CLI use
```bash
python benchmark_dataset.py MD # Save a Maryland dataset of >10,000 samples as ./MD_voice_dataset
python run_openai_benchmark.py # Run a WER evaluation of ./MD_voice_dataset
```