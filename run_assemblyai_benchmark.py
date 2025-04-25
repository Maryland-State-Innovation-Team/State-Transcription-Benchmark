import os
import logging
import argparse
from dotenv import load_dotenv
from datasets import load_from_disk
from benchmark_dataset import LANGUAGES
import assemblyai as aai
from io import BytesIO
import soundfile as sf
from evaluate import load
import json
from datetime import datetime
import click


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
if ASSEMBLYAI_API_KEY is None:
    logger.error('Please provide an ASSEMBLYAI_API_KEY in a .env file.')
aai.settings.api_key = f"{ASSEMBLYAI_API_KEY}"
MODEL = 'assembly-ai'
COST_PER_MINUTE = 0.0067


def assemblyai_transcribe(sample):
    transcriber = aai.Transcriber()
    file_type='mp3'
    audio_file = BytesIO()
    audio_file.name = f'sample.{file_type}'
    sf.write(audio_file, sample['audio']['array'], sample['audio']['sampling_rate'], format=file_type)
    try:
        config = aai.TranscriptionConfig(language_code=sample['locale'][:2])
        transcription = transcriber.transcribe(audio_file.getvalue(), config)
        transcription_text = transcription.text
    except aai.types.TranscriptError:
        try:
            transcription = transcriber.transcribe(audio_file.getvalue())
            transcription_text = transcription.text
        except:
            transcription_text = ''
    except:
        transcription_text = ''
    sample['transcription'] = transcription_text
    return sample


def main(args):
    results_dict = {
        'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input': args.indir,
        'model': MODEL,
        'wer': 0,
        'locales': list()
    }
    outdir = os.path.dirname(args.outfile)
    os.makedirs(outdir, exist_ok=True)
    dataset = load_from_disk(args.indir)
    # Estimate cost
    dataset = dataset.map(lambda sample: {'duration': sample['audio']['array'].shape[0] / sample['audio']['sampling_rate']})
    total_minutes = sum(dataset['duration']) / 60
    total_cost = total_minutes * COST_PER_MINUTE
    if click.confirm(
        f'Total audio length is {round(total_minutes)} minutes and estimated total cost to transcribe with model {MODEL} is ${round(total_cost, 2)}. Do you want to continue?'
    ):
        dataset = dataset.map(assemblyai_transcribe, remove_columns=["audio"])
        # Cache transcribed dataset for later, just in case
        dataset.save_to_disk(f'{args.indir}_{MODEL}_transcribed')
        wer = load("wer")
        results_dict['wer'] = wer.compute(predictions=dataset['transcription'], references=dataset['sentence'])
        unique_locales = list(set(dataset['locale']))
        cv_to_label = dict()
        for lang in LANGUAGES:
            cv = lang['common_voice_code']
            if cv is not None:
                label = lang['label']
                if type(cv) is not list:
                    cv = [cv]
                for cv_i in cv:
                    cv_to_label[cv_i] = label
        for unique_locale in unique_locales:
            language_label = cv_to_label[unique_locale]
            filtered_dataset = dataset.filter(lambda sample: sample['locale'] == unique_locale)
            locale_results_dict = {
                'locale': unique_locale,
                'n': filtered_dataset.num_rows,
                'label': language_label,
                'wer': wer.compute(predictions=filtered_dataset['transcription'], references=filtered_dataset['sentence'])
            }
            results_dict['locales'].append(locale_results_dict)
        with open(args.outfile, 'w') as json_file:
            json_file.write(json.dumps(results_dict, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='AssemblyAI benchmarks',
                description='Run benchmarks of AssemblyAI with custom voice dataset')
    parser.add_argument('-i', '--indir', default='./MD_voice_dataset')
    parser.add_argument('-o', '--outfile', default=f'./results/{MODEL}.json')
    args = parser.parse_args()
    main(args)
