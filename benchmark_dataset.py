import os
import logging
import argparse
import math
from dotenv import load_dotenv
from census import Census
from us import states
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 1634

LANGUAGES = [
  {
    'acs_code': 'B16001_002E',
    'label': 'Speak only English',
    'common_voice_code': 'en'
  },
  {
    'acs_code': 'B16001_003E',
    'label': 'Spanish',
    'common_voice_code': 'es'
  },
  {
    'acs_code': 'B16001_006E',
    'label': 'French (incl. Cajun)',
    'common_voice_code': 'fr'
  },
  {
    'acs_code': 'B16001_009E',
    'label': 'Haitian',
    'common_voice_code': None # 'ht'
  },
  {
    'acs_code': 'B16001_012E',
    'label': 'Italian',
    'common_voice_code': 'it'
  },
  {
    'acs_code': 'B16001_015E',
    'label': 'Portuguese',
    'common_voice_code': 'pt'
  },
  {
    'acs_code': 'B16001_018E',
    'label': 'German',
    'common_voice_code': 'de'
  },
  {
    'acs_code': 'B16001_021E',
    'label': 'Yiddish, Pennsylvania Dutch or other West Germanic languages',
    'common_voice_code': 'yi' # Yiddish
  },
  {
    'acs_code': 'B16001_024E',
    'label': 'Greek',
    'common_voice_code': 'el'
  },
  {
    'acs_code': 'B16001_027E',
    'label': 'Russian',
    'common_voice_code': 'ru'
  },
  {
    'acs_code': 'B16001_030E',
    'label': 'Polish',
    'common_voice_code': 'pl'
  },
  {
    'acs_code': 'B16001_033E',
    'label': 'Serbo-Croatian',
    'common_voice_code': ['sr', 'sq'] # Serbian, Albanian
  },
  {
    'acs_code': 'B16001_036E',
    'label': 'Ukrainian or other Slavic languages',
    'common_voice_code': 'uk' # Ukrainian
  },
  {
    'acs_code': 'B16001_039E',
    'label': 'Armenian',
    'common_voice_code': 'hy-AM'
  },
  {
    'acs_code': 'B16001_042E',
    'label': 'Persian (incl. Farsi, Dari)',
    'common_voice_code': 'fa'
  },
  {
    'acs_code': 'B16001_045E',
    'label': 'Gujarati',
    'common_voice_code': None # 'gu' but not in Common Voice
  },
  {
    'acs_code': 'B16001_048E',
    'label': 'Hindi',
    'common_voice_code': 'hi'
  },
  {
    'acs_code': 'B16001_051E',
    'label': 'Urdu',
    'common_voice_code': 'ur'
  },
  {
    'acs_code': 'B16001_054E',
    'label': 'Punjabi',
    'common_voice_code': 'pa-IN'
  },
  {
    'acs_code': 'B16001_057E',
    'label': 'Bengali',
    'common_voice_code': 'bn'
  },
  {
    'acs_code': 'B16001_060E',
    'label': 'Nepali, Marathi, or other Indic languages',
    'common_voice_code': 'ne-NP'
  },
  {
    'acs_code': 'B16001_063E',
    'label': 'Other Indo-European languages',
    'common_voice_code': None
  },
  {
    'acs_code': 'B16001_066E',
    'label': 'Telugu',
    'common_voice_code': 'te'
  },
  {
    'acs_code': 'B16001_069E',
    'label': 'Tamil',
    'common_voice_code': 'ta'
  },
  {
    'acs_code': 'B16001_072E',
    'label': 'Malayalam, Kannada, or other Dravidian languages',
    'common_voice_code': 'ml' # Malayalam
  },
  {
    'acs_code': 'B16001_075E',
    'label': 'Chinese (incl. Mandarin, Cantonese)',
    'common_voice_code': 'zh-CN'
  },
  {
    'acs_code': 'B16001_078E',
    'label': 'Japanese',
    'common_voice_code': 'ja'
  },
  {
    'acs_code': 'B16001_081E',
    'label': 'Korean',
    'common_voice_code': 'ko'
  },
  {
    'acs_code': 'B16001_084E',
    'label': 'Hmong',
    'common_voice_code': None
  },
  {
    'acs_code': 'B16001_087E',
    'label': 'Vietnamese',
    'common_voice_code': 'vi'
  },
  {
    'acs_code': 'B16001_090E',
    'label': 'Khmer',
    'common_voice_code': 'kmr'
  },
  {
    'acs_code': 'B16001_093E',
    'label': 'Thai, Lao, or other Tai-Kadai languages',
    'common_voice_code': ['th', 'lo']
  },
  {
    'acs_code': 'B16001_096E',
    'label': 'Other languages of Asia',
    'common_voice_code': None
  },
  {
    'acs_code': 'B16001_099E',
    'label': 'Tagalog (incl. Filipino)',
    'common_voice_code': None # 'tl' or 'fil'
  },
  {
    'acs_code': 'B16001_102E',
    'label': 'Ilocano, Samoan, Hawaiian, or other Austronesian languages',
    'common_voice_code': None # 'ilo', 'sm', 'haw'
  },
  {
    'acs_code': 'B16001_105E',
    'label': 'Arabic',
    'common_voice_code': 'ar'
  },
  {
    'acs_code': 'B16001_108E',
    'label': 'Hebrew',
    'common_voice_code': 'he'
  },
  {
    'acs_code': 'B16001_111E',
    'label': 'Amharic, Somali, or other Afro-Asiatic languages',
    'common_voice_code': 'am' # Amharic, 'so' for Somali
  },
  {
    'acs_code': 'B16001_114E',
    'label': 'Yoruba, Twi, Igbo, or other languages of Western Africa',
    'common_voice_code': 'yo' # 'ak', 'tw', 'twi', 'ig', 'ff', 'ful', 'kr', 'kau'
  },
  {
    'acs_code': 'B16001_117E',
    'label': 'Swahili or other languages of Central, Eastern, and Southern Africa',
    'common_voice_code': 'sw' # 'kg', 'kon'
  },
  {
    'acs_code': 'B16001_120E',
    'label': 'Navajo',
    'common_voice_code': None # 'nv'
  },
  {
    'acs_code': 'B16001_123E',
    'label': 'Other Native languages of North America',
    'common_voice_code': None # 'cr' Cree, 'iu' for Inuktitut, 'ik' for Iñupiaq, 'oj' for Ojibwe
  },
  {
    'acs_code': 'B16001_126E',
    'label': 'Other and unspecified languages',
    'common_voice_code': None
  }
]


# Construct a Mozilla Common Voice benchmark dataset in proportion to
# the percentage languages spoken at home in a given state, according the ACS 5-year 2023
def construct_dataset(state_abbrevation, out_length):
    load_dotenv()
    CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
    if CENSUS_API_KEY is None:
        logger.error('Please provide a CENSUS_API_KEY in a .env file.')
        return
    HF_TOKEN = os.getenv('HF_TOKEN')
    if HF_TOKEN is None:
        logger.error('Please provide a HF_TOKEN in a .env file.')
        return
    c = Census(CENSUS_API_KEY, year=2023)
    try:
        state_fips = states.lookup(state_abbrevation).fips
    except AttributeError:
        logger.error(f'{state_abbrevation} is not a recognized State abbreviation.')
        return
    logger.info('Fetching language spoken at home from ACS 5-Year...')
    acs_response = c.acs5.get(
        (
            'B16001_001E', # Total Population 5 Years and Over
        ) + tuple(
            lang['acs_code'] for lang in LANGUAGES
        ),
        {'for': f'state:{state_fips}'}
    )
    acs_language_counts = acs_response[0]
    del acs_language_counts['state']
    denominator = acs_language_counts['B16001_001E']
    del acs_language_counts['B16001_001E']
    acs_language_percents = {key: numerator / denominator for key, numerator in acs_language_counts.items()}
    acs_languages_not_in_cv = [lang['acs_code'] for lang in LANGUAGES if lang['common_voice_code'] is None]
    represented_language_percents = {key: value for key, value in acs_language_percents.items() if key not in acs_languages_not_in_cv}
    not_represented_language_percents = {key: value for key, value in acs_language_percents.items() if key in acs_languages_not_in_cv}
    sum_not_represented_languages = round(sum(not_represented_language_percents.values()) * 100, 1)
    sorted_not_represented_language_percents = dict(sorted(not_represented_language_percents.items(), key=lambda item: item[1], reverse=True))
    acs_to_label = {lang['acs_code']: lang['label'] for lang in LANGUAGES}
    logger.warning(
        (
            f'{sum_not_represented_languages}% of languages spoken at home in {state_abbrevation} are not represented in the Common Voice dataset:'
            f'\n{'\n'.join([f'- {acs_to_label[key]}: {round(value * 100, 3)}%' for key, value in sorted_not_represented_language_percents.items()])}'
        )
    )
    # Adjust for not represented languages
    represented_language_percents = {key: value / (1 - sum(not_represented_language_percents.values())) for key, value in represented_language_percents.items()}
    acs_to_cv = {lang['acs_code']: lang['common_voice_code'] for lang in LANGUAGES}
    voice_datasets = list()
    logger.info('Downloading Common Voice datasets...')
    for acs_code, language_percent in tqdm(represented_language_percents.items()):
        common_voice_code = acs_to_cv[acs_code]
        logger.info(f'Downloading Common Voice dataset: {common_voice_code}')
        out_count = math.ceil(max(language_percent * out_length, 1))
        if type(common_voice_code) is not list:
            common_voice_code = [common_voice_code]
        for indv_cv_code in common_voice_code:
            indv_out_count = math.ceil(max(out_count / len(common_voice_code), 1))
            language_voice_dataset_iter = load_dataset(
                'mozilla-foundation/common_voice_17_0', indv_cv_code, split='test', trust_remote_code=True, streaming=True
            )
            language_voice_dataset_iter = language_voice_dataset_iter.shuffle(seed=SEED).take(indv_out_count).select_columns(['locale', 'audio', 'sentence'])
            language_voice_dataset = Dataset.from_generator(lambda: (yield from language_voice_dataset_iter), features=language_voice_dataset_iter.features)
            found_test_samples = language_voice_dataset.num_rows
            if found_test_samples != indv_out_count:
                logger.warning(f'Unable to find sufficient samples for {common_voice_code}. Needed: {indv_out_count}; Found: {found_test_samples}')
            voice_datasets.append(language_voice_dataset)
    state_voice_dataset = concatenate_datasets(voice_datasets)
    return state_voice_dataset


def main(args):
    state_outdir = os.path.join(args.outdir, f'{args.state_abbrevation}_voice_dataset')
    os.makedirs(state_outdir, exist_ok=True)
    dataset = construct_dataset(args.state_abbrevation, args.length)
    if dataset is not None:
        dataset.save_to_disk(state_outdir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='Benchmark dataset',
                description='Construct a state-specific audio dataset')
    parser.add_argument('-n', '--length', type=int, default=10000, help='The baseline length of output. Actual output may be higher as counts are rounded up.')
    parser.add_argument('-o', '--outdir', default='./')
    parser.add_argument('state_abbrevation', type=str, help='The two letter abbreviation of the state (e.g. MD for Maryland)')
    args = parser.parse_args()
    main(args)
