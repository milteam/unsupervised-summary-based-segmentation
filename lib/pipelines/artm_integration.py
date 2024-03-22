import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import re
import os
import hydra
import pickle
import segeval
from datasets import Dataset, load_from_disk

from utilities.artm_utils.model import build_artm_dataset, \
    train_model, load_vectorizer_and_dict, load_model, topic_probs_prediction
from utilities.artm_utils.text_processing import tokenize_dataset_sample, \
    create_dictionary, convert_to_uci
from utilities.dataset import WikiDataset, AMIDataset
from utilities.general import log, ensure_directory_exists,\
    calc_metrics, get_mean_metrics
from utilities.tiling import classify_borders


def vectorize_dataset(cfg):
    log(f'Starting to load or create the dataset {cfg.dataset_type}...')
    
    if cfg.dataset_type == 'wiki':
        generator = WikiDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type == 'ami':
        generator = AMIDataset(cfg.input_path).get_generator()
    else:
        raise ValueError(f'No such dataset type {cfg.dataset_type} exist!')
    
    ds = Dataset.from_generator(generator)
    log('Dataset is built. Tokenizing texts...')
    ds_tokenized = ds.map(tokenize_dataset_sample, batched=True,
                          batch_size=cfg.batch_size, num_proc=cfg.num_cpu,
                          fn_kwargs={'russian': cfg.russian_language})
    tokenized_corpus = ds_tokenized['tokens']
    log('Text is tokenized. Starting to build a dictionary...')
    if cfg.dictionary is None:
        dictionary = create_dictionary(tokenized_corpus, min_freq=cfg.artm_config.min_dict_freq)
    else:
        dictionary = dict()
        with open(cfg.dictionary, 'r') as file:
            for word in file:
                dictionary[word.strip()] = len(dictionary) + 1
    log('Dictinory is loaded/created. Converting to UCI format...')
    if cfg.dictionary is None:
        # if no dict is provided it means that we are working with train (not good to use implicit way)
        convert_to_uci(tokenized_corpus, dictionary, cfg.output_path, cfg.collection, join_sentences=True)
    else:
        convert_to_uci(tokenized_corpus, dictionary, cfg.output_path, cfg.collection, join_sentences=False)
    log('Converting is finished.')
    build_artm_dataset(cfg.collection, cfg.output_path, os.path.join(cfg.output_path, 'batches'))

def train_and_dump(cfg):
    vectorizer, dictionary = load_vectorizer_and_dict(os.path.join(cfg.input_path, 'batches'))
    train_model(vectorizer, dictionary, model_path=cfg.output_path, topics_qty=cfg.artm_config.topics_qty,
                dec_phi=cfg.artm_config.dec_phi, collection_passes=cfg.artm_config.collection_passes,
                tokens_qty=cfg.artm_config.tokens_qty, sp_phi=cfg.artm_config.sp_phi, sp_pheta=cfg.artm_config.sp_pheta)

def load_and_predict(cfg):
    model = load_model(cfg.model_path, cfg.dictionary)
    vectorizer, _ = load_vectorizer_and_dict(os.path.join(cfg.input_path, 'batches'))
    pattern = r'mapping\..+\.pickle'
    files_in_directory = os.listdir(cfg.input_path)
    matching_files = [file for file in files_in_directory if re.match(pattern, file)]
    if len(matching_files) != 1:
        raise FileNotFoundError('Zero or more than one mapping files found')
    with open(os.path.join(cfg.input_path, matching_files[0]), 'rb') as file:
        mapping = pickle.load(file)
    predicts = topic_probs_prediction(model, vectorizer, mapping)
    borders = []
    for predict in predicts:
        borders.append(classify_borders(predict, window_size=cfg.artm_config.tiling_window_size,
                                        threshold=cfg.artm_config.tiling_threshold,
                                        smoothing_passes=cfg.artm_config.smoothing_passes,
                                        smoothing_window=cfg.artm_config.smoothing_window))
    ensure_directory_exists(cfg.output_path)
    with open(os.path.join(cfg.output_path, 'predicts.pickle'), 'wb') as file:
        pickle.dump(borders, file)

def calculate_scores(cfg):
    log(f'Starting to load or create the dataset {cfg.dataset_type}...')
    
    if cfg.dataset_type == 'wiki':
        generator = WikiDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type == 'ami':
        generator = AMIDataset(cfg.input_path).get_generator()
    else:
        raise ValueError(f'No such dataset type {cfg.dataset_type} exist!')
    
    ds = Dataset.from_generator(generator)
    log('Dataset is built. Loading the predicts...')
    with open(cfg.predicts_path, 'rb') as file:
        predicts = pickle.load(file)

    log('Calculating the metrics...')
    wds, pks, f1s = calc_metrics(ds, predicts)
    wd_mean, pk_mean, f1_mean = get_mean_metrics(wds, pks, f1s)
    log(f'Calculated WD for the dataset {cfg.dataset_type} is {wd_mean:.5f}.')
    log(f'Calculated PK for the dataset {cfg.dataset_type} is {pk_mean:.5f}.')
    log(f'Calculated F1 for the dataset {cfg.dataset_type} is {f1_mean:.5f}.')