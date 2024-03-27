import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import os
import math

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime
from nltk.metrics.segmentation import pk, windowdiff
from hydra import compose, initialize
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


def log(s):
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('[%Y.%m.%d-%H:%M:%S]')
    print(formatted_time + ' ' + s)


def ensure_directory_exists(path):
    if os.path.isfile(path):
        dir_path = os.path.dirname(path)
    else:
        dir_path = path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_mean_metrics(wds, pks, f1s):
    wd_mean = sum(wds) / len(wds)
    pk_mean = sum(pks) / len(pks)
    f1_mean = sum(f1s) / len(f1s)
    return wd_mean, pk_mean, f1_mean


def calc_metric(example, pred_boundaries, refs_passed=False, verbose=False):
    if not refs_passed:
        ref_boundaries = example['boundaries']
    else:
        ref_boundaries = example
    
    assert ref_boundaries.count("1") > 0, f'No boundaries found: {example}'
    k = int(round(len(ref_boundaries) / (ref_boundaries.count("1") * 2.0)))
    wd_ = windowdiff(ref_boundaries, pred_boundaries, k)
    
    pk_ = pk(ref_boundaries, pred_boundaries)
    
    y_true = list(map(int, ref_boundaries))
    y_pred = list(map(int, pred_boundaries))
    
    if verbose:
        precision_ = precision_score(y_true, y_pred)
        recall_ = recall_score(y_true, y_pred)
    f1_ = f1_score(y_true, y_pred)
    
    if verbose:
        print()
        print(ref_boundaries, pred_boundaries, sep='\n---\n')
        print(wd_, pk_, f1_, precision_, recall_, end='\n\n')
    return wd_, pk_, f1_


def calc_metrics(dataset, boundaries):
    wds = []
    pks = []
    f1s = []
    t = tqdm(enumerate(dataset))
    for i, sample in t:
        wd, pk, f1 = calc_metric(sample, boundaries[i])
        wds.append(wd)
        pks.append(pk)
        f1s.append(f1)
        
        # description = f'wd: {sum(wds) / len(wds):.3f}, '
        # description += f'pk: {sum(pks) / len(pks):.3f}, '
        # description += f'f1: {sum(f1s) / len(f1s):.3f}'
        # t.set_description(description)
    return wds, pks, f1s


def embed(example, model, gpu_batch_size=512):
    lengths = [len(section) for section in example['sections']]
    all_sentences = []
    for section in example['sections']:
        for sentence in section:
            all_sentences.append(sentence)
    embeddings = model.encode(all_sentences,
                               show_progress_bar=False,
                               batch_size=gpu_batch_size,
                               convert_to_tensor=True)
    slices = torch.split(embeddings, lengths)
    example['embeddings'] = list(slices)
    return example


def load_config_by(arg_dict, config_path):
    overrides = [f"{key}={val}" for key, val in arg_dict.items() if val is not None]
    try:
        initialize(version_base=None, config_path=config_path)
    except:
        print('Already initialized hydra')
    cfg = compose(config_name="config", overrides=overrides)
    return cfg


def get_sentence_encoder(cfg):
    device = 'cuda' if cfg.gpus else 'cpu'
    if cfg.russian_language:
        sentence_encoder = RussianModel(device=device)
    else:
        sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return sentence_encoder


class RussianModel:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        try:
            self.model = AutoModel.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")
            self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")
        except:
            print('Failed to load sentence encoder in russian, trying to load locally...')
            self.model = AutoModel.from_pretrained("sbert_large_mt_nlu_ru")
            self.tokenizer = AutoTokenizer.from_pretrained("sbert_large_mt_nlu_ru")
        self.model.to(device)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def __call__(self, batch):
        return self.encode(batch, batch_size=len(batch))

    def encode(self, sentences, batch_size=16, **kwargs):
        num_sentences = len(sentences)
        num_batches = math.ceil(num_sentences / batch_size)
        sentence_embeddings = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            batch_sentences = sentences[start_index:end_index]

            encoded_inputs = self.tokenizer(batch_sentences, 
                                            padding=True, 
                                            truncation=True,
                                            max_length=512, 
                                            pad_to_multiple_of=8,
                                            return_tensors="pt")
            encoded_inputs = encoded_inputs.to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_inputs)

            batch_embeddings = self.mean_pooling(model_output, encoded_inputs['attention_mask'])
            sentence_embeddings.append(batch_embeddings)

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings)
        return sentence_embeddings

