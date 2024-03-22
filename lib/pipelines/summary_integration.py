import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import optuna
from tqdm import tqdm
import nltk
import spacy
import torch
import numpy as np
from scipy.special import softmax
from transformers import LEDTokenizer, LEDForConditionalGeneration, pipeline
from datasets import load_from_disk
from utilities.dataset import get_mean_segment_length, load_dataset_by, get_savgol_k
from utilities.general import log, embed, calc_metric, get_sentence_encoder
from utilities.tiling import TopicTilingModel, classify_borders
from transformers.models.led import LEDConfig
from transformers.models.t5 import T5Config
from transformers.models.pegasus import PegasusConfig

SYNTAX_PARSER = spacy.load('en_core_web_sm')


def _generate_summary_led(text, model, tokenizer):
    # For LED models
    inputs_dict = tokenizer(text, padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to('cuda')
    attention_mask = inputs_dict.attention_mask.to('cuda')
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    summary = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    return summary

def _summarize_led(example, model=None, tokenizer=None):
    # For LED models
    text = ' '.join(example['sections'])
    try:
        example['summary'] = _generate_summary_led(text, model, tokenizer)[0]
    except:
        print(f'Failed to summarize an example: {example}')
        example['summary'] = '-1'
    return example


def _generate_summary(text, model):
    # For BART, Flan T5 models
    def preprocess_input(s):
        s = s.replace('\n', ' ').strip()
        return s
    
    def get_chunks(s, max_tokens, tokenizer):
        tokenized_text = tokenizer.encode(s)
        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokenized_text:
            current_chunk.append(token)
            current_length += 1

            if current_length >= max_tokens:
                text = tokenizer.decode(current_chunk, skip_special_tokens=True).strip(' .,;')
                text = preprocess_input(text)
                chunks.append(text)
                current_chunk = []
                current_length = 0

        if current_chunk:
            text = tokenizer.decode(current_chunk, skip_special_tokens=True).strip(' .,;')
            text = preprocess_input(text)
            chunks.append(text)
            
        for chunk in chunks:
            yield chunk
        
    if model.model.config._name_or_path == 'IlyaGusev/mbart_ru_sum_gazeta':
        # Should work with mBART (IlyaGusev/mbart_ru_sum_gazeta)
        input_length = 600
        model_min_length = 30
        model_max_length = 200
    elif type(model.model.config) is LEDConfig:
        # Should work with LED models
        input_length = 16384
        model_min_length = model.model.config.min_length
        model_max_length = model.model.config.max_length
    elif type(model.model.config) is T5Config:
        # Should work with T5 models
        input_length = model.model.config.n_positions
        model_min_length = model.model.config.task_specific_params['summarization']['min_length']
        model_max_length = model.model.config.task_specific_params['summarization']['max_length']
    elif type(model.model.config) is PegasusConfig:
        # Should work with Pegasus models
        input_length = model.model.config.max_position_embeddings
        model_min_length = model.model.config.min_length
        model_max_length = model.model.config.max_length
    else:
        # Should work with BART models
        input_length = model.model.config.max_position_embeddings
        model_min_length = model.model.config.task_specific_params['summarization']['min_length']
        model_max_length = model.model.config.task_specific_params['summarization']['max_length']
    
    summary = ''
    
    for c in get_chunks(preprocess_input(text), input_length - 10, model.tokenizer):
        n = len(model.tokenizer(c)['input_ids'])
        if n > 4:
            try:
                s = model(c, min_length=min(model_min_length, int(0.9 * n)-2), max_length=min(model_max_length, n-2))
                s = s[0]['summary_text'].strip() + ' '
                summary += s
            except:
                print(f'Failed to summarize! Find out the problem with PDB:')
                breakpoint()
    summary = preprocess_input(summary)
    return summary


def _summarize(example, model):
    # For BART, Flan T5 models
    text = ' '.join(example['sections'])
    example['summary'] = _generate_summary(text, model)
    return example


def _find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
    return root_token


def _find_other_verbs(doc, root_token):
    other_verbs = []
    for token in doc:
        ancestors = list(token.ancestors)
        if (token.pos_ == "VERB" and len(ancestors) == 1
                and ancestors[0] == root_token):
            other_verbs.append(token)
    return other_verbs


def _get_clause_token_span_for_verb(verb, doc, all_verbs):
    first_token_index = len(doc)
    last_token_index = 0
    this_verb_children = list(verb.children)
    for child in this_verb_children:
        if child not in all_verbs:
            if child.i < first_token_index:
                first_token_index = child.i
            if child.i > last_token_index:
                last_token_index = child.i
    return first_token_index, last_token_index


def _process_sentence(sentence):
    doc = SYNTAX_PARSER(sentence)
    root = _find_root_of_sentence(doc)
    other_verbs = _find_other_verbs(doc, root)

    token_spans = []
    all_verbs = [root] + other_verbs
    for other_verb in all_verbs:
        if other_verb is None:
            continue
        first_token_index, last_token_index = _get_clause_token_span_for_verb(other_verb, doc, all_verbs)
        token_spans.append((first_token_index, last_token_index))

    sentence_clauses = []
    for token_span in token_spans:
        start = token_span[0]
        end = token_span[1]
        if start < end:
            clause = doc[start:end]
            sentence_clauses.append(clause)

    sentence_clauses = sorted(sentence_clauses, key=lambda tup: tup[0])
    clauses_text = [clause.text for clause in sentence_clauses]
    return clauses_text


def _calculate_prob_vector(batch):
    prob_vectors = []
    for emb, emb_sum in zip(batch['embeddings'], batch['embeddings_summary']):
        if len(emb_sum) == 0 or len(emb) == 0:
            prob_vectors.append(emb)
            continue
        emb = np.array(emb)
        emb_sum = np.array(emb_sum).T
        logits = np.matmul(emb, emb_sum)
        probs = softmax(logits, axis=1)
        prob_vectors.append(probs)

    batch['probs'] = prob_vectors
    return batch


def split_sent_batched(example):
    doc_summary = example['summary']
    sentences = nltk.tokenize.sent_tokenize(doc_summary, language='english')
    simple_sentences = []
    for sentence in sentences:
        splitted_sentence = _process_sentence(sentence)
        simple_sentences += splitted_sentence

    example['splitted_summary'] = simple_sentences
    return example


def embed_summary(example, model, gpu_batch_size=512):
    lengths = [len(section) for section in example['splitted_summary']]
    all_sentences = []
    for section in example['splitted_summary']:
        for sentence in section:
            all_sentences.append(sentence)
    embeddings = model.encode(all_sentences,
                               show_progress_bar=False,
                               batch_size=gpu_batch_size,
                               convert_to_tensor=True)
    slices = torch.split(embeddings, lengths)
    example['embeddings_summary'] = list(slices)
    return example


def _get_borders_sumseg(example, tiling_model, plot):
    probabilities = example['probs']
    boundaries = tiling_model.transform(probabilities, gold_boundaries=example['boundaries'], plot=plot)
    return boundaries


###

def _calculate_metric_on_example(cfg, example, topictiling_parameters=None):
    tiling_model = TopicTilingModel(**topictiling_parameters)
    
    try:
        boundary = _get_borders_sumseg(example, tiling_model, cfg.sumseg.tiling.plot)
    except:
        log('Failed to infer sumseg! Will try to use TopicTiling on embeddings then...')
        boundary = classify_borders(example['embeddings'], 
                                    window_size=cfg.tiling_window_size,
                                    threshold=cfg.tiling_threshold, 
                                    smoothing_passes=cfg.smoothing_passes,
                                    smoothing_window=cfg.smoothing_window)
        
    wd, pk, f1 = calc_metric(example, boundary)
    return boundary, wd, pk, f1


def _calculate_metrics_on_dataset(cfg, ds, topictiling_parameters=None):
    wds = []
    pks = []
    f1s = []
    boundaries = []
    wd_mean, pk_mean, f1_mean = None, None, None
    
    t = tqdm(ds)
    for example in t:
        boundary, wd, pk, f1 = _calculate_metric_on_example(cfg, example, topictiling_parameters=topictiling_parameters)

        boundaries.append(boundary)
        wds.append(wd)
        pks.append(pk)
        f1s.append(f1)
        
        wd_mean = sum(wds) / len(wds)
        pk_mean = sum(pks) / len(pks)
        f1_mean = sum(f1s) / len(f1s)
        description = f'wd: {wd_mean:.5f}, '
        description += f'pk: {pk_mean:.5f}, '
        description += f'f1: {f1_mean:.5f}'
        t.set_description(description)
        
    wd_mean = sum(wds) / len(wds)
    pk_mean = sum(pks) / len(pks)
    f1_mean = sum(f1s) / len(f1s)
    
    return boundaries, wd_mean, pk_mean, f1_mean


def _find_parameters_topictiling(cfg, ds):
    def objective(trial):
        '''Returns mean WD metric'''
        min_window_size = max(int(get_mean_segment_length(cfg) / 10), 1)
        max_window_size = int(get_mean_segment_length(cfg))
        params = {
            'window_size': trial.suggest_int('window_size', min_window_size, max_window_size), 
            'threshold': trial.suggest_float('threshold', 0.50, 0.90),
            'smoothing_passes': trial.suggest_categorical('smoothing_passes', [1, 1]),
            'smoothing_window': trial.suggest_categorical('smoothing_window', [1, min_window_size, max_window_size]),
            'n_smooth_savgol': trial.suggest_categorical('n_smooth_savgol', [1, 3]), 
            'savgol_k': trial.suggest_float('savgol_k', get_savgol_k(cfg) * 0.5, get_savgol_k(cfg) * 1.5),
            'polyorder': trial.suggest_categorical('polyorder', [3])
        }
        _, wd_mean, _, _ = _calculate_metrics_on_dataset(cfg, ds, topictiling_parameters=params)
        return wd_mean
    
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=cfg.n_trials)
    return study.best_params

###

def summarize_and_save(cfg):
    log(f'Loading dataset {cfg.input_path}')
    ds = load_dataset_by(cfg)
    log('Dataset generator is built. Loading a sentence-bert model...')
    sentence_model = get_sentence_encoder(cfg)
    
    log('Calculating embeddings for the dataset...')
    ds = ds.map(embed, batched=True, batch_size=cfg.sbert.cpu_batch_size,
                         fn_kwargs={'model': sentence_model, 'gpu_batch_size': cfg.sbert.gpu_batch_size})

    log('Calculating summaries for the dataset...')
    model_type = 'bart'
    if model_type == 'led':
        summary_model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed").to('cuda:0')
        summary_tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
        ds = ds.map(_summarize, fn_kwargs={'model': summary_model, 'tokenizer': summary_tokenizer})
    else:
        model_name = "philschmid/bart-large-cnn-samsum"
        # model_name = "philschmid/flan-t5-base-samsum"
        # model_name = "rooftopcoder/led-base-book-summary-samsum"
        # model_name = "facebook/bart-large-cnn"
        # model_name = "google/flan-t5-base"
        model = pipeline("summarization", model=model_name, device='cuda:0')
        ds = ds.map(_summarize, fn_kwargs={'model': model})
    
    log('Splitting summaries on simple sentences...')
    ds = ds.map(split_sent_batched, num_proc=16)
    log('Calculating embeddings for splitted summaries...')
    ds = ds.map(embed_summary, batched=True, batch_size=cfg.sbert.cpu_batch_size,
                                       fn_kwargs={'model': sentence_model, 'gpu_batch_size': cfg.sbert.gpu_batch_size})
    log('Calculating closeness to summaries...')
    ds = ds.map(_calculate_prob_vector, batch_size=cfg.batch_size,
                                            num_proc=cfg.num_cpu, batched=True)
    ds = ds.with_format('numpy', columns=['embeddings', 'embeddings_summary', 'probs', 'boundaries', 'summary', 'splitted_summary', 'sections'])
    ds.save_to_disk(cfg.output_path)


def get_scores_summary(cfg):
    log(f'Starting to load or create the dataset {cfg.dataset_type}...')
    ds_val = load_from_disk(cfg.val_path)
    ds_test = load_from_disk(cfg.test_path)
    
    log('Calculating scores...')
    
    topictiling_parameters = {
            'window_size': cfg.sumseg.tiling.window_size, 
            'threshold': cfg.sumseg.tiling.threshold, 
            'smoothing_passes': cfg.sumseg.tiling.smoothing_passes,
            'smoothing_window': cfg.sumseg.tiling.smoothing_window,
            'n_smooth_savgol': cfg.sumseg.tiling.savgol.n_smooth_savgol,
            'polyorder': cfg.sumseg.tiling.savgol.polyorder,
            'savgol_k': cfg.sumseg.tiling.savgol.savgol_k,
        }
    
    if cfg.find_topictiling_parameters:
        topictiling_parameters = _find_parameters_topictiling(cfg, ds_val)
        _, wd_mean, pk_mean, f1_mean = _calculate_metrics_on_dataset(cfg, ds_val, topictiling_parameters=topictiling_parameters)
        score = (2*f1_mean+(1-pk_mean)+(1-wd_mean)) / 4
        output = f'''
Using TT parameters after {cfg.n_trials} trials: {topictiling_parameters}
Metrics for {cfg.val_path} VAL:
WD {wd_mean:.5f}
PK {pk_mean:.5f}
F1 {f1_mean:.5f}
SCORE {score:.5f}
        '''
        print(output)
        
    _, wd_mean, pk_mean, f1_mean = _calculate_metrics_on_dataset(cfg, ds_test, topictiling_parameters=topictiling_parameters)
    score = (2*f1_mean+(1-pk_mean)+(1-wd_mean)) / 4
    output = f'''
Using TT parameters: {topictiling_parameters}
Metrics for {cfg.test_path} TEST:
WD {wd_mean:.5f}
PK {pk_mean:.5f}
F1 {f1_mean:.5f}
SCORE {score:.5f}
    '''
    print(output)
