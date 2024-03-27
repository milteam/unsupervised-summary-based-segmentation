import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import os
import re
import optuna
from tqdm import tqdm
import nltk
import spacy
import torch
import numpy as np
from scipy.special import softmax
from transformers import pipeline, LEDTokenizer, LEDForConditionalGeneration, MBartTokenizer, \
    MBartForConditionalGeneration, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, \
        AutoModelForSeq2SeqLM, BigBirdPegasusForConditionalGeneration
from datasets import load_from_disk
from utilities.dataset import get_mean_segment_length, load_dataset_by, get_savgol_k, colorize, get_topics
from utilities.general import log, embed, calc_metric, get_sentence_encoder
from utilities.tiling import TopicTilingModel, classify_borders


def _generate_summary_led(text, model, tokenizer, device, min_length, max_length):
    # For LED models
    with torch.no_grad():
        inputs_dict = tokenizer(text, padding="max_length", min_length=min_length, max_length=max_length,
                                return_tensors="pt", truncation=True)
        input_ids = inputs_dict.input_ids.to(device)
        attention_mask = inputs_dict.attention_mask.to(device)
        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1

        predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask,
                                                global_attention_mask=global_attention_mask)
        summary = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
        return summary


def _generate_summary(text, model_name, device):
    no_repeat_ngram_size = 4
    num_beams = 5
                    
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

    if model_name == 'IlyaGusev/mbart_ru_sum_gazeta':
        # Should work with mBART (IlyaGusev/mbart_ru_sum_gazeta)
        model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = MBartTokenizer.from_pretrained(model_name)

        input_length = 600
        model_min_length = 30
        model_max_length = 200

        def return_summary(text, min_length, max_length):
            with torch.no_grad():
                input_ids = tokenizer(
                    [text],
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )["input_ids"].to(device)
                output_ids = model.generate(
                    input_ids=input_ids,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    num_beams=num_beams
                )[0]
                summary = tokenizer.decode(output_ids, skip_special_tokens=True)
                return summary
        
    elif model_name == 'IlyaGusev/rut5_base_sum_gazeta':
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        input_length = 600
        model_min_length = 30
        model_max_length = 200

        def return_summary(text, min_length, max_length):
            with torch.no_grad():
                input_ids = tokenizer(
                    [text],
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )["input_ids"].to(device)
                output_ids = model.generate(
                    input_ids=input_ids,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    num_beams=num_beams
                )[0]
                summary = tokenizer.decode(output_ids, skip_special_tokens=True)
                return summary
        
    elif model_name == 'IlyaGusev/rugpt3medium_sum_gazeta':
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        input_length = 600
        model_min_length = 30
        model_max_length = 200

        def return_summary(text, min_length, max_length):
            with torch.no_grad():
                input_ids = tokenizer(
                    text,
                    max_length=max_length,
                    return_tensors="pt",
                    add_special_tokens=False, 
                    padding=False,
                    truncation=True
                )["input_ids"]
                input_ids = input_ids.tolist()[0] + [tokenizer.sep_token_id]
                input_ids = torch.LongTensor([input_ids]).to(device)
                output_ids = model.generate(
                    input_ids=input_ids,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    num_beams=num_beams
                )[0]
                summary = tokenizer.decode(output_ids, skip_special_tokens=False)
                summary = summary.split(tokenizer.sep_token)[1]
                summary = summary.split(tokenizer.eos_token)[0]
                return summary
            
    elif model_name == 'csebuetnlp/mT5_m2m_crossSum':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
        
        get_lang_id = lambda lang: tokenizer._convert_token_to_id(
            model.config.task_specific_params["langid_map"][lang][1]
        ) 

        target_lang = "russian"

        input_length = 512
        model_min_length = 30
        model_max_length = 512

        def return_summary(text, min_length, max_length):
            with torch.no_grad():
                input_ids = tokenizer(
                    [WHITESPACE_HANDLER(text)],
                    max_length=max_length,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length"
                )["input_ids"].to(device)
                output_ids = model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=get_lang_id(target_lang),
                    max_length=max_length,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    num_beams=num_beams
                )[0]
                summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return summary

    elif model_name == 'patrickvonplaten/led-large-16384-pubmed':
        # Should work with LED models
        model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = LEDTokenizer.from_pretrained(model_name)

        input_length = 16384
        model_min_length = model.config.min_length
        model_max_length = model.config.max_length

        def return_summary(text, min_length, max_length):
            return _generate_summary_led(text, model, tokenizer, device, min_length, max_length)[0]

    elif model_name == 'rooftopcoder/led-base-book-summary-samsum':
        # Should work with LED models
        model = pipeline('summarization', model=model_name, device=device)
        tokenizer = model.tokenizer

        input_length = 16384
        model_min_length = model.model.config.min_length
        model_max_length = model.model.config.max_length

        def return_summary(text, min_length, max_length):
            with torch.no_grad():
                output = model(text, min_length, max_length)
                return output[0]['summary_text'].strip()

    elif model_name in ['philschmid/flan-t5-base-samsum', 'google/flan-t5-base', 'Falconsai/medical_summarization']:
        # Should work with T5 models
        model = pipeline('summarization', model=model_name, device=device)
        tokenizer = model.tokenizer

        input_length = model.model.config.n_positions
        model_min_length = model.model.config.task_specific_params['summarization']['min_length']
        model_max_length = model.model.config.task_specific_params['summarization']['max_length']

        def return_summary(text, min_length, max_length):
            with torch.no_grad():
                output = model(text, min_length, max_length)
                return output[0]['summary_text'].strip()

    else:
        # Should work with BART models
        model = pipeline('summarization', model=model_name, device=device)
        tokenizer = model.tokenizer

        try:
            input_length = model.model.config.max_position_embeddings
            model_min_length = model.model.config.task_specific_params['summarization']['min_length']
            model_max_length = model.model.config.task_specific_params['summarization']['max_length']
        except:
            input_length = model.model.config.max_position_embeddings
            model_min_length = model.model.config.min_length
            model_max_length = model.model.config.max_length

        def return_summary(text, min_length, max_length):
            with torch.no_grad():
                output = model(text, min_length, max_length)
                return output[0]['summary_text'].strip()

    summary = ''
    for c in get_chunks(preprocess_input(text), input_length - 10, tokenizer):
        n = len(tokenizer(c)['input_ids'])
        if n > 4:
            try:
                s = return_summary(c, min(model_min_length, int(0.9 * n) - 2), min(model_max_length, n - 2))
                summary += s + ' '
            except:
                print(f'Failed to summarize! Find out the problem with PDB:')
                breakpoint()
    summary = preprocess_input(summary)
    return summary


def summarize(example, model_name, device):
    # For BART, Flan T5 models
    text = ' '.join(example['sections'])
    example['summary'] = _generate_summary(text, model_name, device)
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


def _process_sentence(sentence, language):
    ru_local_parser_path = '/data6/user/mipt_dialog_segmentation/en_core_web_sm-3.2.0'
    if language == 'russian' and os.path.exists(ru_local_parser_path):
        SYNTAX_PARSER = spacy.load(ru_local_parser_path)
    elif language == 'russian':
        SYNTAX_PARSER = spacy.load('ru_core_news_sm')
    else:
        SYNTAX_PARSER = spacy.load('en_core_web_sm')

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
    
    clauses_text = []
    for clause in sentence_clauses:
        s = clause.text.strip(' :;.,/')
        if len(s) > 10:
            clauses_text.append(s)

    return clauses_text


def calculate_prob_vector(batch):
    prob_vectors = []
    for emb, emb_sum in zip(batch['embeddings'], batch['embeddings_summary']):
        if len(emb_sum) < 2 or len(emb) == 0:
            prob_vectors.append(emb)
            continue
        emb = np.array(emb)
        emb_sum = np.array(emb_sum).T
        logits = np.matmul(emb, emb_sum)
        probs = softmax(logits, axis=1)
        prob_vectors.append(probs)

    batch['probs'] = prob_vectors
    return batch


def split_sent_batched(example, language):
    doc_summary = example['summary']
    sentences = nltk.tokenize.sent_tokenize(doc_summary, language=language)
    simple_sentences = []
    for sentence in sentences:
        splitted_sentence = _process_sentence(sentence, language)
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


def get_borders_sumseg(example, tiling_model, plot):
    try:
        gold_boundaries = example['boundaries']
    except:
        gold_boundaries = None

    if plot:
        print(example['sections'])
        print('Gold:', gold_boundaries)
        print(example['summary'])
        print(example['splitted_summary'])

    probabilities = example['probs']
    boundaries = tiling_model.transform(probabilities, gold_boundaries=gold_boundaries, plot=plot)

    if plot:
        print('Ours:', boundaries)
        if input('Continue?') == 'no':
            quit()
        print()
    return boundaries


###

def _calculate_metric_on_example(example, cfg, topictiling_parameters=None):
    tiling_model = TopicTilingModel(**topictiling_parameters)

    try:
        boundaries_pred = get_borders_sumseg(example, tiling_model, cfg.sumseg.tiling.plot)
    except:
        log('Failed to infer sumseg! Will try to use TopicTiling on embeddings then...')
        breakpoint()
        boundaries_pred = classify_borders(example['embeddings'],
                                           window_size=cfg.tiling_window_size,
                                           threshold=cfg.tiling_threshold,
                                           smoothing_passes=cfg.smoothing_passes,
                                           smoothing_window=cfg.smoothing_window)

    wd, pk, f1 = calc_metric(example, boundaries_pred)

    example['boundaries_pred'] = boundaries_pred
    example['topics_pred'] = get_topics(boundaries_pred)
    example['wd'] = wd
    example['pk'] = pk
    example['f1'] = f1

    return example


def _calculate_metrics_on_dataset(cfg, ds, topictiling_parameters=None):
    ds = ds.map(_calculate_metric_on_example, 
                keep_in_memory=True,
                num_proc=cfg.num_cpu,
                fn_kwargs={
        'cfg': cfg,
        'topictiling_parameters': topictiling_parameters
    })

    if cfg.colorize:
        colorize(ds)

    wds = ds['wd']
    pks = ds['pk']
    f1s = ds['f1']

    wd_mean = sum(wds) / len(wds)
    pk_mean = sum(pks) / len(pks)
    f1_mean = sum(f1s) / len(f1s)

    return wd_mean, pk_mean, f1_mean


def _find_parameters_topictiling(cfg, ds):
    def objective(trial):
        '''Returns mean WD metric'''
        min_window_size = max(int(get_mean_segment_length(cfg) / 10), 1)
        max_window_size = int(get_mean_segment_length(cfg))
        params = {
            'window_size': trial.suggest_int('window_size', min_window_size, max_window_size),
            'threshold': trial.suggest_float('threshold', 0.30, 0.90),
            'smoothing_passes': trial.suggest_categorical('smoothing_passes', [1, 1]),
            'smoothing_window': trial.suggest_categorical('smoothing_window', [1, min_window_size, max_window_size]),
            'n_smooth_savgol': trial.suggest_categorical('n_smooth_savgol', [1, 3]),
            'savgol_k': trial.suggest_float('savgol_k', get_savgol_k(cfg) * 0.5, get_savgol_k(cfg) * 1.5),
            'polyorder': trial.suggest_categorical('polyorder', [3])
        }
        wd_mean, _, _ = _calculate_metrics_on_dataset(cfg, ds, topictiling_parameters=params)
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
    model_name = cfg.sumseg.model_name
    device = torch.device('cuda:0' if len(cfg.gpus) else 'cpu')
    ds = ds.map(summarize, fn_kwargs={'model_name': model_name, 'device': device})

    log('Splitting summaries on simple sentences...')
    language = 'russian' if cfg.russian_language else 'english'
    ds = ds.map(split_sent_batched, num_proc=16, fn_kwargs={'language': language})
    log('Calculating embeddings for splitted summaries...')
    ds = ds.map(embed_summary, batched=True, batch_size=cfg.sbert.cpu_batch_size,
                fn_kwargs={'model': sentence_model, 'gpu_batch_size': cfg.sbert.gpu_batch_size})
    log('Calculating closeness to summaries...')
    ds = ds.map(calculate_prob_vector, batch_size=cfg.batch_size,
                num_proc=cfg.num_cpu, batched=True)
    ds = ds.with_format('numpy', columns=['embeddings', 'embeddings_summary', 'probs', 'boundaries', 'summary',
                                          'splitted_summary', 'sections', 'labels'])
    ds.save_to_disk(cfg.output_path)


def get_scores_summary(cfg):
    log(f'Starting to load or create the dataset {cfg.dataset_type}...')
    ds_test = load_from_disk(cfg.test_path)

    log('Calculating scores...')

    topictiling_parameters = {
        'window_size': cfg.sumseg.tiling.window_size,
        'threshold': cfg.sumseg.tiling.threshold,
        'smoothing_passes': cfg.sumseg.tiling.smoothing_passes,
        'smoothing_window': cfg.sumseg.tiling.smoothing_window,
        'n_smooth_savgol': cfg.sumseg.tiling.savgol.n_smooth_savgol,
        'savgol_k': cfg.sumseg.tiling.savgol.savgol_k,
        'polyorder': cfg.sumseg.tiling.savgol.polyorder
    }

    if cfg.find_topictiling_parameters:
        ds_val = load_from_disk(cfg.val_path)

        topictiling_parameters = _find_parameters_topictiling(cfg, ds_val)
        wd_mean, pk_mean, f1_mean = _calculate_metrics_on_dataset(cfg, ds_val,
                                                                  topictiling_parameters=topictiling_parameters)
        score = (2 * f1_mean + (1 - pk_mean) + (1 - wd_mean)) / 4
        output = f'''
Using TT parameters after {cfg.n_trials} trials: {topictiling_parameters}
Metrics for {cfg.val_path} VAL:
WD {wd_mean:.5f}
PK {pk_mean:.5f}
F1 {f1_mean:.5f}
SCORE {score:.5f}
        '''
        print(output)

    wd_mean, pk_mean, f1_mean = _calculate_metrics_on_dataset(cfg, ds_test,
                                                              topictiling_parameters=topictiling_parameters)
    score = (2 * f1_mean + (1 - pk_mean) + (1 - wd_mean)) / 4
    output = f'''
Using TT parameters: {topictiling_parameters}
Metrics for {cfg.test_path} TEST:
WD {wd_mean:.5f}
PK {pk_mean:.5f}
F1 {f1_mean:.5f}
SCORE {score:.5f}
    '''
    print(output)
