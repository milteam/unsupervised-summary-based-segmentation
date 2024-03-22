'''
Run clustering one by one, which means you train a model on each document separately.
'''

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import optuna
import numpy as np
from datasets import load_from_disk
from utilities.dataset import get_mean_segment_length, load_dataset_by, get_savgol_k
from utilities.general import log, embed, ensure_directory_exists, calc_metric, get_sentence_encoder
from utilities.tiling import TopicTilingModel, classify_borders
from utilities.clustering import BERTopicModel
from tqdm import tqdm


def _calculate_metric_on_example(cfg, example, topictiling_parameters=None):
    if topictiling_parameters is None:
        tiling_model = TopicTilingModel(window_size=cfg.bertopic.tiling.window_size, 
                                        threshold=cfg.bertopic.tiling.threshold, 
                                        smoothing_passes=cfg.bertopic.tiling.smoothing_passes, 
                                        smoothing_window=cfg.bertopic.tiling.smoothing_window,
                                        n_smooth_savgol=cfg.bertopic.tiling.savgol.n_smooth_savgol, 
                                        savgol_k=get_savgol_k(cfg),
                                        polyorder=cfg.bertopic.tiling.savgol.polyorder)
    else:
        tiling_model = TopicTilingModel(**topictiling_parameters)
    
    def get_models():
        topic_model = BERTopicModel(cfg)
        return topic_model, tiling_model
    
    if cfg.clustering:
        try:
            boundary = _get_borders_clustering(example, get_models, cfg.bertopic.tiling.plot)
        except:
            log('Failed to predict using clustering model!')
            boundary = classify_borders(example['embeddings'], 
                                        window_size=cfg.tiling_window_size,
                                        threshold=cfg.tiling_threshold, 
                                        smoothing_passes=cfg.smoothing_passes,
                                        smoothing_window=cfg.smoothing_window)
    else:
        boundary = classify_borders(example['embeddings'], 
                                    **topictiling_parameters)
        
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
    ds_part = ds#.select(list(range(0, len(ds) // 10)))
    
    def objective(trial):
        '''Returns mean PK metric'''
        min_window_size = max(int(get_mean_segment_length(cfg) / 10), 1)
        max_window_size = int(get_mean_segment_length(cfg))
        if cfg.clustering:
            params = {
                'window_size': trial.suggest_int('window_size', 1, max_window_size), 
                'threshold': trial.suggest_float('threshold', 0.1, 0.9), 
                'smoothing_passes': trial.suggest_categorical('smoothing_passes', [0]), 
                'smoothing_window': trial.suggest_categorical('smoothing_window', [0]),
                'n_smooth_savgol': trial.suggest_int('n_smooth_savgol', 3, 3),
                'savgol_k': trial.suggest_categorical('savgol_k', [get_savgol_k(cfg)]),
                'polyorder': trial.suggest_categorical('polyorder', [3])
            }
        else:
            params = {
                'window_size': trial.suggest_int('window_size', min_window_size, max_window_size), 
                'threshold': trial.suggest_float('threshold', 0.10, 0.90), 
                'smoothing_passes': trial.suggest_categorical('smoothing_passes', [0, 1]),
                'smoothing_window': trial.suggest_categorical('smoothing_window', [1, min_window_size, max_window_size]),
            }
        _, wd_mean, pk_mean, f1_mean = _calculate_metrics_on_dataset(cfg, ds_part, topictiling_parameters=params)
        # return -(2*f1_mean+(1-pk_mean)+(1-wd_mean))/4
        return wd_mean
    
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=cfg.n_trials)
    return study.best_params


def _get_borders_clustering(example, models_gen, plot):
    topic_model, tiling_model = models_gen()
    
    text = example['sections']
    embeddings = np.array(example['embeddings'])

    topic_model.fit(text, embeddings)
    probabilities = topic_model.transform(text, embeddings)
    
    boundaries = tiling_model.transform(probabilities, gold_boundaries=example['boundaries'], plot=plot)
    
    # topic_ids, all_topics = topic_model.get_topics(probabilities)
    # assert len(topic_ids) == len(embeddings)
    
    return boundaries


def embed_and_save(cfg):
    log(f'Starting to load or create the dataset {cfg.dataset_type}...')
    ds = load_dataset_by(cfg)
    log('Dataset generator is built. Loading a sentence-bert model...')
    sentence_model = get_sentence_encoder(cfg)
    log('Calculating embeddings for the dataset...')
    ds_embedded = ds.map(embed, batched=True, batch_size=cfg.sbert.cpu_batch_size,
                         fn_kwargs={'model': sentence_model, 'gpu_batch_size': cfg.sbert.gpu_batch_size})
    ensure_directory_exists(cfg.output_path)
    ds_embedded.save_to_disk(cfg.output_path)
    

def get_scores_sentence_bert(cfg):
    log(f'Loading the dataset {cfg.dataset_type} from disk...')
    ds_val = load_from_disk(cfg.val_path)#.select(list(range(10)))
    ds_test = load_from_disk(cfg.test_path)#.select(list(range(10)))
            
    log('Making a segmentation...')
    
    if cfg.clustering:
        topictiling_parameters = {
            'window_size': cfg.bertopic.tiling.window_size, 
            'threshold': cfg.bertopic.tiling.threshold, 
            'smoothing_passes': cfg.bertopic.tiling.smoothing_passes,
            'smoothing_window': cfg.bertopic.tiling.smoothing_window,
            'n_smooth_savgol': cfg.bertopic.tiling.savgol.n_smooth_savgol,
            'polyorder': cfg.bertopic.tiling.savgol.polyorder,
            'savgol_k': get_savgol_k(cfg)
        }
    else:
        topictiling_parameters = {
            'window_size': cfg.tiling_window_size, 
            'threshold': cfg.tiling_threshold, 
            'smoothing_passes': cfg.smoothing_passes,
            'smoothing_window': cfg.smoothing_window
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
