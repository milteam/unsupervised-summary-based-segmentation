'''
Run clustering by training on train part of a dataset, and then get results on test part of the dataset.
'''

from statistics import mode
import os
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import optuna
import numpy as np
from datasets import load_from_disk
from utilities.dataset import get_mean_segment_length, load_dataset_by, get_savgol_k
from utilities.general import log, calc_metric
from utilities.tiling import TopicTilingModel, classify_borders
from utilities.clustering import BERTopicModel
from tqdm import tqdm


def calc_probabilities(example, topic_model):
    text = example['sections']
    embeddings = np.array(example['embeddings'])
    probabilities = topic_model.transform(text, embeddings)
    example['probabilities'] = np.array(probabilities)
    return example


def _get_borders_clustering(example, topic_model, tiling_model, plot):
    probabilities = example['probabilities']
    boundaries = tiling_model.transform(probabilities, gold_boundaries=example['boundaries'], plot=plot)
    
    print_topics = False
    if print_topics:
        topic_ids, all_topics = topic_model.get_topics(probabilities)
        
        sentences_by_segment = []
        topics_by_segment = []
        boundary_indices = [0] + list(np.where(np.array(list(map(int, boundaries))) == 1)[0]) + [len(boundaries)]
        for i in range(len(boundary_indices)-1):
            sentences = example['sections'][boundary_indices[i]:boundary_indices[i+1]]
            topic_ids_for_section = topic_ids[boundary_indices[i]:boundary_indices[i+1]]
            most_popular_topic_id = mode(topic_ids_for_section)
            topic = all_topics[most_popular_topic_id]
            
            sentences_by_segment.append(sentences)
            topics_by_segment.append(topic)
        print(sentences_by_segment)
        print(topics_by_segment)

    return boundaries


def _calculate_metric_on_example(cfg, example, topic_model, topictiling_parameters=None):
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
    
    try:
        boundary = _get_borders_clustering(example, topic_model, tiling_model, cfg.bertopic.tiling.plot)
    except:
        log('Failed to predict using clustering model!')
        boundary = classify_borders(example['embeddings'], 
                                    window_size=cfg.tiling_window_size,
                                    threshold=cfg.tiling_threshold, 
                                    smoothing_passes=cfg.smoothing_passes,
                                    smoothing_window=cfg.smoothing_window)
        
    wd, pk, f1 = calc_metric(example, boundary)
    return boundary, wd, pk, f1


def _calculate_metrics_on_dataset(cfg, ds, topic_model, topictiling_parameters=None):
    wds = []
    pks = []
    f1s = []
    boundaries = []
    wd_mean, pk_mean, f1_mean = None, None, None
    
    t = tqdm(ds)
    i = 0
    for example in t:
        if i > 0:
            wd_mean = sum(wds) / len(wds)
            pk_mean = sum(pks) / len(pks)
            f1_mean = sum(f1s) / len(f1s)
            
            description = f'wd: {wd_mean:.3f}, '
            description += f'pk: {pk_mean:.3f}, '
            description += f'f1: {f1_mean:.3f}'
            t.set_description(description)
        i+=1
        
        boundary, wd, pk, f1 = _calculate_metric_on_example(cfg, example, topic_model, topictiling_parameters=topictiling_parameters)

        boundaries.append(boundary)
        wds.append(wd)
        pks.append(pk)
        f1s.append(f1)
        
    wd_mean = sum(wds) / len(wds)
    pk_mean = sum(pks) / len(pks)
    f1_mean = sum(f1s) / len(f1s)
    
    return boundaries, wd_mean, pk_mean, f1_mean


def _find_parameters_topictiling(cfg, ds, topic_model):
    def objective(trial):
        '''Returns mean WD metric'''
        min_window_size = max(int(get_mean_segment_length(cfg) / 10), 1)
        max_window_size = int(get_mean_segment_length(cfg))
        params = {
            'window_size': trial.suggest_int('window_size', min_window_size, max_window_size), 
            'threshold': trial.suggest_float('threshold', 0.40, 0.90),
            'smoothing_passes': trial.suggest_categorical('smoothing_passes', [1, 1]),
            'smoothing_window': trial.suggest_categorical('smoothing_window', [1, min_window_size, max_window_size]),
            'n_smooth_savgol': trial.suggest_categorical('n_smooth_savgol', [1, 3]), 
            'savgol_k': trial.suggest_float('savgol_k', get_savgol_k(cfg) * 0.5, get_savgol_k(cfg) * 1.5),
            'polyorder': trial.suggest_categorical('polyorder', [3])
        }
        _, wd_mean, _, _ = _calculate_metrics_on_dataset(cfg, ds, topic_model, topictiling_parameters=params)
        return wd_mean
    
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=cfg.n_trials)
    return study.best_params


def get_scores_bert_topic(cfg):
    log(f'Working with dataset {cfg.train_path}')
    
    bt_path = cfg.bertopic.pickle_path + '_' + cfg.dataset_type
    topic_model = BERTopicModel(cfg, log_status=True, checkpoints_path=bt_path)
    if bt_path is not None and os.path.exists(bt_path):
        log(f'Loading BERTopic from a checkpoint {bt_path}...')
        topic_model.load(bt_path)
    else:
        log('No BERTopic checkpoints found.')
        
        log(f'Loading train dataset {cfg.dataset_type} from disk...')
        ds_train = load_from_disk(cfg.train_path) \
            .with_format('numpy', columns=['embeddings', 'sections', 'boundaries'])#.select(list(range(50000)))
        
        log('Preparing train text and embeddings...')
        embeddings_train = []
        text_train = []
        n_leave = cfg.bertopic.n_leave
        all_embeddings = ds_train['embeddings']
        all_text = ds_train['sections']
        for i in tqdm(range(len(all_embeddings))):
            n = all_embeddings[i].shape[0]
            step = n // n_leave if n_leave else 0
            indices_to_select = list(range(0, n, step + 1))
            embeddings_train.append(all_embeddings[i][indices_to_select, :])
            text_train.append(all_text[i][indices_to_select])
        embeddings_train = np.concatenate(embeddings_train)
        text_train = np.concatenate(text_train)
        log('Training BERTopic...')
        topic_model.fit(text_train, embeddings_train)
        if bt_path is not None:
            log(f'Saving BERTopic to {bt_path}...')
        
        del embeddings_train
        del text_train
        
    print(topic_model.get_topic_info())
            
    log(f'Loading val and test datasets {cfg.dataset_type} from disk...')
    ds_val = load_from_disk(cfg.val_path)#.select(range(10))
    ds_test = load_from_disk(cfg.test_path)#.select(range(10))
    
    ds_val = ds_val.map(calc_probabilities, fn_kwargs={'topic_model': topic_model})
    ds_test = ds_test.map(calc_probabilities, fn_kwargs={'topic_model': topic_model})
    
    topictiling_parameters = {
            'window_size': cfg.bertopic.tiling.window_size, 
            'threshold': cfg.bertopic.tiling.threshold, 
            'smoothing_passes': cfg.bertopic.tiling.smoothing_passes,
            'smoothing_window': cfg.bertopic.tiling.smoothing_window,
            'n_smooth_savgol': cfg.bertopic.tiling.savgol.n_smooth_savgol,
            'polyorder': cfg.bertopic.tiling.savgol.polyorder,
            'savgol_k': get_savgol_k(cfg)
    }
            
    if cfg.find_topictiling_parameters:
        topictiling_parameters = _find_parameters_topictiling(cfg, ds_val, topic_model)
        _, wd_mean, pk_mean, f1_mean = _calculate_metrics_on_dataset(cfg, ds_val, topic_model, topictiling_parameters=topictiling_parameters)
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
        
    _, wd_mean, pk_mean, f1_mean = _calculate_metrics_on_dataset(cfg, ds_test, topic_model, topictiling_parameters=topictiling_parameters)
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
        