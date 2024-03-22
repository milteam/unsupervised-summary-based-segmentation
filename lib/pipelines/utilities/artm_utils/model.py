import os
import pickle
import artm
import numpy as np

from ..general import log


def build_artm_dataset(collection_name, input_dir, batches_dir):
    vocab_name = f'vocab.{collection_name}.txt'
    log('Starting to build ARTM batch vectorizer. It may take few minutes')
    artm.BatchVectorizer(data_path=input_dir, data_format='bow_uci',
                         collection_name=collection_name, target_folder=batches_dir)
    log('Dataset saved in ARTM binaries. Starting assembling dictionary')
    dictionary = artm.Dictionary()
    dictionary.gather(data_path=batches_dir,
                      vocab_file_path=os.path.join(input_dir, vocab_name))

    dictionary.save(dictionary_path=os.path.join(batches_dir, 'dictionary'))
    log('Dictionary assembled and saved')


def load_vectorizer_and_dict(batches_path):
    log('Loading batch vectorizer')
    batch_vectorizer = artm.BatchVectorizer(data_path=batches_path,
                                            data_format='batches')
    log('Loaded batch vectorizer. Loading dictionary')
    my_dictionary = artm.Dictionary()
    my_dictionary.load(dictionary_path=os.path.join(batches_path, 'dictionary.dict'))
    log('Dictionary loaded')
    return batch_vectorizer, my_dictionary


def train_model(vectorizer, dictionary, model_path=None, topics_qty=50, tokens_qty=10,
                sp_phi=-1, sp_pheta=-0.5, dec_phi=1e5, collection_passes=50):
    model = artm.ARTM(num_topics=topics_qty, dictionary=dictionary)
    model.num_tokens = tokens_qty
    model.scores.add(artm.PerplexityScore(name='perplexity_score',
                                          dictionary=dictionary))
    model.scores.add(artm.TopTokensScore(name='top_tokens_score'))

    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_reg'))
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_reg'))
    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_reg'))
    model.regularizers['sparse_phi_reg'].tau = sp_phi
    model.regularizers['sparse_theta_reg'].tau = sp_pheta
    model.regularizers['decorrelator_phi_reg'].tau = dec_phi

    log('Model initialized. Starting train')
    for ix in range(collection_passes):
        model.fit_offline(batch_vectorizer=vectorizer, num_collection_passes=1)
        log(f'Processed batch number {ix + 1} from {collection_passes}.'
            f' Perplexity score is {int(model.score_tracker["perplexity_score"].last_value)}')

    log('Training finished. Dumping model')
    saved_top_tokens = model.score_tracker['top_tokens_score'].last_tokens
    if model_path is not None:
        model.dump_artm_model(model_path)
    with open(os.path.join(model_path, 'topic_tokens.pickle'), 'wb') as file:
        pickle.dump(saved_top_tokens, file)

    return model


def topic_probs_prediction(model, vectorizer, doc_id_mapping):
    log('Calculating topic probs')
    topic_probs = model.transform(vectorizer)
    log('Topic probs calculated')
    num_topics = topic_probs.shape[0]
    columns = topic_probs.columns.values.tolist()
    columns = sorted(columns)
    columns_set = set(columns)
    log('Combining predicts')
    predicts = []
    if isinstance(doc_id_mapping[0], list):
        for sentences in doc_id_mapping:
            predicts.append([])
            for doc_id in sentences:
                if doc_id in columns_set:
                    predicts[-1].append(topic_probs[doc_id].to_numpy())
                else:
                    try:
                        predicts[-1].append(predicts[-1][-1])
                    except IndexError:
                        predicts[-1].append(np.zeros(num_topics))
            predicts[-1] = np.array(predicts[-1])
    else:
        for doc_id in doc_id_mapping:
            if doc_id in columns_set:
                predicts.append(topic_probs[doc_id].to_numpy())
            else:
                predicts.append(np.zeros(num_topics))
    log('Predicts combined')

    return predicts


def load_model(model_dir, dict_path):
    log('Loading model')
    dictionary = artm.Dictionary()
    dictionary.load(dictionary_path=dict_path)
    with open(os.path.join(model_dir, 'topic_tokens.pickle'), 'rb') as file:
        topic_tokens = pickle.load(file)
    topics_qty = len(topic_tokens)
    model = artm.ARTM(num_topics=topics_qty, dictionary=dictionary)
    model.load(os.path.join(model_dir, 'n_wt.bin'))
    model.load(os.path.join(model_dir, 'p_wt.bin'))
    log('Model loaded')
    return model
