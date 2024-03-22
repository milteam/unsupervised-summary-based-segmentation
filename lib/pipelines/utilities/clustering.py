import pickle
from pathlib import Path

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.cluster import BaseCluster
from bertopic.representation import MaximalMarginalRelevance
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utilities.general import get_sentence_encoder, log

try:
    from cuml.common.device_selection import set_global_device_type
    from cuml.manifold import UMAP as UMAP_cuml
    from cuml.cluster import HDBSCAN as HDBSCAN_cuml
    from cuml.cluster.hdbscan.prediction import membership_vector as membership_vector_cuml
    from cuml.cluster.hdbscan.prediction import approximate_predict as approximate_predict_cuml
except ImportError as e:
    print(f'No libraries cuml found with error: {e}')

from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.prediction import membership_vector, approximate_predict


def get_umap_model(cfg):
    args = {
        'n_components': cfg.umap.n_components, 
        'random_state': cfg.random_state,
        'n_neighbors': cfg.umap.n_neighbors,
        'min_dist': cfg.umap.min_dist, 
        'metric': cfg.umap.metric
    }
    if cfg.cuml.use_cuml:
        set_global_device_type(cfg.cuml.device)
        umap_model = UMAP_cuml(**args)
    else:
        umap_model = UMAP(**args)
    return umap_model
            
            
def get_hdbscan_model(cfg):
    args = {
        'min_cluster_size': cfg.hdbscan.min_cluster_size, 
        'min_samples': cfg.hdbscan.min_samples,
        'prediction_data': True,
        'gen_min_span_tree': True,
    }
    if cfg.cuml.use_cuml:
        set_global_device_type(cfg.cuml.device)
        hdbscan_model = HDBSCAN_cuml(**args)
    else:
        hdbscan_model = HDBSCAN(**args)
    
    return hdbscan_model


def get_bertopic_model(cfg):
    empty_reduction_model = BaseDimensionalityReduction()
    empty_cluster_model = BaseCluster()

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, 
                                        bm25_weighting=True)
    representation_model = MaximalMarginalRelevance(diversity=cfg.bertopic.MaximalMarginalRelevance.diversity)
    stop_words_dict = None
    stop_words = 'english' if not cfg.russian_language else stop_words_dict
    vectorizer_model = CountVectorizer(stop_words=stop_words, 
                                        ngram_range=(1, cfg.bertopic.CountVectorizer.ngrams), 
                                        token_pattern=r'\b[^\d\W][^\d\W]+\b',
                                        min_df=cfg.bertopic.CountVectorizer.min_df, 
                                        max_df=cfg.bertopic.CountVectorizer.max_df)
    sentence_model = get_sentence_encoder(cfg)

    topic_model = BERTopic(umap_model=empty_reduction_model, 
                            hdbscan_model=empty_cluster_model, 
                            embedding_model=sentence_model, 
                            vectorizer_model=vectorizer_model,
                            ctfidf_model=ctfidf_model,
                            calculate_probabilities=True,
                            representation_model=representation_model,
                            verbose=cfg.bertopic.verbose)
    return topic_model


class ClusteringModel:
    def __init__(self, cfg):    
        self.cfg = cfg    
        self.umap_model = get_umap_model(cfg)
        self.hdbscan_model = get_hdbscan_model(cfg)
        self.batch_size = cfg.umap.batch_size
        
    def fit_umap(self, embeddings):
        self.umap_model.fit(embeddings)
        
    def transform_umap(self, embeddings):
        k = len(embeddings) // self.batch_size
        indices = [self.batch_size * i for i in range(1, k + 1)]
        batches = np.split(embeddings, indices)

        embeddings_umap = []
        for batch in batches:
            if batch.shape[0] > 0:
                embeddings_umap.append(self.umap_model.transform(batch))
        embeddings_umap = np.concatenate(embeddings_umap)
        return embeddings_umap
    
    def plot_umap(self, embeddings):
        fig = px.scatter(*embeddings[:, :2].T)
        fig.show()
    
    def fit_hdbscan(self, embeddings):
        self.hdbscan_model.fit(embeddings)
        
    def transform_hdbscan(self, embeddings):
        if self.cfg.cuml.use_cuml:
            probabilities = membership_vector_cuml(
                self.hdbscan_model,
                embeddings,
                batch_size=min(4096, len(embeddings))) 
        else:
            probabilities = membership_vector(
                self.hdbscan_model,
                embeddings)
        return probabilities
    
    def plot_hdbscan(self, embeddings_umaped):
        color_palette = sns.color_palette('Paired', 
                                          self.hdbscan_model.n_clusters_)
        cluster_colors = [color_palette[x] if x >= 0
                        else (0.5, 0.5, 0.5)
                        for x in self.hdbscan_model.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                zip(cluster_colors, 
                                    self.hdbscan_model.probabilities_)]
        plt.scatter(*embeddings_umaped[:, :2].T, 
                    s=50, 
                    linewidth=0, 
                    c=cluster_member_colors, 
                    alpha=0.1)

class BERTopicModel:
    def __init__(self, cfg, log_status=False, checkpoints_path=None):
        self.cfg = cfg
        self.clustering_model = ClusteringModel(cfg)
        self.reduce_outliers = cfg.bertopic.reduce_outliers
        self.topic_model = get_bertopic_model(cfg)
        self.log_status = log_status
        self.checkpoints_path = checkpoints_path
        self.n_skip = cfg.bertopic.n_skip
        
    def fit(self, text, embeddings):
        '''Fit topic model and reduce outliers'''
        embeddings_part = embeddings[::self.n_skip, :]
        
        if self.log_status:
            log(f'BERTopic: fit umap on {embeddings_part.shape[0]} examples...')
        self.clustering_model.fit_umap(embeddings_part)
        
        if self.checkpoints_path is not None:
            self.save(self.checkpoints_path)
        
        embeddings_part_umap = self.clustering_model.transform_umap(embeddings_part)
        embeddings_umap = self.clustering_model.transform_umap(embeddings)
        
        if self.log_status:
            log(f'BERTopic: fit hdbscan on {embeddings_part_umap.shape[0]} examples...')
        self.clustering_model.fit_hdbscan(embeddings_part_umap)
        
        if self.checkpoints_path is not None:
            self.save(self.checkpoints_path)
        
        if self.cfg.cuml.use_cuml:
            labels, strengths = approximate_predict_cuml(self.clustering_model.hdbscan_model, 
                                                embeddings_umap)
        else:
            labels, strengths = approximate_predict(self.clustering_model.hdbscan_model, 
                                                embeddings_umap)
        
        topics, probs = self.topic_model.fit_transform(text, 
                                                       embeddings_umap, 
                                                       y=labels)
        
        if self.cfg.cuml.use_cuml:
            probabilities = membership_vector_cuml(self.clustering_model.hdbscan_model, 
                                  embeddings_umap, 
                                  batch_size=min(4096, len(embeddings_umap))) 
        else:
            probabilities = membership_vector(self.clustering_model.hdbscan_model, 
                                  embeddings_umap) 
        
        if self.log_status:
            log('BERTopic: reduce outliers')
            print('Before reducing outliers:')
            print(self.get_topic_info())
            # self.get_topic_info().to_excel('topics_before_reducing.xlsx')
        if self.reduce_outliers:
            try:
                new_topics = self.topic_model.reduce_outliers(
                    documents=text,
                    topics=topics,
                    probabilities=probabilities,
                    strategy="probabilities")
                self.topic_model.update_topics(
                    text,
                    topics=new_topics,
                    vectorizer_model=self.topic_model.vectorizer_model,
                    ctfidf_model=self.topic_model.ctfidf_model,
                    representation_model=self.topic_model.representation_model)
            except:
                if self.log_status:
                    log('Can\'t reduce outliers')
                    
        if self.log_status:
            print('After reducing outliers:')
            print(self.get_topic_info())
            # self.get_topic_info().to_excel('topics_after_reducing.xlsx')
        
        if self.checkpoints_path is not None:
            self.save(self.checkpoints_path)
        
    def transform(self, text, embeddings):
        '''Get topic vectors'''
        embeddings_umap = self.clustering_model.transform_umap(embeddings)
        if self.cfg.cuml.use_cuml:
            probabilities = membership_vector_cuml(
            self.clustering_model.hdbscan_model,
            embeddings_umap,
            batch_size=min(4096, len(embeddings_umap))) 
        else:
            probabilities = membership_vector(
            self.clustering_model.hdbscan_model,
            embeddings_umap) 
            
        return probabilities
    
    def get_topics(self, probabilities):
        all_topics = self.topic_model.get_topics()
        topic_ids = probabilities.argmax(axis=1)
        return topic_ids, all_topics
    
    def get_topic_info(self):
        return self.topic_model.get_topic_info()
    
    def save(self, dir):
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.cfg, open(dir / 'cfg.pickle', 'wb'))
        pickle.dump(self.clustering_model, open(dir / 'clustering_model.pickle', 'wb'))
        pickle.dump(self.topic_model, open(dir / 'topic_model.pickle', 'wb'))

    def load(self, dir):
        dir = Path(dir)
        try:
            self.cfg = pickle.load(open(dir / 'cfg.pickle', 'rb'))
            self.clustering_model = pickle.load(open(dir / 'clustering_model.pickle', 'rb'))
            self.topic_model = pickle.load(open(dir / 'topic_model.pickle', 'rb'))
            
            self.reduce_outliers = self.cfg.bertopic.reduce_outliers
            self.n_skip = self.cfg.bertopic.n_skip
        except:
            log(f'Error while loading BERTopic model! Check {dir}')
