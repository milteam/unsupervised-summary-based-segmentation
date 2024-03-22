import numpy as np
from numpy.linalg import norm
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def cosine_similarity(vector1, vector2):
    vector1 = vector1 + 1e-9
    vector2 = vector2 + 1e-9
    dot_product = np.dot(vector1, vector2)
    norm_product = norm(vector1) * norm(vector2)
    similarity = dot_product / norm_product
    return similarity


def depth_score(timeseries, k):
    depth_scores = []
    for i in range(1, len(timeseries) - 1):
        left, right = i - 1, i + 1
        while left > 0 and timeseries[left - 1] > timeseries[left]:
            left -= 1
        while (
                right < (len(timeseries) - 1) and timeseries[right + 1] > timeseries[right]
        ):
            right += 1
        depth_scores.append(
            (timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i])
        )
    return [0] * (k + 1) + depth_scores + [0] * (k + 1)


def smooth(timeseries, n, s):
    smoothed_timeseries = timeseries[:]
    for _ in range(n):
        for index in range(len(smoothed_timeseries)):
            neighbours = smoothed_timeseries[
                         max(0, index - s): min(len(timeseries) - 1, index + s)
                         ]
            smoothed_timeseries[index] = sum(neighbours) / len(neighbours)
    return smoothed_timeseries


def sentences_similarity(first_sentence_features, second_sentence_features) -> float:
    return float(cosine_similarity(first_sentence_features, second_sentence_features))


def compute_window(timeseries, start_index, end_index, do_weighting=False):
    """given start and end index of embedding, compute pooled window value

    [window_size, 768] -> [1, 1, 768]
    """
    block = np.asarray(timeseries[start_index:end_index])
    if do_weighting:
        magnitudes = np.sum(block, axis=1)#.reshape(-1, 1)
        window = block.T @ magnitudes
        window /= block.shape[0]
    else:
        window = np.sum(block, axis=0) / len(block)
    return window


def block_comparison_score(timeseries, k):
    """
    comparison score for a gap (i)
    """
    res = []
    for i in range(k, len(timeseries) - k):
        first_window_features = compute_window(timeseries, i - k, i + 1)
        second_window_features = compute_window(timeseries, i + 1, i + k + 2)
        res.append(
            sentences_similarity(first_window_features, second_window_features)
        )

    return res


def get_local_maxima(array):
    local_maxima_indices = []
    local_maxima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(array[i])
    return local_maxima_indices, local_maxima_values


def depth_score_to_topic_change_indexes(depth_score_timeseries, topic_change_threshold=0.65):
    threshold = topic_change_threshold * max(depth_score_timeseries)

    if not depth_score_timeseries:
        return []

    local_maxima_indices, local_maxima = get_local_maxima(depth_score_timeseries)
    if not local_maxima:
        return []

    filtered_local_maxima_indices = []
    filtered_local_maxima = []

    for i, m in enumerate(local_maxima):
        if m > threshold:
            filtered_local_maxima.append(m)
            filtered_local_maxima_indices.append(local_maxima_indices[i])

    local_maxima = filtered_local_maxima
    local_maxima_indices = filtered_local_maxima_indices

    return local_maxima_indices

def classify_borders(topic_vectors, window_size=3, threshold=0.65, smoothing_passes=3, smoothing_window=3):
    if len(topic_vectors) < 5:
        return '0' * len(topic_vectors)

    while len(topic_vectors) <= 2 * window_size + 2:
        window_size = window_size // 2
        window_size = max(1, window_size)

    block_features = block_comparison_score(topic_vectors, window_size)
    if smoothing_passes >= 1 and smoothing_window >= 1:
        block_features = smooth(block_features, n=smoothing_passes, s=smoothing_window)
    depths = depth_score(block_features, window_size)    
    change_indexes = depth_score_to_topic_change_indexes(depths, topic_change_threshold=threshold)

    boundaries = []
    for ix in range(len(topic_vectors)):
        if ix in change_indexes:
            boundaries.append('1')
        else:
            boundaries.append('0')
    return ''.join(boundaries)


class TopicTilingModel:
    def __init__(self, 
                 window_size=3, 
                 threshold=0.65, 
                 smoothing_passes=3, 
                 smoothing_window=3, 
                 n_smooth_savgol=0, 
                 savgol_k=10,
                 polyorder=3,
        ):
        self.window_size = window_size
        self.threshold = threshold 
        self.smoothing_passes = smoothing_passes
        self.smoothing_window = smoothing_window
        self.n_smooth_savgol = n_smooth_savgol
        self.savgol_k = savgol_k
        self.polyorder = polyorder
        self.use_max_topic_vector = False
    
    def transform(self, vectors, gold_boundaries=None, plot=False):
        probabilities = np.array(vectors).T.copy()  # topics, examples
        
        if plot:
            plt.figure(figsize=(10,6))
            for line in probabilities:
                plt.plot(list(range(len(line))), line)
            plt.xlabel('Sentence index')
            plt.ylabel('Topic score')
            plt.savefig('example_before.jpg')
            plt.close()
        
        if self.n_smooth_savgol > 0:
            window_length = int(min(probabilities.shape[1] * self.savgol_k, probabilities.shape[1]))
            # window_length = int(min(self.savgol_k, probabilities.shape[1]))
            # window_length = int(self.savgol_k)
            
            if window_length > 1:
                polyorder = min(window_length - 1, self.polyorder)
                for _ in range(self.n_smooth_savgol):
                    probabilities = savgol_filter(
                        probabilities, 
                        window_length=window_length, 
                        polyorder=polyorder,
                        mode='nearest',
                        axis=1)
            
        if self.use_max_topic_vector:
            boundaries = '0'
            max_ids = probabilities.argmax(axis=0)
            for i in range(1, len(max_ids)):
                boundaries += '1' if max_ids[i-1] != max_ids[i] else '0'
        else:
            boundaries = classify_borders(probabilities.T, 
                                        window_size=self.window_size, 
                                        threshold=self.threshold, 
                                        smoothing_passes=self.smoothing_passes, 
                                        smoothing_window=self.smoothing_window)
        
        if plot and gold_boundaries:
            plt.figure(figsize=(10,6))
            indices = np.where(np.array(list(map(int, gold_boundaries))) == 1)
            plt.vlines(indices, ymin=probabilities.min(), ymax=probabilities.max(), colors='black')
            for line in probabilities:
                plt.plot(list(range(len(line))), line)
            plt.xlabel('Sentence index')
            plt.ylabel('Topic score')
            plt.savefig('example_after_gold.jpg')
            plt.close()
            
            plt.figure(figsize=(10,6))
            indices = np.where(np.array(list(map(int, boundaries))) == 1)
            plt.vlines(indices, ymin=probabilities.min(), ymax=probabilities.max(), colors='black')
            for line in probabilities:
                plt.plot(list(range(len(line))), line)
            plt.xlabel('Sentence index')
            plt.ylabel('Topic score')
            plt.savefig('example_after_ours.jpg')
            plt.close()
            
            if input('Continue?') == 'no':
                quit()
        
        return boundaries
