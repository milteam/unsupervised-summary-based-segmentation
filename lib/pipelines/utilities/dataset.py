from pathlib2 import Path
import codecs
import json
import os
import re
import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize, sent_tokenize

from datasets import Dataset
from .wiki_loader import WikipediaDataSet


def get_boundaries(labels):
    assert len(labels) > 1
    boundaries = ['0']
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append('1')
        else:
            boundaries.append('0')
    return ''.join(boundaries)


def get_labels(indices):
    max_length = indices[-1] + 1
    labels = []
    label = 0
    for i in range(max_length):
        labels.append(label)
        if i + 1 in indices[:-1]:
            label += 1
    return labels


def get_topics(boundaries):
    assert len(boundaries) > 1
    topics = []
    topic = 0
    for i in range(len(boundaries)):
        if boundaries[i] == '1':
            topic += 1 
        topics.append(topic)
    return topics


# Wiki:

def get_files(path):
    'Ref: https://github.com/koomri/text-segmentation'
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


class DocToDialDataset:
    def __init__(self, path):
        self.folder_path = os.path.dirname(path)
        self.dataset_path = path
        self.docs_path = os.path.join(self.folder_path, 'doc2dial_doc.json')
        self.dialogues = []
        self._parse_dialogues()

    def _parse_dialogues(self):
        with open(self.dataset_path, 'r') as file:
            dial_dict = json.load(file)
        with open(self.docs_path, 'r') as file:
            docs_dict = json.load(file)

        for domain in dial_dict['dial_data'].keys():
            for doc_name in dial_dict['dial_data'][domain].keys():
                for dial in dial_dict['dial_data'][domain][doc_name]:
                    sample = dict()
                    sample['domain'] = domain
                    sample['doc_id'] = doc_name
                    sample['dial_id'] = dial['dial_id']
                    sample['sections'] = []
                    sample['labels'] = []
                    sample['boundaries'] = ''
                    sample['split_indices'] = []
                    sample['topic_names'] = None
                    prev_sp_id = None
                    for utterance in dial['turns']:
                        sample['sections'].append(utterance['utterance'])
                        mins = []
                        maxs = []
                        for ref in utterance['references']:
                            sp_id = ref['sp_id']
                            mins.append(docs_dict['doc_data'][domain][doc_name]['spans'][sp_id]['start_sec'])
                            maxs.append(docs_dict['doc_data'][domain][doc_name]['spans'][sp_id]['end_sec'])
                        current_sp_id = (min(mins), max(maxs))
                        if prev_sp_id and current_sp_id != prev_sp_id:
                            sample['boundaries'] += '1'
                            sample['split_indices'].append(len(sample['boundaries']) - 1)
                            sample['labels'].append(sample['labels'][-1] + 1)
                        else:
                            sample['boundaries'] += '0'
                            if not sample['labels']:
                                sample['labels'].append(0)
                            else:
                                sample['labels'].append(sample['labels'][-1])
                        prev_sp_id = current_sp_id

                    self.dialogues.append(sample)
    def _get_sample(self):
        for sample in self.dialogues:
            yield sample

    def get_generator(self):
        return self._get_sample


class DialSegDataset:
    def __init__(self, path):
        self.dialogues_path = path
        self.dialogues = []
        self._parse_dialogues()


    def _parse_dialogues(self):
        with open(self.dialogues_path, 'r') as file:
            sample = None
            for line in file:
                if len(line.strip().split()) == 1:
                    if sample:
                        self.dialogues.append(sample)
                    sample = dict()
                    sample['sections'] = []
                    sample['labels'] = []
                    sample['boundaries'] = ''
                    sample['split_indices'] = []
                    sample['topic_names'] = None
                else:
                    words = line.strip().split()
                    sample['labels'].append(int(words[0]))
                    sample['sections'].append(' '.join(words[1:]))
                    if len(sample['labels']) == 1:
                        sample['boundaries'] += '0'
                    else:
                        if sample['labels'][-1] == sample['labels'][-2]:
                            sample['boundaries'] += '0'
                        else:
                            sample['boundaries'] += '1'
                            sample['split_indices'].append(len(sample['labels']) - 1)
            self.dialogues.append(sample)

    def _get_sample(self):
        for sample in self.dialogues:
            if sample['boundaries'].count('1') > 0:
                yield sample

    def get_generator(self):
        return self._get_sample


class WikiDataset:
    def __init__(self, root):
        self.dataset = WikipediaDataSet(root, None)

    def _get_sample(self):
        for i in range(self.dataset.__len__()):
            sections, targets, path = self.dataset.__getitem__(i)
            if len(sections) <= 1 or len(targets) <= 1:
                continue
            else:
                labels = get_labels(targets)
                boundaries = get_boundaries(labels)
                assert len(sections) == len(labels)
                assert len(sections) == len(boundaries)
                
                output = {'path': str(path),
                            'sections': sections,
                            'labels': labels,
                            'boundaries': boundaries,
                            'split_indices': targets,
                            'topic_names': None}
                
                yield output

    def get_generator(self):
        return self._get_sample
    
    
# AMI:

def load_json(file_path):
    with codecs.open(file_path, "r", "utf-8") as f:
        datas = json.load(f)
    print("Load {} finished, Data size:{}".format(file_path.split("/")[-1], len(datas)))
    return datas


def preprocess(text):
    # filter some noises caused by speech recognition
    def clean_data(text_):
        text_ = text_.replace('<vocalsound>', '')
        text_ = text_.replace('<disfmarker>', '')
        text_ = text_.replace('a_m_i_', 'ami')
        text_ = text_.replace('l_c_d_', 'lcd')
        text_ = text_.replace('p_m_s', 'pms')
        text_ = text_.replace('t_v_', 'tv')
        text_ = text_.replace('<pause>', '')
        text_ = text_.replace('<nonvocalsound>', '')
        text_ = text_.replace('<gap>', '')
        return text_
    
    text = clean_data(text)
    
    fillers = ["um", "uh", "oh", "hmm", "you know", "like"]
    fillers += [filler + " " for filler in fillers]  # filler inside caption with other words
    fillers = [re.compile(f"(?i){filler}") for filler in fillers]  # make it case-insensitive

    for filler in fillers:
        text = filler.sub("", text)

    # captions_with_multiple_sentences = text.count(".")
    # if captions_with_multiple_sentences > 0:
    #     print(f"WARNING: Found {captions_with_multiple_sentences} captions with "
    #           "multiple sentences; sentence embeddings may be inaccurate.", file=sys.stderr)

    if len(text) <= 20:
        return None
    
    text = text.strip()
    text = ' '.join(text.split())

    return text


class QMSumDataset:
    def __init__(self, path):
        self.path = path
        self.dialogues = []
        self._parse_dialogues()

    def _parse_dialogues(self):
        raw_docs = []
        with open(self.path, 'r') as file:
            for line in file:
                raw_docs.append(json.loads(line))
        for doc in raw_docs:
            id_to_topic = dict()
            topic_to_label = dict()
            for topic in doc['topic_list']:
                topic_to_label[topic['topic']] = len(topic_to_label) + 1
                for text_span_range in topic['relevant_text_span']:
                    if len(text_span_range) == 1:
                        id_to_topic[int(text_span_range[0])] = topic['topic']
                        continue
                    for ix in range(int(text_span_range[0]), int(text_span_range[1]) + 1):
                        id_to_topic[ix] = topic['topic']

            sample = dict()
            sample['sections'] = []
            sample['labels'] = []
            sample['boundaries'] = ''
            sample['split_indices'] = []
            sample['topic_names'] = []

            for ix, span in enumerate(doc['meeting_transcripts']):
                cleaned_text = preprocess(span['content'])
                if not cleaned_text:
                    continue
                sample['sections'].append(cleaned_text)

                if ix in id_to_topic:
                    topic = id_to_topic[ix]
                    label = topic_to_label[topic]
                else:
                    topic = 'No Topic'
                    label = 0

                sample['labels'].append(label)
                sample['topic_names'].append(topic)
                if len(sample['sections']) == 1 or sample['labels'][-2] == sample['labels'][-1]:
                    sample['boundaries'] += '0'
                else:
                    sample['boundaries'] += '1'
                    sample['split_indices'].append(ix)

            self.dialogues.append(sample)

    def _get_sample(self):
        for sample in self.dialogues:
            yield sample

    def get_generator(self):
        return self._get_sample
    
    
class AMIDataset:
    def __init__(self, root):
        '''Set topic_key='id' if you want to set each section to different topic_id, and 
        set topic_key='topic' if you want to preserve topic names (which will imply that topics can be
        the same for different section, so it can 'come back').
        Set high_granularity to True if you want to use subtopics as topics, which makes lots of boundaries.
        '''
        self.textfiles = get_files(root)
        self.topic_key = 'id'
        self.high_granularity = False

    def _get_sections(self, segments):
        sections = []  # text in each section
        labels = []  # topic id for each section
        topic_ids = {}  # topic name : topic id

        for segment in segments:
            topic_name = segment[self.topic_key]

            if segment['dialogueacts'] != 'None':
                dialogue = []
                starttimes = []
                for d in segment['dialogueacts']:
                    if d['starttime'] not in starttimes:
                        starttimes.append(d['starttime'])
                        preprocessed = preprocess(d['text'])
                        if preprocessed is not None:
                            dialogue.append(preprocessed)
                
                if len(dialogue) > 1:
                    if topic_name not in topic_ids.keys():
                        topic_ids[topic_name] = len(topic_ids)
                    topic_id = topic_ids[topic_name]
                    
                    sections += dialogue
                    labels += [topic_id] * len(dialogue)
                
            if segment['subtopics'] != 'None':
                for subsegment in segment['subtopics']:
                    subtopic_name = topic_name
                    if self.high_granularity:
                        subtopic_name += ': ' + subsegment[self.topic_key]
                        
                    if subsegment['dialogueacts'] != 'None':
                        subdialogue = []
                        for d in subsegment['dialogueacts']:
                            preprocessed = preprocess(d['text'])
                            if preprocessed is not None:
                                subdialogue.append(preprocessed)
                        
                        if len(subdialogue) > 1:
                            if subtopic_name not in topic_ids.keys():
                                topic_ids[subtopic_name] = len(topic_ids)
                            subtopic_id = topic_ids[subtopic_name]
                        
                            sections += subdialogue
                            labels += [subtopic_id] * len(subdialogue)

        return sections, labels, topic_ids

    def _get_sample(self):
        for path in self.textfiles:
            with open(path, 'r') as f:
                segments = json.load(f)
            
            sections, labels, topic_ids = self._get_sections(segments)
            if len(labels) <= 1:
                continue

            boundaries = get_boundaries(labels)
            assert len(sections) == len(labels)
            assert len(sections) == len(boundaries)
            
            targets = [index for index, value in enumerate(list(map(int, boundaries))) if value == 1] + [len(boundaries)]
            
            # Sort topic_ids' keys by values:
            topic_names = sorted(topic_ids, key=topic_ids.get)
            
            yield {'path': str(path),
                   'sections': sections,
                   'labels': labels,
                   'boundaries': boundaries,
                   'split_indices': targets,
                   'topic_names': topic_names}

    def get_generator(self):
        return self._get_sample


def load_dataset_by(cfg):
    print(f'Loading dataset from file: {cfg.input_path}')
    if cfg.dataset_type == 'wiki':
        generator = WikiDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type == 'ami':
        generator = AMIDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type in ['dialseg', 'superdialseg', 'tiage']:
        generator = DialSegDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type == 'doc2dial':
        generator = DocToDialDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type == 'qmsum':
        generator = QMSumDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type == 'sber':
        generator = SberDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type == 'wikisection':
        generator = WikiSectionDataset(cfg.input_path).get_generator()
    else:
        raise ValueError(f'No such dataset type {cfg.dataset_type} exist!')
    
    ds = Dataset.from_generator(generator)
    if cfg.sample_size is not None:
        ds = Dataset.from_dict(ds[:cfg.sample_size])
    return ds


def calculate_statistics(ds, verbose=True):
    def mean(array):
        if len(array):
            return sum(array) / len(array)
        else:
            raise ValueError('Failed to calculate mean metric, check dataset!')
    
    doc_utterances = ds['sections']
    doc_texts = [' '.join(text) for text in doc_utterances]
    
    # create sections
    sections = []
    for example in ds:
        section = ''
        utterances = example['sections']
        boundaries = example['boundaries']
        for i in range(len(boundaries)):
            if boundaries[i] == '0':
                section += ' ' + utterances[i]
            else:
                sections.append(section.strip())
                section = utterances[i]
    
    # avg # sections in doc
    boundaries  = ds['boundaries']
    avg_segment_length = mean([len(b) / (b.count('1') + 1) for b in boundaries])

    # avg # words in section
    avg_n_words_in_section = mean([len(word_tokenize(section)) for section in sections])
    
    # min / max / avg # words in doc
    n_words_in_doc = [len(word_tokenize(text)) for text in doc_texts]
    avg_n_words_in_doc = mean(n_words_in_doc)
    min_n_words_in_doc = min(n_words_in_doc)
    max_n_words_in_doc = max(n_words_in_doc)
    
    # avg # utterances in doc
    avg_n_utterances_in_doc = mean([len(doc_utterance) for doc_utterance in doc_utterances])
    
    if verbose:
        print(f'# docs: {len(ds)}')
        print(f'min / avg / max # words in doc: {min_n_words_in_doc:.1f} / {avg_n_words_in_doc:.1f} / {max_n_words_in_doc:.1f}')
        print(f'avg # words in section: {avg_n_words_in_section:.1f}')
        print(f'avg # utterances in doc: {avg_n_utterances_in_doc:.1f}')
        print(f'avg # utterances in section: {avg_segment_length:.1f}')
    
    return avg_segment_length


def get_mean_segment_length(cfg):
    if cfg.dataset_type == 'wiki':
        mean_segment_length = cfg.mean_segment_length_wiki
    elif cfg.dataset_type == 'ami':
        mean_segment_length = cfg.mean_segment_length_ami
    elif cfg.dataset_type == 'dialseg':
        mean_segment_length = cfg.mean_segment_length_dialseg
    elif cfg.dataset_type == 'tiage':
        mean_segment_length = cfg.mean_segment_length_tiage
    elif cfg.dataset_type == 'superdialseg':
        mean_segment_length = cfg.mean_segment_length_superdialseg
    elif cfg.dataset_type == 'qmsum':
        mean_segment_length = cfg.mean_segment_length_qmsum
    elif cfg.dataset_type == 'doc2dial':
        mean_segment_length = cfg.mean_segment_length_doc2dial
    elif cfg.dataset_type == 'sber':
        mean_segment_length = cfg.mean_segment_length_sber
    elif cfg.dataset_type == 'wikisection':
        mean_segment_length = cfg.mean_segment_length_wikisection
    else:
        raise ValueError(f'No such dataset type {cfg.dataset_type} exist!')
    return mean_segment_length


def get_savgol_k(cfg):
    savgol_k = 2
    return savgol_k / get_mean_segment_length(cfg)

# Local dataset:

class SberDataset:    
    def __init__(self, path):
        self.path = path
        df = pd.read_csv(self.path, index_col=0)
        self.dataset_dict = self.get_dict(df)
    
    def remove_tags(self, phrase):
        """Do small data preprocessing
        """
        return phrase.replace('@', '').replace('#', '')
    
    def transform_label(self, label_list, encoded_labels, x):
        """ Transforms label's name to it's encoded type
            in label's dictionary
        """
        for i in range(len(label_list)):
            if x == label_list[i]:
                return encoded_labels[i]
            else:
                continue
        
    def get_boundaries(self, df, separator = ''):        
        """Forms boundaries in dialogs in format of our
           preprocessing classes
        """
        i=0 
        boundary = [] # boundary in one dialog
        all_boundaries = [] 
        for i in range(len(self.df['phrase_num'])):
            try:
                if int(self.df['phrase_num'][i]) <= int(self.df['phrase_num'][i+1]):
                    boundary.append(self.df['segment_start'][i])
                else:
                    boundary.append(self.df['segment_start'][i])
                    all_boundaries.append(''.join(str(x) for x in boundary))
                    boundary = []
            except KeyError:
                boundary.append(self.df['segment_start'][i])
                all_boundaries.append(''.join(str(x) for x in boundary))
                boundary = []    
        return all_boundaries
    
    
    def get_sections(self, df, separator=''):
        """Gets full dialogs in the format:
            1) All sentences are on a new line.
            2) Each sentence is separated by a comma.
            3) Each document packed in a list and all_sections is also a nested list.
        """
        section = []
        all_sections = []
        for i in range(len(df['phrase_num'])):
            try:
                if int(df['phrase_num'][i]) <= int(df['phrase_num'][i+1]):
                    section.append(self.df['clean_phrase'][i])
                else:
                    section.append(self.df['clean_phrase'][i])
                    all_sections.append(section)
                    section = []
            except KeyError:
                section.append(self.df['clean_phrase'][i])
                all_sections.append(section)
                section = []
        return all_sections

    
    def get_labels(self, df):
        """Gets labels of sentences in dialogs in format of our
           preprocessing classes
        """
        label_list = df['type'].unique()
        encoder = LabelEncoder()
        encoded_list = encoder.fit_transform(label_list)
        i=0 
        label = [] # label in one dialog
        all_labels = [] 
        for i in range(len(self.df['phrase_num'])):
            try:
                if int(self.df['phrase_num'][i]) <= int(self.df['phrase_num'][i+1]):
                    label.append(self.transform_label(label_list,
                                                      encoded_list,
                                                      self.df['type'][i]))
                else:
                    label.append(self.transform_label(label_list,
                                                      encoded_list,
                                                      self.df['type'][i]))
                    all_labels.append(label)
                    label = []
            except KeyError:
                label.append(self.transform_label(label_list,
                                                      encoded_list,
                                                      self.df['type'][i]))
                all_labels.append(label)
                label = []    
        return all_labels
    
    def get_topic_names(self, df):
        """Gets all topic names with their encoded ones"""
        label_list = df['type'].unique()
        encoder = LabelEncoder()
        encoded_list = encoder.fit_transform(label_list)
        topic_lst = []
        for i in range(len(label_list)):
            topic_lst.append(f'{label_list[i]}, {encoded_list[i].item()}')    
        return topic_lst
     
    def get_dict(self, df):
        """Starts the entire class process
        """
        sections = self.get_sections(df)
        boundaries = self.get_boundaries(df)
        labels = self.get_labels(df)
#         topic_names = self.get_topic_names(df)
        sections = [[self.remove_tags(sentence) for sentence in section] for section in sections]
#         result = [''.join(section) for section in sections]
        format_dict = {
            'sections': sections,
            'boundaries': boundaries,
            'labels': labels
        }
        return format_dict
    
    def _get_sample(self):
        sec_list = self.dataset_dict['sections']
        bnd_list = self.dataset_dict['boundaries']
        lbl_list = self.dataset_dict['labels']
        for dlg_section, dlg_boundaries, dlg_labels in zip(sec_list, bnd_list, lbl_list):
            yield {'sections': dlg_section,
                   'labels': dlg_labels,
                   'boundaries': dlg_boundaries}

    def get_generator(self):
        return self._get_sample


def colorize(ds):
    def create_class_colors(sentence_classes, seed=123):
        random.seed =seed
        
        class_colors = {}
        for sentence_class in set(sentence_classes):
            r = lambda: random.randint(0, 255)
            color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
            class_colors[sentence_class] = '\033[38;2;{};{};{}m'.format(
                *tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            )
        return class_colors      
            
    def print_colored_text(sentences, sentence_classes, class_colors):
        print('Classes:', *sentence_classes)
        for i, sentence in enumerate(sentences):
            sentence_class = sentence_classes[i]
            color = class_colors.get(sentence_class, '')
            print(color + sentence + '\033[0m', end=' ')
        print('\n')
            
    for i, data in enumerate(ds):
        # get original segmentation:
        # print(i, data['path'])
        print(i)
        # print(data['topic_names'])
        sentences = data['sections']
        sentence_classes = data['labels']
        print(f'N of sections in document: {len(set(sentence_classes))}')
        print(f'N of simple sentences in summary: {len(data["splitted_summary"])}')
        
        # print('Original:')
        class_colors = create_class_colors(sentence_classes)
        # print_colored_text(sentences, sentence_classes, class_colors)
        
        if 'summary' in ds.column_names:
            print(f'Summary: {data["summary"]}')
            print(f'Summary split: {data["splitted_summary"]}')
            print()
        
        print('Predicted:')
        topics = data['topics_pred']
        class_colors = create_class_colors(topics)
        print_colored_text(sentences, topics, class_colors)
        
        # input('Enter to continue, Ctrl-C to exit:')
        
        
class WikiSectionDataset:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def parse_document(document):
        sections = []
        labels = []
        boundaries = ''
        targets = []
        topic_names = []
        prev_topic = None
        label = 0
        
        for annotation in document['annotations']:
            s_ix = annotation['begin']
            e_ix = s_ix + annotation['length']
            section = document['text'][s_ix:e_ix]
            topic = annotation['sectionLabel']
            sentences = sent_tokenize(section, language="english")
            for ix, sentence in enumerate(sentences):
                if ix == 0:
                    if prev_topic is not None and prev_topic != topic:
                        label += 1
                        topic_names.append(topic)
                        targets.append(len(sections))
                        sections.append(sentence)
                        labels.append(label)
                        boundaries += '1'
                        prev_topic = topic
                        continue
                    elif prev_topic is None:
                        prev_topic = topic
                        topic_names.append(topic)
    
                sections.append(sentence)
                labels.append(label)
                boundaries += '0'
    
        output = {'path': 'path',
                  'id': document['id'],
                  'sections': sections,
                  'labels': labels,
                  'boundaries': boundaries,
                  'split_indices': targets,
                  'topic_names': topic_names}
        
        return output
                    

    def _get_sample(self):
        with open(self.path) as file:
            documents = json.load(file)
            for document in documents:
                parsed_document = self.parse_document(document)
                if len(parsed_document['sections']) > 1 and parsed_document['boundaries'].count('1') > 0:
                    yield parsed_document

    
    def get_generator(self):
        return self._get_sample
