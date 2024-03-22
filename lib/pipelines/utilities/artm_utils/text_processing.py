import re
import os
import unicodedata
import pickle
import pymorphy2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from collections import defaultdict
from ..general import ensure_directory_exists
import contractions


def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'words']
    for resource in resources:
        nltk.download(resource, quiet=True)


download_nltk_resources()
tag_pattern = re.compile(r'<[^<]+?>')
url_pattern = re.compile(r'http\S+')
non_alphabetic_pattern = re.compile(r'[^a-zA-Z\s]')
non_alphabetic_pattern_russian = re.compile(r'[^а-яА-Я\s]')
multiple_space = re.compile(r'\s+')
characters = re.compile(r'[^\w\s]')
characters_russian = re.compile(r'[^\w\s]', re.UNICODE)
stop_words = set(stopwords.words('english'))
stop_words_russian = set(stopwords.words("russian"))
available_words = set(words.words())
lemmatizer = WordNetLemmatizer()
lemmatizer_russian = pymorphy2.MorphAnalyzer()


def tokenize_sentence(input_sentence, russian=False):
    clean_text = tag_pattern.sub('', input_sentence)
    clean_text = url_pattern.sub('', clean_text)
    clean_text = clean_text.lower()
    clean_text = multiple_space.sub(' ', clean_text)
    clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    if not russian:
        clean_text = contractions.fix(clean_text)
        clean_text = non_alphabetic_pattern.sub('', clean_text)
        clean_text = characters.sub('', clean_text)
        tokens = word_tokenize(clean_text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if
                (token not in stop_words and token in available_words)]
    else:
        clean_text = non_alphabetic_pattern_russian.sub('', clean_text)
        clean_text = characters_russian.sub('', clean_text)
        tokens = word_tokenize(clean_text, language='russian')
        tokens = [lemmatizer_russian.parse(token)[0].normal_form for token in tokens if
                token not in stop_words_russian]
    return tokens


def tokenize_text(input_text):
    sentences = sent_tokenize(input_text)
    return list(map(tokenize_sentence, sentences))


def create_dictionary(corpus, min_freq=30):
    word_freq = defaultdict(int)
    if isinstance(corpus[0][0], list):
        tokens = [token for document in corpus for sentence in document for token in sentence]
    else:
        tokens = [token for document in corpus for token in document]

    for token in tokens:
        word_freq[token] += 1

    filtered_counts = {word: count for word, count in word_freq.items() if count >= min_freq}
    sorted_words = sorted(filtered_counts.items(), key=lambda x: x[0])
    dictionary = {word: (index + 1) for index, (word, _) in enumerate(sorted_words)}
    return dictionary


def convert_to_uci(tokenized_corpus, dictionary, output_dir, collection_name, join_sentences=True):
    sorted_words = sorted(dictionary.items(), key=lambda x: x[0])
    vocab_name = f'vocab.{collection_name}.txt'
    corpus_name = f'docword.{collection_name}.txt'
    mapping_name = f'mapping.{collection_name}.pickle'
    ensure_directory_exists(output_dir)
    with open(os.path.join(output_dir, vocab_name), 'w') as file:
        for token, _ in sorted_words:
            file.write(f"{token}\n")

    if join_sentences:
        if not isinstance(tokenized_corpus[0][0], list):
            raise ValueError('If joining sentences document should have list[list[tokens]] struct')
        for ix, doc in enumerate(tokenized_corpus):
            flatten_doc = [token for sentence in doc for token in sentence]
            tokenized_corpus[ix] = flatten_doc
    if isinstance(tokenized_corpus[0][0], list):
        sentence = False
    else:
        sentence = True
    counter = 0
    if sentence:
        mapping = [-1] * len(tokenized_corpus)
        for ix, example in enumerate(tokenized_corpus):
            if len(example) > 0:
                counter += 1
                mapping[ix] = counter
    else:
        mapping = []
        for doc in tokenized_corpus:
            mapping.append([])
            for example in doc:
                if len(example) > 0:
                    counter += 1
                    mapping[-1].append(counter)
                else:
                    mapping[-1].append(-1)

    d_qty = counter
    w_qty = len(dictionary)
    if not sentence:
        tokenized_corpus_flattened = [example for doc in tokenized_corpus for example in doc]
    else:
        tokenized_corpus_flattened = tokenized_corpus
    output = []

    counter = 0
    for doc in tokenized_corpus_flattened:
        if len(doc) == 0:
            continue
        counter += 1
        word_freq = defaultdict(int)
        for word in doc:
            if word not in dictionary:
                continue
            word_freq[word] += 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[0])
        for token, freq in sorted_words:
            output.append([counter, dictionary[token], freq])

    nnz_qty = len(output)

    with open(os.path.join(output_dir, corpus_name), 'w') as file:
        file.write(str(d_qty) + '\n')
        file.write(str(w_qty) + '\n')
        file.write(str(nnz_qty) + '\n')

        for doc_id, word_id, count in output:
            line = f"{doc_id} {word_id} {count}"
            file.write(line + '\n')

    with open(os.path.join(output_dir, mapping_name), 'wb') as file:
        pickle.dump(mapping, file)

    return mapping


def tokenize_dataset_sample(batch, russian=True):
    batch['tokens'] = []
    for doc in batch['sections']:
        batch['tokens'].append([])
        for sentence in doc:
            batch['tokens'][-1].append(tokenize_sentence(sentence, russian=russian))
    return batch
