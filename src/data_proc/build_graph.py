import logging
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from __init__ import data_dir
import pandas as pd
from collections import defaultdict

module_logger = logging.getLogger('pygat.build_graph')


def build_graph(dataset):

    word_embeddings_dim = 300
    word_vector_map = {}

    df_info = pd.read_csv(data_dir/'ids_and_labels.txt', sep='\t', header=None, names=['id', 'test_train', 'label'])
    df_data = pd.read_csv(data_dir/'clean_filtered.txt', sep='\t', header=None, names=['text'])
    df = pd.concat([df_info, df_data], axis=1)
    df.text = df.text.apply(str)
    df.sample(frac=1).reset_index(drop=True)
    train_size = len(df[df.test_train == 'train'])
    test_size = len(df[df.test_train == 'test'])

    def build_windows(text, window_size=20):
        windows = []
        words = text.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
        return windows

    t = df.text.apply(build_windows)
    windows = [sublist for l in t for sublist in l]

    word_window_freq = {}
    # Find number of non-repeated words in windows
    for window in windows:
        appeared = set()
        for window_word in window:
            # skip if we have seen it before in window
            if window_word in appeared:
                continue
            if window_word in word_window_freq:
                word_window_freq[window_word] += 1
            else:
                word_window_freq[window_word] = 1
            appeared.add(window_word)

    vocab = set()
    module_logger.info(f"Building Vocab...")
    for window in windows:
        for window_word in window:
            vocab.add(window_word)
    vocab = list(vocab)
    vocab_size = len(vocab)
    module_logger.info(f"\tVocab length: {len(vocab)}")

    def create_word_doc_list(series):
        """
        Create word_doc_list
        :param series:
        :return:
        """
        word_doc_list = {}
        for word_id, doc_words in enumerate(series):
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(word_id)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [word_id]
                appeared.add(word)
        return word_doc_list
    word_doc_list = create_word_doc_list(df.text)
    word_doc_freq = {}

    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i, word in enumerate(vocab):
        word_id_map[word] = i

    word_pair_count = {}
    module_logger.info(f"Building word_pair counts...")
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    module_logger.info(f"\Word Pairs Count: {len(word_pair_count)}")

    # pmi as weights
    row = []
    col = []
    weight = []
    num_window = len(windows)

    module_logger.info(f"Calculating PMI...")
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)


    # doc word frequency
    doc_word_freq = {}

    for doc_id, doc_words in enumerate(df.text):
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for doc_id, doc_words in enumerate(df.text):
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(doc_id) + ',' + str(j)
            freq = doc_word_freq[key]
            if doc_id < train_size:
                row.append(doc_id)
            else:
                row.append(doc_id + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(df.text) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    with open(data_dir/'adj.pkl', 'wb') as f:
        pkl.dump(adj, f)


if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    build_graph('mr')
