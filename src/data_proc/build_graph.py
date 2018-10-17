import logging
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from __init__ import data_dir
import pandas as pd

module_logger = logging.getLogger('pygat.build_graph')


def build_graph(dataset):

    word_embeddings_dim = 300
    word_vector_map = {}

    # shuffling
    corpus_name_list = []
    corpus_train_list = []
    corpus_test_list = []

    df_info = pd.read_csv(data_dir/'ids_and_labels.txt', sep='\t', header=None, names=['id', 'test_train', 'label'])
    df_data = pd.read_csv(data_dir/'clean_filtered.txt', sep='\t', header=None, names=['text'])
    df = pd.concat([df_info, df_data], axis=1)
    with open(data_dir/'ids_and_labels.txt', 'r') as f:
        module_logger.info(f"Reading {f.name} for {dataset}")
        for line in f.readlines():
            assert len(line) > 0, f"Empty Line"
            corpus_name_list.append(line.strip())
            id, test_train, label = line.split("\t")
            if test_train == 'test':
                corpus_test_list.append(line.strip())
            elif test_train == 'train':
                corpus_train_list.append(line.strip())
            else:
                raise ValueError(f"Line (printed below) did not contain test or train\n{line}")

    clean_corpus_list = []

    with open(data_dir/'clean_filtered.txt', 'r') as f:
        module_logger.info(f"Reading {f.name} for {dataset}")
        for line in f.readlines():
            clean_corpus_list.append(line.strip())

    def build_id_list(corpus_list, id_type):
        """
        Will take a corpus list and perform:
            shuffling
            file writing
        :param corpus_list:
        :param id_type:
        :return:
        """
        ids = []
        module_logger.info(f"Building {id_type}ing ID list...")
        for name in corpus_list:
            # list.index finds first index of arg
            train_id = corpus_name_list.index(name)
            ids.append(train_id)
        random.shuffle(ids)

        ids_str = '\n'.join(str(index) for index in ids)
        with open(data_dir/f'{id_type}_ids.txt', 'w') as f:
            module_logger.info(f"Writing {id_type}ing ID list to {f.name}")
            f.write(ids_str)
        module_logger.info(f"Total number of {id_type}ing IDs: {len(ids)}")
        return ids

    train_ids = build_id_list(corpus_train_list, 'train')
    test_ids = build_id_list(corpus_test_list, 'test')

    ids = train_ids + test_ids
    module_logger.info(f"Total number of IDs: {len(ids)}")

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    module_logger.info(f"Shuffling names and corpus")
    for id in ids:
        shuffle_doc_name_list.append(corpus_name_list[int(id)])
        shuffle_doc_words_list.append(clean_corpus_list[int(id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    with open(data_dir/'name_shuffle.txt', 'w') as f:
        module_logger.info(f"Writing shuffled names to: {f.name}")
        f.write(shuffle_doc_name_str)

    with open(data_dir/'doc_words_shuffle.txt', 'w') as f:
        module_logger.info(f"Writing shuffled corpus to: {f.name}")
        f.write(shuffle_doc_words_str)

    # build vocab
    word_freq = {}
    module_logger.info(f"Building vocab")
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_freq.keys())
    vocab_size = len(vocab)

    word_doc_list = {}

    for i, shuffle_doc_word in enumerate(shuffle_doc_words_list):
        words = shuffle_doc_word.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    with open(data_dir/'vocab.txt', 'w') as f:
        module_logger.info(f"Writing vocab to: {f.name}")
        f.write(vocab_str)

    '''
    Word definitions begin
    '''
    '''
    definitions = []

    for word in vocab:
        word = word.strip()
        synsets = wn.synsets(clean_str(word))
        word_defs = []
        for synset in synsets:
            syn_def = synset.definition()
            word_defs.append(syn_def)
        word_des = ' '.join(word_defs)
        if word_des == '':
            word_des = '<PAD>'
        definitions.append(word_des)

    string = '\n'.join(definitions)


    f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
    f.write(string)
    f.close()

    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vec.fit_transform(definitions)
    tfidf_matrix_array = tfidf_matrix.toarray()
    print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

    word_vectors = []

    for i in range(len(vocab)):
        word = vocab[i]
        vector = tfidf_matrix_array[i]
        str_vector = []
        for j in range(len(vector)):
            str_vector.append(str(vector[j]))
        temp = ' '.join(str_vector)
        word_vector = word + ' ' + temp
        word_vectors.append(word_vector)

    string = '\n'.join(word_vectors)

    f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
    f.write(string)
    f.close()

    word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
    _, embd, word_vector_map = loadWord2Vec(word_vector_file)
    word_embeddings_dim = len(embd[0])
    '''

    '''
    Word definitions end
    '''

    # label list
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    label_list_str = '\n'.join(label_list)
    with open(data_dir/'labels.txt', 'w') as f:
        f.write(label_list_str)

    # x: feature vectors of training docs, no initial features
    # slect 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    # different training rates

    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    with open(data_dir/'real_train_name.txt', 'w') as f:
        f.write(real_train_doc_names_str)

    row_x = []
    col_x = []
    data_x = []
    y = []
    for i in range(real_train_size):
        # Calculating x
        doc_vec = np.zeros(word_embeddings_dim)
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        if not doc_len:
            continue

        assert doc_len > 0, "Doc_len is 0"
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

        # Calculating y
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)

    # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))

    y = np.array(y)

    # tx: feature vectors of test docs, no initial features
    test_size = len(test_ids)

    row_tx = []
    col_tx = []
    data_tx = []
    ty = []
    for i in range(test_size):
        # Calculating tx
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        if not doc_len:
            continue

        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

        # Calculating ty
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for _ in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))

    ty = np.array(ty)

    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words

    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size, word_embeddings_dim))

    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    row_allx = []
    col_allx = []
    data_allx = []
    ally = []

    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        if doc_len == 0:
            continue
        assert doc_len > 0, "Doc_len is 0"
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len

        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for _ in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))


    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))


    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)

    module_logger.info(f"\nx: {x.shape}\ty: {y.shape}\ntx: {tx.shape}\tty: {ty.shape}\nallx: {allx.shape}\tally: {ally.shape}")

    '''
    Doc word heterogeneous graph
    '''

    # word co-occurence with context windows
    window_size = 20
    windows = []

    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)


    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
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

    row = []
    col = []
    weight = []

    # pmi as weights

    num_window = len(windows)

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

    # word vector cosine similarity as weights

    '''
    for i in range(vocab_size):
        for j in range(vocab_size):
            if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
                vector_i = np.array(word_vector_map[vocab[i]])
                vector_j = np.array(word_vector_map[vocab[j]])
                similarity = 1.0 - cosine(vector_i, vector_j)
                if similarity > 0.9:
                    print(vocab[i], vocab[j], similarity)
                    row.append(train_size + i)
                    col.append(train_size + j)
                    weight.append(similarity)
    '''
    # doc word frequency
    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    # dump objects
    f = open(data_dir/"ind_x.pkl".format(dataset), 'wb')
    pkl.dump(x, f)
    f.close()

    f = open(data_dir/"ind_y.pkl".format(dataset), 'wb')
    pkl.dump(y, f)
    f.close()

    f = open(data_dir/"ind_tx.pkl".format(dataset), 'wb')
    pkl.dump(tx, f)
    f.close()

    f = open(data_dir/"ind_ty.pkl".format(dataset), 'wb')
    pkl.dump(ty, f)
    f.close()

    f = open(data_dir/"ind_allx.pkl".format(dataset), 'wb')
    pkl.dump(allx, f)
    f.close()

    f = open(data_dir/"ind_ally.pkl".format(dataset), 'wb')
    pkl.dump(ally, f)
    f.close()

    f = open(data_dir/"ind_adj.pkl".format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()


if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    build_graph('mr')
