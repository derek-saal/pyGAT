import logging

import nltk
from nltk.corpus import stopwords

from __init__ import data_dir
from src.data_proc.utils import clean_str

module_logger = logging.getLogger('pygat.remove_words')
try:
    nltk.data.find('corpora/stopwords/english')
except LookupError:
    nltk.download('english')
stop_words = set(stopwords.words('english'))
module_logger.info(stop_words)

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
# vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])


def remove_words():
    raw_corpus_list = []

    # Read raw data
    with open(data_dir / 'raw_corpus.txt', 'rb') as f:
        module_logger.info(f"Reading raw data from {f.name}")

        for line in f.readlines():
            raw_corpus_list.append(line.strip().decode('latin1'))
        f.close()
    word_freq = {}  # to remove rare words

    module_logger.info("Cleaning strings and building word_freq")
    for raw_doc in raw_corpus_list:
        temp = clean_str(raw_doc)
        words = temp.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    clean_corpus_list = []

    module_logger.info("Filtering low_count words and stop_words")
    for raw_doc in raw_corpus_list:
        temp = clean_str(raw_doc)
        words = temp.split()
        doc_words = []
        for word in words:
            # and word in word_vector_map
            if word not in stop_words and word_freq[word] >= 5:  # word not in stop_words and word_freq[word] >= 5
                doc_words.append(word)
        doc_str = ' '.join(doc_words).strip()
        clean_corpus_list.append(doc_str)

    clean_corpus_str = '\n'.join(clean_corpus_list)
    with open(data_dir / 'clean_filtered.txt', 'w') as f:
        module_logger.info(f"Writing cleaned and filtered text to {f.name}")
        f.write(clean_corpus_str)

    min_len = 1000000 # Large number for min line condition
    aver_len = 0
    max_len = 0
    line_count = 0
    with open(data_dir / 'clean_filtered.txt', 'r') as f:
        module_logger.info(f"Calculating document min_len, max_len, and average_len for cleaned and filtered corpus...")
        for i, line in enumerate(f.readlines()):
            line_len = len(line)
            if line_len == 0:
                module_logger.warning(f"Line number {i} has a len of 0")
                # continue
            line = line.strip()
            temp = line.split()
            aver_len = aver_len + len(temp)  # aver_len is actually total_words
            min_len = min(line_len, min_len)
            max_len = max(line_len, max_len)
            line_count += 1
    aver_len = aver_len / line_count
    module_logger.info('\tmin_len : ' + str(min_len))
    module_logger.info('\tmax_len : ' + str(max_len))
    module_logger.info('\taverage_len : ' + str(aver_len))
    module_logger.info('\tTotal Lines : ' + str(line_count))


if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    remove_words()
