# coding: utf-8

# # Feature Extraction for Deep Learning (WORD2VEC)

# * Input:
#   * ./data/ICD9CODES.p (pickle file of all ICD9 codes)
#   * ./data/ICD9CODES_TOP10.p (pickle file of top 10 ICD9 codes)
#   * ./data/ICD9CODES_TOP50.p (pickle file of top 50 ICD9 codes)
#   * ./data/ICD9CAT_TOP10.p (pickle file of top 10 ICD9 categories)
#   * ./data/ICD9CAT_TOP50.p (pickle file of top 50 ICD9 categories)
#   * ./data/TRAIN-VAL-TEST-HADMID.p (pickle file of train-val-test sets. each set contains a list of hadm_id)
#   * ./data/DATA_HADM_CLEANED.csv (contains top 50 icd9code, top 50 icd9cat, and cleaned clinical text w/out stopwords for each admission, source for seqv1)
#   * ./data/model_word2vec_v2_*dim.txt (our custom word2vec model)
#   * ./data/bio_nlp_vec/PubMed-shuffle-win-*.txt (pre-trained bionlp word2vec model. convert from .bin to .txt using gensim)
# * Output:
#   * ./data/DATA_WORDSEQV[0/1]_HADM_TOP[10/10CAT/50/50CAT].p (pickle file of train-val-test data and label)
#   * ./data/DATA_WORDSEQV[0/1]_WORDINDEX.p (pickle of file of word sequence index)
# * Description:
#   * All sequential feature extraction tried in the paper.
#   * WORDSEQV0 = seqv0 in the paper
#   * WORDSEQV1 = seqv1 in the paper
#   * word2vec_*dim = custom word2vec with * features in the paper
#   * PubMed-shuffle-win-*.txt = pre trained word2vec in the paper (bio*)

# ## Initialization

# In[1]:

import pandas as pd
import numpy as np
import pickle

ICD9CODES = pickle.load(open("./data/ICD9CODES.p", "rb"))
ICD9CODES_TOP10 = pickle.load(open("./data/ICD9CODES_TOP10.p", "rb"))
ICD9CODES_TOP50 = pickle.load(open("./data/ICD9CODES_TOP50.p", "rb"))

# In[4]:

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
print(len(stopwords.words('english')))


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import re

STOPWORDS_WORD2VEC = stopwords.words('english') + ICD9CODES


def preprocessor_word2vec(text):
    text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = re.sub(" \d+", " ", text)

    return text


def create_WORD2VEC_DL(df, max_sequence_len=600, inputCol='text'):
    texts = df[inputCol].apply(preprocessor_word2vec)


    toke = Tokenizer()
    toke.fit_on_texts(texts)
    sequence = toke.texts_to_sequences(texts)

    ave_seq = [len(i) for i in sequence]
    print(1.0 * sum(ave_seq) / len(ave_seq))

    word_index = toke.word_index
    reverse_word_index = dict(zip(word_index.values(), word_index.keys()))  # dict e.g. {1:'the', 2:'a' ...}
    # index_list = word_index.values()

    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequence, maxlen=max_sequence_len)

    return data, word_index, reverse_word_index


def create_EmbeddingMatrix_V0(word_index, word2vec_model_path, remove_stopwords=True):
    embeddings_index = {}
    f = open(word2vec_model_path)
    for line in f:
        values = line.split()
        # print(values)
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    if remove_stopwords:
        keys_updated = [word for word in embeddings_index.keys() if word not in STOPWORDS_WORD2VEC]
        index2word_set = set(keys_updated)
    else:
        index2word_set = set(embeddings_index.keys())

    EMBEDDING_DIM = list(embeddings_index.values())[1].size  # dimensions of the word2vec model
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in index2word_set:
            embedding_matrix[i] = embeddings_index.get(word)

    return embedding_matrix


import random, pickle


def separate(seed, N):
    idx = list(range(N))
    random.seed(seed)
    random.shuffle(idx)
    idx_train = idx[0:int(N * 0.50)]
    idx_val = idx[int(N * 0.50):int(N * 0.75)]
    idx_test = idx[int(N * 0.75):N]

    return idx_train, idx_val, idx_test


def separate_2(df, hadmid_pickle):
    f = open(hadmid_pickle, 'rb')
    hadmid_train = pickle.load(f)
    hadmid_val = pickle.load(f)
    hadmid_test = pickle.load(f)
    f.close()

    df2 = df.copy()
    df2['_idx'] = df2.index
    df2.set_index('id', inplace=True)

    idx_train = df2.loc[hadmid_train]['_idx'].tolist()
    idx_val = df2.loc[hadmid_val]['_idx'].tolist()
    idx_test = df2.loc[hadmid_test]['_idx'].tolist()

    return idx_train, idx_val, idx_test


def batch_output_pickle(df, data, reversemap, fname, labels, hadmid_pickle='./data/TRAIN-VAL-TEST-HADMID.p'):
    idx_tuple = separate_2(df, hadmid_pickle)

    f = open(fname, 'wb')
    pickle.dump(reversemap, f, protocol=pickle.HIGHEST_PROTOCOL)
    for i in idx_tuple:
        pickle.dump(data[i], f, protocol=pickle.HIGHEST_PROTOCOL)
    for i in idx_tuple:
        pickle.dump(df.loc[i][labels].values, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def output_pickle(obj, fname):
    f = open(fname, 'wb')
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def main():

    df = pd.read_csv("./data/DATA_HADM_CLEANED.csv", escapechar='\\',index_col=0)
    print(df.head())

    data, word_index, reverse_word_index = create_WORD2VEC_DL(df.copy(), max_sequence_len=1500)
    output_pickle(word_index, "./data/DATA_WORDSEQV1_WORDINDEX.p")
    batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV1_HADM_TOP10.p", ICD9CODES_TOP10)
    batch_output_pickle(df, data, reverse_word_index, "./data/DATA_WORDSEQV1_HADM_TOP50.p", ICD9CODES_TOP50)

    em = create_EmbeddingMatrix_V0(word_index, "./data/model_word2vec_v2_300dim.txt", remove_stopwords=True)
    output_pickle(em, "./data/EMBMATRIXV1_WORD2VEC_v2_300dim.p")

def test():
    import pickle

    datafile = "./data/DATA_WORDSEQV1_HADM_TOP10.p"
    f = open(datafile, 'rb')
    loaded_data = []
    for i in range(7):  # [reverse_dictionary, train_sequence, test_sequence, train_label, test_label]:
        loaded_data.append(pickle.load(f))
    f.close()

    dictionary = loaded_data[0]
    train_sequence = loaded_data[1]
    val_sequence = loaded_data[2]
    test_sequence = loaded_data[3]
    train_label = loaded_data[4]
    val_label = loaded_data[5]
    test_label = loaded_data[6]

    print(train_sequence[:5, :])
    print(train_label[:5, :])

if __name__ =="__main__":
    main()


