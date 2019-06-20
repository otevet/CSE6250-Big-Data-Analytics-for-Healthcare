import numpy as np
import bcolz
import pickle

def pre_trained_word_vector(embedding_size):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='./data/6B.300.dat', mode='w')

    with open('./data/glove.6B.300d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, embedding_size)), rootdir='./data/6B.300.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open('./data/6B.300_words.pkl', 'wb'))
    pickle.dump(word2idx, open('./data/6B.300_idx.pkl', 'wb'))

def load_glove(target_vocab,emb_dim):
    vectors = bcolz.open('./data/6B.300.dat')[:]
    words = pickle.load(open('./data/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open('./data/6B.300_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(target_vocab)
    #print(matrix_len)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    return weights_matrix


#pre_trained_word_vector()