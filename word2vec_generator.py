
#### Dependecies

#scipy = 0.17.0
#gensim = 1.0.1
#pandas = 0.19.2
#numpy = 1.11.2

#### I used DATA_HADM.csv created using feature_extraction_nonseq.ipynb


import pandas as pd
import pickle
import random
import re
from gensim.models import Word2Vec
from gensim import utils
from time import time

class WordVecGenerator(object):
    def load_data(self,hadm_file,icd9code_file):

        df_hadm = pd.read_csv(hadm_file, escapechar='\\')
        icd9_codes = pickle.load(open(icd9code_file, "rb"))
        return df_hadm,icd9_codes

    def separate(self,seed, N):
        idx=list(range(N))
        random.seed(seed)
        random.shuffle(idx)
        idx_train= idx[0:int(N*0.50)]
        idx_val= idx[int(N*0.50):int(N*0.75)]
        idx_test= idx[int(N*0.75):N]

        return idx_train, idx_val, idx_test
    # Cleanning the data
    # Light preprocesing done on purpose (so word2vec understand sentence structure)

    def preprocessor(self,text):
        text = re.sub('<[^>]*>', '', text)
        text = re.sub('[\W]+', ' ', text.lower())
        text = text.split()
        return text




    # Apply word2vec
    # assumptions: window is 5 words left and right, eliminate words than dont occur in
    # more than 10 docs, use 4 workers for a quadcore machine. Size is the size of vector
    # negative=5 implies negative sampling and makes doc2vec faster to train
    # sg=0 means CBOW architecture used. sg=1 means skip-gram is used
    # model = Word2Vec(sentence, size=100, window=5, workers=4, min_count=5)
    def word2Vec_generate(self,vectorSize,token_review):


        #instantiate our  model
        model_w2v = Word2Vec(min_count=10, window=5, size=vectorSize, sample=1e-3, negative=5, workers=4, sg=0)

        #build vocab over all reviews
        model_w2v.build_vocab(token_review)

        #We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
        Idx=list(range(len(token_review)))

        t0 = time()
        for epoch in range(5):
            random.shuffle(Idx)
            perm_sentences = [token_review[i] for i in Idx]
            #model_w2v.train(perm_sentences)
            model_w2v.train(perm_sentences, total_examples = model_w2v.corpus_count, epochs = model_w2v.epochs )
            print(epoch)

        elapsed=time() - t0
        print("Time taken for Word2vec training: ", elapsed, "seconds.")


        # saves the word2vec model to be used later.
        #model_w2v.save('./model_word2vec_skipgram_300dim')
        model_w2v.save('./data/model_word2vec_%ddim' % vectorSize)

        # open a saved word2vec model
        #import gensim
        model_w2v=Word2Vec.load('./data/model_word2vec_%ddim' % vectorSize)

        # save the model in txt format
        model_w2v.wv.save_word2vec_format('./data/model_word2vec_v2_%ddim.txt' % vectorSize, binary=False)

    def output_pickle(self,train, val, test, fname):
        f = open(fname, 'wb')
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

def main():
    generator = WordVecGenerator()
    df_hadm,icd9_codes = generator.load_data("./data/DATA_HADM_CLEANED.csv","./data/ICD9CODES.p")

    idx_train, idx_val, idx_test = generator.separate(1234, df_hadm.shape[0])
    generator.output_pickle(idx_train, idx_val, idx_test, "./data/TRAIN-VAL-TEST-HADMID.p")

    idx_join_train = idx_train + idx_val

    df_hadm_w2v = df_hadm.iloc[idx_join_train].copy()
    token_review = list(df_hadm_w2v['text'].apply(generator.preprocessor))

    #change to 100 and 600 to generate vectors with those dimensions
    generator.word2Vec_generate(300,token_review)

if __name__ =="__main__":
    main()