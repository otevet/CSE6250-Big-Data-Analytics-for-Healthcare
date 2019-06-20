import pandas as pd
import pickle

import random

def load_data(hadm_file):
    df_hadm = pd.read_csv(hadm_file, escapechar='\\')
    return df_hadm

def separate(seed, N):
    idx=list(range(N))
    random.seed(seed)
    random.shuffle(idx)
    idx_train= idx[0:int(N*0.50)]
    idx_val= idx[int(N*0.50):int(N*0.75)]
    idx_test= idx[int(N*0.75):N]

    return idx_train, idx_val, idx_test

def output_pickle(train,val,test, fname):
    f = open(fname, 'wb')
    pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def main():

    df_hadm = load_data("./data/DATA_HADM_CLEANED.csv")
    idx_train, idx_val, idx_test = separate(1234, df_hadm.shape[0])

    #print(idx_train)
    #print(idx_val)
    #print(idx_test)
    output_pickle(idx_train,idx_val,idx_test,"./data/TRAIN-VAL-TEST-HADMID.p")

if __name__ =="__main__":
    main()