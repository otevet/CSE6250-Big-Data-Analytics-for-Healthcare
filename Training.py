# import libraries
import fastai
from fastai import *
from fastai.text import * 
import pandas as pd
import os
from sklearn.metrics import classification_report

SAVE_MODELS = False  #if you want to save model files
SAMPLE = 1   # fraction of the dataset you want to use for training
DATAFRAME_SAVE_FILENAME = os.path.join(os.getcwd(),'df_sample.csv')
ENCODER_SAVE_FILENAME = 'fine_tuned_enc_test'   # what is the name to save the encoder
DATA_HADM_FILE = os.path.join(os.getcwd(),'Data','sampled_DATA_HADM_CLEANED.csv')   # directory of the preprocessed data
BATCH_SIZE = 16
path=''

# loading and preprocessing of the csv

df = pd.read_csv(DATA_HADM_FILE)
c = df.filter(like='c').columns.tolist()
df.drop(c+['id'], axis=1,inplace=True)
if SAMPLE < 1:
    df = df.sample(frac = SAMPLE)

df.to_csv(DATAFRAME_SAVE_FILENAME,index=False)
    
lab = df.columns.tolist()[0:-1]
# Language model data
data_lm = TextLMDataBunch.from_csv(path, DATAFRAME_SAVE_FILENAME,label_cols=lab, text_cols=['text'])
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.fit_one_cycle(1, 1e-2,moms=(0.8,0.7))
learn.unfreeze()


learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save_encoder(ENCODER_SAVE_FILENAME)


# Classifier model data
data_clas = TextClasDataBunch.from_csv(path, DATAFRAME_SAVE_FILENAME, vocab=data_lm.train_ds.vocab, bs=BATCH_SIZE,label_cols=lab, text_cols=['text'])

learn = text_classifier_learner(data_clas,AWD_LSTM, drop_mult=0.5)
learn.load_encoder(ENCODER_SAVE_FILENAME)
learn.freeze()

# training the first layer
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
if SAVE_MODELS:
    learn.save('first')

# training the first and second layer

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
if SAVE_MODELS:
    learn.save('second')

# Training first to third layer  

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
if SAVE_MODELS:
    learn.save('third')

# Training the entire model    
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

if SAVE_MODELS:
    learn.save('fourth')
    
y_pred, y_true =  learn.get_preds()
y_true = y_true.numpy()
scores = y_pred.numpy()
metrics = classification_report(y_true, scores>0.35, target_names=data_clas.valid_ds.classes)
print(metrics)