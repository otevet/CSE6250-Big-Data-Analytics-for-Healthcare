# CSE6250_project_team2  
Natural Language Processing on MIMIC III dataset - CSE6250 Class project, Spring 2019, Team 2.   Authors: Neal Cheng, Joshua Sherfey, Oren Tevat, Kevin Zhu  

**Environment setup:**

Python version
python 3.6.5

Instructions:
data files and saved model can be found on google drive: https://drive.google.com/file/d/1OI7W6SV_RgsSSvAtoB2M8O2-7rGvXKvM/view?usp=sharing  
ULMFit saved model can be download from google drive: https://drive.google.com/file/d/1TAiG7nK7jThWDhA_b9VrDSSIkWyQiFwn/view?usp=sharing

**Preprocessing Environment:**

pip install --upgrade pip;

pip install keras;

pip install gensim;

pip install -U nltk;

pip install pyspark;

pip install fastai;

**Training Environment:**

conda create -n fastai python=3.6;

conda activate fastai;

conda install -c pytorch -c fastai fastai=1.0.51;

conda install pandas;

**1. to run preprocess flow**

python preprocess.py 

OR 

python preprocess_reduced.py

**2. to generate word2vec model**

python word2vec_generator.py

**3. Train ULMFit model:**

python Training.py
