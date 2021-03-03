# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 01:18:27 2021

@author: REUS
"""

import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/hindienglish-corpora/Hindi_English_Truncated_Corpus.csv")
df.head()
!pip install torch==1.3.1
!pip install inltk
from inltk.inltk import setup
setup("hi")
df.head()
text = df.hindi_sentence[1]
from inltk.inltk import tokenize
tokenize(df.hindi_sentence[1],"hi")
from inltk.inltk import get_similar_sentences

get_similar_sentences(text, 3, 'hi', degree_of_aug = 0.1)
from inltk.inltk import predict_next_words

predict_next_words(text , 6, 'hi') 