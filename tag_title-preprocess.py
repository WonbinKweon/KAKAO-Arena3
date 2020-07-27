#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from data import load, load_tmp
from tqdm import tqdm

import json

import os
from scipy import sparse
from collections import Counter
import re

import pickle


# In[14]:


base_path = './data_kakao/'
train_playId_set,public_playId_set,final_playId_set,train_songId_set,public_songId_set,final_songId_set,playId2songIds,playId2tags,playId2tagIds,playId2title, songId2playIds,tag2playIds,tag2tagId,tagId2tag,songId2albumId,songId2artistIds,songId2name,songId2gnrs,songId2dtlgnrs,songId2gnrIds,songId2date,songId2year,songId2month, playId_w_tags, playId2updt, songId2artists = load_tmp(base_path)


# In[15]:


from khaiii import KhaiiiApi
api = KhaiiiApi()

playId2tag_morph = dict()

for playId in playId2tags:
    title = playId2tags[playId]
    title = ' '.join(title)
    words = []
    if title == '\u3000\u3000\u3000\u3000\u3000\u3000\u3000\u3000' or len(title) == 0 or len(title.strip()) ==0: title = '무제'
    
    for word in api.analyze(title):
        for morph in word.morphs:
            tmp = str(morph.lex +"/" + morph.tag)
            words.append(tmp)
            
    playId2tag_morph[playId] = words


# In[16]:


from khaiii import KhaiiiApi
api = KhaiiiApi()

playId2title_morph = dict()

for playId in playId2title:
    title = playId2title[playId]
    words = []
    if title == '\u3000\u3000\u3000\u3000\u3000\u3000\u3000\u3000' or len(title) == 0 or len(title.strip()) ==0: title = '무제'
    
    for word in api.analyze(title):
        for morph in word.morphs:
            tmp = str(morph.lex +"/" + morph.tag)
            words.append(tmp)
            
    playId2title_morph[playId] = words


# In[17]:


with open(base_path + 'data/playId2tag_morph', 'wb') as f:
    pickle.dump(playId2tag_morph, f)     


# In[18]:


with open(base_path + 'data/playId2title_morph', 'wb') as f:
    pickle.dump(playId2title_morph, f)        


# In[19]:


base_path = './data_kakao/'
train_playId_set,public_playId_set,final_playId_set,train_songId_set,public_songId_set,final_songId_set,playId2songIds,playId2tags,playId2tagIds,playId2title, songId2playIds,tag2playIds,tag2tagId,tagId2tag,songId2albumId,songId2artistIds,songId2name,songId2gnrs,songId2dtlgnrs,songId2gnrIds,songId2date,songId2year,songId2month, playId_w_tags, playId2title_morph, playId2tag_morph, playId2updt, songId2artists = load(base_path)


# In[20]:


total_playId_set = set(train_playId_set) | set(public_playId_set) | set(final_playId_set)
len(total_playId_set), max(total_playId_set)
total_playlist_count = max(total_playId_set) + 1

total_song_set = set(train_songId_set) | set(public_songId_set) | set(final_songId_set)
len(total_song_set), max(total_song_set)
total_song_count = max(total_song_set) + 1

final_playId_set = set(final_playId_set)
public_playId_set = set(public_playId_set)


# In[21]:


from khaiii import KhaiiiApi
api = KhaiiiApi()


# In[22]:


playId2title_tag_morph = dict()

for playId in tqdm(list(set(list(playId2tags.keys()) + list(playId2title.keys())))):
    tags = playId2tags[playId]
    tags = ' '.join(tags)
    
    title = playId2title[playId]
    
    words = []
    if tags == '\u3000\u3000\u3000\u3000\u3000\u3000\u3000\u3000' or len(tags) == 0 or len(tags.strip()) ==0: tags = ''
    if title == '\u3000\u3000\u3000\u3000\u3000\u3000\u3000\u3000' or len(title) == 0 or len(title.strip()) ==0: title = ''
    if tags == '' and title == '': continue
    
    for word in api.analyze(tags + title):
        for morph in word.morphs:
            tmp = str(morph.lex +"/" + morph.tag)
            words.append(tmp)
            
    playId2title_tag_morph[playId] = words


# In[23]:


# 품사 filtering

playId2title_tag = dict()

POS = ['NNG', 'SN', 'MAG', 'SL', 'VV', 'NR', 'NNP']

for playId in tqdm(playId2title_tag_morph):
    morphs = playId2title_tag_morph[playId]
    
    new_morphs = []
    for morph in morphs:
        tmp = morph.split('/')
        if tmp[-1] not in POS: continue
        new_morphs.append(tmp[0])
    
    # tags
    tags = playId2tags[playId]
    new_morphs.extend(tags)
    
    # title
    title = playId2title[playId]
    for word in re.findall(r"[\w']+", title):
        if word in tag2tagId:
            new_morphs.append(word)
    
    playId2title_tag[playId] = new_morphs


# In[26]:


with open('playId2title_tag', 'wb') as f:
    pickle.dump(playId2title_tag, f)


# In[ ]:




