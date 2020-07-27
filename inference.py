#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import pickle


# In[2]:


with open('./ensemble/results_tag_final.json', 'rb') as f:
    tag_dict = json.load(f)


# In[3]:


with open('./ensemble/ensemble_A', 'rb') as f:
    result_A_dict = pickle.load(f)

with open('./ensemble/ensemble_CD', 'rb') as f:
    result_CD_dict = pickle.load(f)


# In[4]:


len(result_A_dict), len(result_CD_dict), len(tag_dict)


# In[5]:


import copy
from tqdm import tqdm
final_dict = copy.deepcopy(tag_dict)


# In[6]:


for idx, result in tqdm(enumerate(tag_dict)):
    play_id = result['id']
    if play_id in result_A_dict:
        final_dict[idx]['songs'] = np.asarray(result_A_dict[play_id]).tolist()[:]
        assert len(result_A_dict[play_id][:]) == 100
        
    elif play_id in result_CD_dict:
        final_dict[idx]['songs'] = np.asarray(result_CD_dict[play_id]).tolist()[:]
        assert len(result_CD_dict[play_id][:]) == 100
        
    else:
        print("something worng")


# In[7]:


with open('final_result.json', 'w') as f:
    json.dump(final_dict, f)


# In[8]:


### 검사하는 코드


# In[9]:


for idx, result in tqdm(enumerate(final_dict)):
    play_id = result['id']
    if play_id in result_A_dict:
        assert result_A_dict[play_id] == result['songs']
    else:
        assert result_CD_dict[play_id] == result['songs']


# In[ ]:




