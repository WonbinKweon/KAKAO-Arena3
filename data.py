import numpy as np
import pickle

def load_tmp(base_path):


    #load index set
    with open(base_path + 'data/train_playId_set', 'rb') as f:
        train_playId_set = pickle.load(f)
    with open(base_path + 'data/public_playId_set', 'rb') as f:
        public_playId_set = pickle.load(f)
    with open(base_path + 'data/final_playId_set', 'rb') as f:
        final_playId_set = pickle.load(f)
    with open(base_path + 'data/train_songId_set', 'rb') as f:
        train_songId_set = pickle.load(f)
    with open(base_path + 'data/public_songId_set', 'rb') as f:
        public_songId_set = pickle.load(f)
    with open(base_path + 'data/final_songId_set', 'rb') as f:
        final_songId_set = pickle.load(f)
    
    
    #load dictionary for data
    with open(base_path + 'data/playId2songIds', 'rb') as f:
        playId2songIds = pickle.load(f)
    with open(base_path + 'data/playId2tags', 'rb') as f:
        playId2tags = pickle.load(f)
    with open(base_path + 'data/playId2tagIds','rb') as f:
        playId2tagIds = pickle.load(f)
    with open(base_path + 'data/playId2title','rb') as f:
        playId2title = pickle.load(f)
        
    with open(base_path + 'data/songId2playIds', 'rb') as f:
        songId2playIds = pickle.load(f)
    with open(base_path + 'data/tag2playIds', 'rb') as f:
        tag2playIds = pickle.load(f)
    with open(base_path + 'data/tag2tagId', 'rb') as f:
        tag2tagId = pickle.load(f)
    with open(base_path + 'data/tagId2tag','rb') as f:
        tagId2tag = pickle.load(f)
    
    #load dictionary for meta data (song)
    
    with open(base_path + 'data/songId2albumId', 'rb') as f:
        songId2albumId = pickle.load( f)
    with open(base_path + 'data/songId2artistIds', 'rb') as f:
        songId2artistIds = pickle.load(f)
        
    with open(base_path + 'data/songId2name', 'rb') as f:
        songId2name = pickle.load(f)
    with open(base_path + 'data/songId2gnrs', 'rb') as f:
        songId2gnrs = pickle.load(f)
    with open(base_path + 'data/songId2dtlgnrs', 'rb') as f:
        songId2dtlgnrs = pickle.load(f)
        
    with open(base_path + 'data/songId2gnrIds', 'rb') as f:
        songId2gnrIds = pickle.load(f)
        
    with open(base_path + 'data/songId2date', 'rb') as f:
        songId2date = pickle.load(f)

    with open(base_path + 'data/songId2year','rb') as f:
        songId2year = pickle.load(f)
    with open(base_path + 'data/songId2month','rb') as f:
        songId2month = pickle.load(f)
        
    with open(base_path + 'data/playId2updt','rb') as f:
        playId2updt = pickle.load(f)
    with open(base_path + 'data/songId2artists','rb') as f:
        songId2artists = pickle.load(f)
    
    # miscellaneous
    with open(base_path + 'data/playId_w_tags','rb') as f:
        playId_w_tags = pickle.load(f)
    

    return train_playId_set,public_playId_set,final_playId_set,train_songId_set,public_songId_set,final_songId_set,playId2songIds,playId2tags,playId2tagIds,playId2title, songId2playIds,tag2playIds,tag2tagId,tagId2tag,songId2albumId,songId2artistIds,songId2name,songId2gnrs,songId2dtlgnrs,songId2gnrIds,songId2date,songId2year,songId2month, playId_w_tags, playId2updt, songId2artists

def load(base_path):


    #load index set
    with open(base_path + 'data/train_playId_set', 'rb') as f:
        train_playId_set = pickle.load(f)
    with open(base_path + 'data/public_playId_set', 'rb') as f:
        public_playId_set = pickle.load(f)
    with open(base_path + 'data/final_playId_set', 'rb') as f:
        final_playId_set = pickle.load(f)
    with open(base_path + 'data/train_songId_set', 'rb') as f:
        train_songId_set = pickle.load(f)
    with open(base_path + 'data/public_songId_set', 'rb') as f:
        public_songId_set = pickle.load(f)
    with open(base_path + 'data/final_songId_set', 'rb') as f:
        final_songId_set = pickle.load(f)
    
    
    #load dictionary for data
    with open(base_path + 'data/playId2songIds', 'rb') as f:
        playId2songIds = pickle.load(f)
    with open(base_path + 'data/playId2tags', 'rb') as f:
        playId2tags = pickle.load(f)
    with open(base_path + 'data/playId2tagIds','rb') as f:
        playId2tagIds = pickle.load(f)
    with open(base_path + 'data/playId2title','rb') as f:
        playId2title = pickle.load(f)
        
    with open(base_path + 'data/songId2playIds', 'rb') as f:
        songId2playIds = pickle.load(f)
    with open(base_path + 'data/tag2playIds', 'rb') as f:
        tag2playIds = pickle.load(f)
    with open(base_path + 'data/tag2tagId', 'rb') as f:
        tag2tagId = pickle.load(f)
    with open(base_path + 'data/tagId2tag','rb') as f:
        tagId2tag = pickle.load(f)
    
    #load dictionary for meta data (song)
    
    with open(base_path + 'data/songId2albumId', 'rb') as f:
        songId2albumId = pickle.load( f)
    with open(base_path + 'data/songId2artistIds', 'rb') as f:
        songId2artistIds = pickle.load(f)
        
    with open(base_path + 'data/songId2name', 'rb') as f:
        songId2name = pickle.load(f)
    with open(base_path + 'data/songId2gnrs', 'rb') as f:
        songId2gnrs = pickle.load(f)
    with open(base_path + 'data/songId2dtlgnrs', 'rb') as f:
        songId2dtlgnrs = pickle.load(f)
        
    with open(base_path + 'data/songId2gnrIds', 'rb') as f:
        songId2gnrIds = pickle.load(f)
        
    with open(base_path + 'data/songId2date', 'rb') as f:
        songId2date = pickle.load(f)

    with open(base_path + 'data/songId2year','rb') as f:
        songId2year = pickle.load(f)
    with open(base_path + 'data/songId2month','rb') as f:
        songId2month = pickle.load(f)
        
    with open(base_path + 'data/playId2updt','rb') as f:
        playId2updt = pickle.load(f)
    with open(base_path + 'data/songId2artists','rb') as f:
        songId2artists = pickle.load(f)
    
    # miscellaneous
    with open(base_path + 'data/playId_w_tags','rb') as f:
        playId_w_tags = pickle.load(f)
        
    with open(base_path + 'data/playId2title_morph', 'rb') as f:
        playId2title_morph = pickle.load(f)
        
    with open(base_path + 'data/playId2tag_morph', 'rb') as f:
        playId2tag_morph = pickle.load(f)        

    return train_playId_set,public_playId_set,final_playId_set,train_songId_set,public_songId_set,final_songId_set,playId2songIds,playId2tags,playId2tagIds,playId2title, songId2playIds,tag2playIds,tag2tagId,tagId2tag,songId2albumId,songId2artistIds,songId2name,songId2gnrs,songId2dtlgnrs,songId2gnrIds,songId2date,songId2year,songId2month, playId_w_tags, playId2title_morph, playId2tag_morph, playId2updt, songId2artists