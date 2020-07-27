import numpy as np
from data import load
from tqdm import tqdm
import re

import json
import os


base_path = './data_kakao/'
train_playId_set,public_playId_set,final_playId_set,train_songId_set,public_songId_set,final_songId_set,playId2songIds,playId2tags,playId2tagIds,playId2title,songId2playIds,tag2playIds,tag2tagId,tagId2tag,songId2albumId,songId2artistIds,songId2name,songId2gnrs,songId2dtlgnrs,songId2gnrIds,songId2date,songId2year,songId2month, playId_w_tags, playId2title_morph, playId2tag_morph, playId2updt, songId2artists = load(base_path)

playId2artist = dict()
for play_id in tqdm(playId2songIds):
    songs_in_play = playId2songIds[play_id]
    artist_set = set()
    for song_id in songs_in_play:
        artist_set.update(songId2artistIds[song_id])
    playId2artist[play_id] = artist_set

playId2title_morph_new = dict()
title_morph2playId = dict()

POS = ['NNG', 'SN', 'MAG', 'SL', 'VV', 'NR'] # 'JKB' (부사), 'VA' (형용사), 'SW' (기타기호)

for play_id in tqdm(playId2title_morph):
    morphs = playId2title_morph[play_id]
    new_morphs = []
    for morph in morphs:
        if '무제' in morph: continue
        if morph.split('/')[1] not in POS: continue
        new_morphs.append(morph)
        if morph not in title_morph2playId:
            title_morph2playId[morph] = [play_id]
        else:
            title_morph2playId[morph] += [play_id]
    playId2title_morph_new[play_id] = new_morphs

tag_set = [tag for tag in tag2playIds if len(tag2playIds[tag]) >= 1]
tag_dict = dict()
for idx, tag in enumerate(tag_set):
    tag_dict[tag] = idx
    
tag2playIds_new = dict()
playId2tags_new = dict()

for tag in tqdm(tag2playIds):
    if tag not in tag_set: continue
    for play_id in tag2playIds[tag]:
        if play_id not in playId2tags_new:
            playId2tags_new[play_id] = [tag_dict[tag]]
        else:
            playId2tags_new[play_id] += [tag_dict[tag]]
            
        if play_id not in playId2tagIds:
            playId2tagIds[play_id] = [tag_dict[tag]]
        else:
            playId2tagIds[play_id] += [tag_dict[tag]]
            
    tag2playIds_new[tag_dict[tag]] = tag2playIds[tag]
print(len(tag_set))
tag_set = np.array(tag_set)
print(len(tag_set))

total_playId_set = set(train_playId_set) | set(public_playId_set) | set(final_playId_set)
len(total_playId_set), max(total_playId_set)
total_playlist_count = max(total_playId_set) + 1

total_song_set = set(train_songId_set) | set(public_songId_set) | set(final_songId_set)
len(total_song_set), max(total_song_set)
total_song_count = max(total_song_set) + 1

total_playlist_count, total_song_count

th = 0

# A에 해당하는 playlist 개수
A = []

for play_id in final_playId_set:
    songs_in_play = playId2songIds[play_id]
    if len(songs_in_play) <= th:
        continue
    A.append(play_id)
         
len(A)

def User_KNN_A(A, result_dict):

    CF_K = 45
    CF_S = 10
    
    CB_tag_S = 20
    alpha_tag = 2.2
    CB_artist_S = 30    
    alpha_artist = 0.3

    for play_id in [A]:
        result_dict[play_id] = []
        songs_in_play = set(playId2songIds[play_id])


        tags_in_play = []
        if play_id in playId2tags_new:
            tags_in_play = playId2tags_new[play_id]
        titles_in_play = playId2title_morph_new[play_id]

        tags_in_play = set(tags_in_play)
        titles_in_play = set(titles_in_play)

        if play_id in playId2artist:
            artists_in_play = playId2artist[play_id]

        sim_scores = []
        play_id2s = []
        for play_id2 in playId2songIds: #train_playId_set 
            if play_id2 == play_id: continue
            if play_id2 not in playId2tags_new: continue
            
            # CF
            songs_in_play2 = set(playId2songIds[play_id2])
            if len(songs_in_play2) == 0: continue
            sim_score = 0.
            CF_sim_score = len(songs_in_play & songs_in_play2) / ((np.sqrt(len(songs_in_play)) * np.sqrt(len(songs_in_play2))) + CF_S)
            sim_score += CF_sim_score

            # CB-tag
            if play_id2 in playId2tags_new:
                tags_in_play2 = playId2tags_new[play_id2]
            titles_in_play2 = playId2title_morph_new[play_id2]

            tags_in_play2 = set(tags_in_play2)
            titles_in_play2 = set(titles_in_play2)

            CB_sim_score = (len(tags_in_play & tags_in_play2) + len(titles_in_play & titles_in_play2))             / ((np.sqrt(len(tags_in_play)+len(titles_in_play)) * np.sqrt(len(tags_in_play2)+len(titles_in_play2))) + CB_tag_S)
            sim_score += alpha_tag * CB_sim_score

            artists_in_play2 = playId2artist[play_id2]

            CB_sim_score = (len(artists_in_play & artists_in_play2))             / ((np.sqrt(len(artists_in_play)) * np.sqrt(len(artists_in_play2))) + CB_artist_S)
            sim_score += alpha_artist * CB_sim_score                
                
            sim_scores.append(sim_score)
            play_id2s.append(play_id2)
            
        sorted_indices = np.argsort(sim_scores)[::-1][:CF_K]
        sorted_scores = np.asarray(sim_scores)[sorted_indices]
        sorted_play_id2s = np.asarray(play_id2s)[sorted_indices]
        
        play_mat = np.zeros((CF_K, len(tag_set)))
        for idx, play_id2 in enumerate(sorted_play_id2s):
            play_mat[idx, playId2tags_new[play_id2]] = 1.
        play_scores = sorted_scores.reshape([CF_K, 1])
        play_CF_vec = (play_mat * play_scores).mean(axis=0)
        
        # 이미 구입한 아이템 제외
        play_CF_vec[list(tags_in_play)] = 0.

        results = np.argsort(-play_CF_vec)[:10]
        result_dict[play_id] = tag_set[results]

import multiprocessing
from multiprocessing import Manager
from itertools import repeat
manager = Manager()
num_cores  = multiprocessing.cpu_count()

tmp = {}
for play_id in A:
    tmp[play_id] = []

result_A_CF = manager.dict(tmp)

pool = multiprocessing.Pool(num_cores)
pool.starmap(User_KNN_A, zip(A, repeat(result_A_CF)))

result_A_CF = result_A_CF.copy()

# ## CD 추천

CD = []
for play_id in final_playId_set:
    if len(playId2songIds[play_id]) == 0:
        CD.append(play_id)
len(CD)

def User_KNN_CD(smallCD, result_dict):
    # params
    K = 80
    S = 20
    
    for play_id in [smallCD]:
        result_dict[play_id] = []
        
        tags_in_play = []
        if play_id in playId2tags_new:
            tags_in_play = playId2tags_new[play_id]
        titles_in_play = playId2title_morph_new[play_id]
        
        tags_in_play = set(tags_in_play)
        titles_in_play = set(titles_in_play)
            
        sim_scores = []
        play_id2s = []
        for play_id2 in train_playId_set: # playId2songIds
            if play_id2 == play_id: continue
            if play_id2 not in playId2tags_new: continue

            if play_id2 in playId2tags_new:
                tags_in_play2 = playId2tags_new[play_id2]
            titles_in_play2 = playId2title_morph_new[play_id2]

            tags_in_play2 = set(tags_in_play2)
            titles_in_play2 = set(titles_in_play2)

            sim_score = (len(tags_in_play & tags_in_play2) + len(titles_in_play & titles_in_play2))                 / ((np.sqrt(len(tags_in_play)+len(titles_in_play)) * np.sqrt(len(tags_in_play2)+len(titles_in_play2))) + S)

            sim_scores.append(sim_score)
            play_id2s.append(play_id2)
            
        sorted_indices = np.argsort(sim_scores)[::-1][:K]
        sorted_scores = np.asarray(sim_scores)[sorted_indices]
        sorted_play_id2s = np.asarray(play_id2s)[sorted_indices]
        
        play_mat = np.zeros((K, len(tag_set)))
        for idx, play_id2 in enumerate(sorted_play_id2s):
            play_mat[idx, playId2tags_new[play_id2]] = 1.
        play_scores = sorted_scores.reshape([K, 1])
        play_CF_vec = (play_mat * play_scores).mean(axis=0)
        
        # 이미 구입한 아이템 제외
        play_CF_vec[list(tags_in_play)] = 0.
        
        results = np.argsort(-play_CF_vec)[:10]
        result_dict[play_id] = tag_set[results]

import multiprocessing
from multiprocessing import Manager
from itertools import repeat
manager = Manager()
num_cores  = multiprocessing.cpu_count()

tmp = {}
for play_id in CD:
    tmp[play_id] = []

result_CD_CF = manager.dict(tmp)

pool = multiprocessing.Pool(num_cores)
pool.starmap(User_KNN_CD, zip(CD, repeat(result_CD_CF)))

result_CD_CF = result_CD_CF.copy()


## 제공된 장르 mostpop 결과 - 형식 맞추기 위해서 사용 (직접 만드니까 오류가 발생해서 그대로 사용하기로 결정했습니다.)
with open("results_gnrpop_final.json", 'r', encoding='windows-1252') as f:
    result_dict = json.load(f)

import copy
new_result_dict = copy.deepcopy(result_dict)

for idx, result in tqdm(enumerate(new_result_dict)):
    play_id = result['id'] 
    if play_id in A:
        CF_recommendation = result_A_CF[play_id].tolist()
        assert len(CF_recommendation) == 10
        new_result_dict[idx]['tags'] = CF_recommendation
        
    if play_id in CD:
        CF_recommendation = result_CD_CF[play_id].tolist()
        assert len(CF_recommendation) == 10
        new_result_dict[idx]['tags'] = CF_recommendation

with open('ensemble/results_tag_final.json', 'w') as f:
    json.dump(new_result_dict, f)