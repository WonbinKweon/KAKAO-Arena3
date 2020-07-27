#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from data import load
from tqdm import tqdm

import json
import pickle

from scipy import sparse
import copy


# In[ ]:


def play_id_same_album_th(play_id, prev_results, top_result=0):
    
    same_album_id_set = set()
    existing_set = set(playId2songIds[play_id]) | set(prev_results)
    
    for song_id in playId2songIds[play_id]:
        album_id = songId2albumId[song_id]
        same_album_id_set.update(albumId2songId[album_id])
        
    for song_id in prev_results[:top_result]:
        album_id = songId2albumId[song_id]
        same_album_id_set.update(albumId2songId[album_id])        
    
    same_album_id_set = list(same_album_id_set - existing_set)
    
    #
    candidates = np.zeros((total_song_count, ))
    candidates[same_album_id_set] = 1.
    candidates[prev_results] = 0.
    candidates[playId2songIds[play_id]] = 0.
    candidates[song_date_vector > playId2date[play_id]] = 0.
    
    candidate_ids = candidates.nonzero()[0]
    candidate_pops = []

    for candidate in candidate_ids:
        if candidate not in songId2playIds:
            #candidate_pops.append(0)
            final_pop = meta_only_song_pop_dict[candidate]
            candidate_pops.append(final_pop)
            
        else:
            candidate_pops.append(len(songId2playIds[candidate]))

    results = candidate_ids[np.argsort(-np.asarray(candidate_pops))]
    return results


# score가 0.009 이하의 곡들은 제거 후, 기존 playlist에 들어있는 앨범 수록곡의 pop 정렬로 채우기
def play_id_same_artist_th(play_id, prev_results, top_result=0):
    same_artist_id_set = set()
    
    existing_set = set(playId2songIds[play_id]) | set(prev_results)
    
    for song_id in playId2songIds[play_id]:
        artist_id = songId2artistIds[song_id][0]
        same_artist_id_set.update(artistId2songId[artist_id])
        
    for song_id in prev_results[:top_result]:
        artist_id = songId2artistIds[song_id][0]
        same_artist_id_set.update(artistId2songId[artist_id])
    
    same_artist_id_set = list(same_artist_id_set - existing_set)
    
    #
    candidates = np.zeros((total_song_count, ))
    candidates[same_artist_id_set] = 1.
    candidates[prev_results] = 0.
    candidates[playId2songIds[play_id]] = 0.
    candidates[song_date_vector > playId2date[play_id]] = 0.
    
    candidate_ids = candidates.nonzero()[0]
    candidate_pops = []
        
    for candidate in candidate_ids:
        if candidate not in songId2playIds:
             # candidate_pops.append(0)
            final_pop = meta_only_song_pop_dict[candidate]
            candidate_pops.append(final_pop)
                   
        else:
            candidate_pops.append(len(songId2playIds[candidate]))

    results = candidate_ids[np.argsort(-np.asarray(candidate_pops))]
    return results


# score가 0.009 이하의 곡들은 제거 후, 기존 playlist에 들어있는 앨범 수록곡의 pop 정렬로 채우기
def play_id_same_genre_th(play_id, prev_results):
    same_genre_id_set = set()
    
    existing_set = set(playId2songIds[play_id]) | set(prev_results)
    
    for song_id in playId2songIds[play_id]:
        if len(songId2gnrIds[song_id]) < 1: continue
        gnr_id = songId2gnrIds[song_id][0]
        same_genre_id_set.update(gnrId2songId[gnr_id])
    
    same_genre_id_set = list(same_genre_id_set - existing_set)
    
    candidates = np.zeros((total_song_count, ))
    candidates[same_genre_id_set] = 1.
    candidates[prev_results] = 0.
    candidates[playId2songIds[play_id]] = 0.
    candidates[song_date_vector > playId2date[play_id]] = 0.
    
    candidate_ids = candidates.nonzero()[0]
    candidate_pops = []

    for candidate in candidate_ids:
        if candidate not in songId2playIds:
            final_pop = meta_only_song_pop_dict[candidate]
            candidate_pops.append(final_pop)
        else:
            candidate_pops.append(len(songId2playIds[candidate]))


    results = candidate_ids[np.argsort(-np.asarray(candidate_pops))]
    return results

def rerank(play_id, result, is_artist=True, is_album=True):
    new_result1 = []
    new_result2 = []
    
    if play_id not in playId2songIds:
        return result
        
    songs_in_play = playId2songIds[play_id]
    
    album_set = set()
    artist_set = set()
    for song in songs_in_play:
        album_set.update([songId2albumId[song]])
        artist_set.update([songId2artistIds[song][0]])
    artist_set = artist_set - set([0])
    
    if len(songs_in_play) < 0: return result
    
    # 들어가는 조건
    if (len(artist_set) < 3) or ((len(songs_in_play) / len(artist_set)) >= 2):
    
        for r in result:
            is_reranked = False
            if is_artist:
                if songId2artistIds[r][0] in artist_set:
                    is_reranked = True
            if is_album:
                if songId2albumId[r] in album_set:
                    is_reranked = True

            if is_reranked:
                new_result1.append(r)
            else:
                new_result2.append(r)

        return new_result1 + new_result2
    else:
        return result


# ### 기본 데이터 로드

# In[ ]:


base_path = './data_kakao/'
train_playId_set,public_playId_set,final_playId_set,train_songId_set,public_songId_set,final_songId_set,playId2songIds,playId2tags,playId2tagIds,playId2title, songId2playIds,tag2playIds,tag2tagId,tagId2tag,songId2albumId,songId2artistIds,songId2name,songId2gnrs,songId2dtlgnrs,songId2gnrIds,songId2date,songId2year,songId2month, playId_w_tags, playId2title_morph, playId2tag_morph, playId2updt, songId2artists = load(base_path)


# In[ ]:


total_playId_set = set(train_playId_set) | set(public_playId_set) | set(final_playId_set)
len(total_playId_set), max(total_playId_set)
total_playlist_count = max(total_playId_set) + 1

total_song_set = set(train_songId_set) | set(public_songId_set) | set(final_songId_set)
len(total_song_set), max(total_song_set)
total_song_count = max(total_song_set) + 1

final_playId_set = set(final_playId_set)

play_id2final_idx = dict()
play_id2final_idx_R = dict()

for idx, play_id in enumerate(final_playId_set):
    play_id2final_idx[play_id] = idx
    play_id2final_idx_R[idx] = play_id


# In[ ]:


song_date_vector = []
for song_id in range(total_song_count):
    song_date_vector.append(songId2date[song_id])
song_date_vector = np.asarray(song_date_vector)

playId2date = dict()
for playId in playId2updt:
    playId2date[playId] = int(''.join(playId2updt[playId].split(' ')[0].split('-')))


# In[ ]:


th = 0

# A에 해당하는 playlist 개수
A = []

for play_id in final_playId_set:
    songs_in_play = playId2songIds[play_id]
    if len(songs_in_play) <= th:
        continue
    A.append(play_id)
         
CD = []
for play_id in final_playId_set:
    if len(playId2songIds[play_id]) == 0:
        CD.append(play_id)


# In[ ]:


gnrId2songId = dict()
artistId2songId = dict()
albumId2songId = dict()

for song_id in tqdm(songId2gnrIds):
    for genre_id in songId2gnrIds[song_id]:
        if genre_id not in gnrId2songId:
            gnrId2songId[genre_id] = [song_id]
        else:
            gnrId2songId[genre_id].append(song_id)
            
for song_id in tqdm(songId2artistIds):
    for artist_id in songId2artistIds[song_id]:
        if artist_id not in artistId2songId:
            artistId2songId[artist_id] = [song_id]
        else:
            artistId2songId[artist_id].append(song_id)
            
for song_id in tqdm(songId2albumId):
    album_id = songId2albumId[song_id]
    if album_id not in albumId2songId:
        albumId2songId[album_id] = [song_id]
    else:
        albumId2songId[album_id].append(song_id)


# In[ ]:


# artist
playId2artistId = dict()

for play_id in tqdm(range(total_playlist_count)):
    if play_id not in playId2songIds: continue
        
    songs_in_play = playId2songIds[play_id]
    
    artist_set = set()
    for song in songs_in_play:
        for artist_id in songId2artistIds[song]:
            if artist_id == 0: continue
            if play_id not in playId2artistId:
                playId2artistId[play_id] = {artist_id:1}
            else:
                if artist_id in playId2artistId[play_id]:
                    playId2artistId[play_id][artist_id] += 1
                else:
                    playId2artistId[play_id][artist_id] = 1
          
        
# artist
playId2albumId = dict()

for play_id in tqdm(range(total_playlist_count)):
    if play_id not in playId2songIds: continue
        
    songs_in_play = playId2songIds[play_id]
    
    album_set = set()
    for song in songs_in_play:
        album_Id =  songId2albumId[song]
            
        if play_id not in playId2albumId:
            playId2albumId[play_id] = {album_Id:1}
        else:
            if album_Id in playId2albumId[play_id]:
                playId2albumId[play_id][album_Id] += 1
            else:
                playId2albumId[play_id][album_Id] = 1


# In[ ]:


meta_only_song_pop_dict = dict()

for song in tqdm(range(total_song_count)):
    if song not in songId2playIds:
        artist_id = songId2artistIds[song][0]

        # 그 아티스트의 평균적인 플레이 횟수
        artist_pop = 0.
        denom = 0.
        for song_id in artistId2songId[artist_id]:
            if song_id not in songId2playIds: continue
            artist_pop += len(songId2playIds[song_id])
            if len(songId2playIds[song_id]) > 0:
                denom += 1

        if denom > 0:
            artist_pop /= denom            

        # 0이면 artist로
        if artist_pop == 0:

            # 앨범
            album_id = songId2albumId[song]

            # 그 앨범의 평균적인 플레이 횟수
            album_pop = 0.
            denom = 0.
            for song_id in albumId2songId[album_id]:

                if song_id not in songId2playIds: continue
                album_pop += len(songId2playIds[song_id])
                if len(songId2playIds[song_id]) > 0:
                    denom += 1

            if denom > 0:
                album_pop /= denom

        if artist_pop > 0:
            final_pop = artist_pop
        else:
            final_pop = album_pop

        meta_only_song_pop_dict[song] = final_pop


# #### title/tag

# In[ ]:


with open('playId2title_tag', 'rb') as f:
    playId2title_tag = pickle.load(f)


# ### conduct inference

# #### AB

# In[ ]:


def User_KNN_A(small_A, result_dict):
    
    CF_S = 10
    tag_S = 20
    artist_S = 10
    
    for play_id in [small_A]:

        play_id = int(play_id)
        songs_in_play = playId2songIds[play_id]
        #result_dict[play_id] = []
        
        sim_scores = np.zeros((total_playlist_count,))
        
        # CF
        for play_id2 in playId2songIds:
            if play_id == play_id2: continue
            if len(playId2songIds[play_id2]) == 0: continue
                
            songs_in_play2 = set(playId2songIds[play_id2])
            
            num_co_occur = len(set(songs_in_play) & songs_in_play2)
            if num_co_occur == 0: continue
                
            CF_sim_score = num_co_occur / ((np.sqrt(len(set(songs_in_play))) * np.sqrt(len(songs_in_play2))) + CF_S)
            sim_scores[play_id2] += CF_sim_score
            
            
        # tag/title 1
        if is_tag:
            if play_id in playId2title_tag:
                for play_id2 in playId2title_tag:
                    if play_id == play_id2: continue

                    tag_in_play1 = set(playId2title_tag[play_id])
                    tag_in_play2 = set(playId2title_tag[play_id2])

                    num_co_occur = len(tag_in_play1 & tag_in_play2)
                    if num_co_occur == 0: continue

                    CB_sim_score = num_co_occur / ((np.sqrt(len(tag_in_play1)) * np.sqrt(len(tag_in_play2))) + tag_S)
                    sim_scores[play_id2] += CB_sim_score * tag_lambda
                
                
        # tag/title 2
        if is_tag2:
            if play_id in playId2title_tag:       
                for play_id2 in playId2title_tag:
                    if play_id == play_id2: continue

                    tag_title_in_play1 = set(playId2title_tag[play_id])
                    tags_in_play2 = set(playId2tags[play_id2])

                    # title morph
                    titles_in_play1 = set(playId2title_morph[play_id])
                    titles_in_play2 = set(playId2title_morph[play_id2])                

                    score = (len(tag_title_in_play1 & tags_in_play2) + len(titles_in_play1 & titles_in_play2))

                    if score == 0.: continue
                    denom = ((np.sqrt(len(tag_title_in_play1)+len(titles_in_play1)) * np.sqrt(len(tags_in_play2)+len(titles_in_play2))) + 20)

                    CB_sim_score2 = score / denom
                    sim_scores[play_id2] += CB_sim_score2 * tag_lambda2
                
                
        # artist
        if is_artist:
            if play_id in playId2artistId:                
                for play_id2 in playId2artistId:
                    if play_id == play_id2: continue

                    artist_in_play1 = set(playId2artistId[play_id])
                    artist_in_play2 = set(playId2artistId[play_id2])

                    co_occur_artists = artist_in_play1 & artist_in_play2
                    if len(co_occur_artists) == 0: continue

                    artist_sim_score = len(co_occur_artists) / ((np.sqrt(len(artist_in_play1)) * np.sqrt(len(artist_in_play2))) + artist_S)
                    sim_scores[play_id2] += artist_sim_score * artist_lambda      
        
                
        # album
        if is_album:
            if play_id in playId2albumId:      
                for play_id2 in playId2albumId:
                    if play_id == play_id2: continue

                    album_in_play1 = set(playId2albumId[play_id])
                    album_in_play2 = set(playId2albumId[play_id2])

                    co_occur_albums = album_in_play1 & album_in_play2
                    if len(co_occur_albums) == 0: continue

                    album_sim_score = len(co_occur_albums) / ((np.sqrt(len(album_in_play1)) * np.sqrt(len(album_in_play2))) + artist_S)
                    sim_scores[play_id2] += album_sim_score * artist_lambda          
          
        sorted_play_id2s = np.argsort(-sim_scores)[:CF_K]
        sorted_scores = sim_scores[sorted_play_id2s]

        play_mat = np.zeros((CF_K, total_song_count))

        for idx, play_id2 in enumerate(sorted_play_id2s):
            if play_id2 not in playId2songIds: continue
            play_mat[idx, playId2songIds[play_id2]] = 1.

        play_scores = sorted_scores.reshape([CF_K, 1])

        # 1 x total_playlist_count
        play_CF_vec = (play_mat * play_scores).mean(axis=0)

        # 이미 구입한 아이템 제외
        play_CF_vec[list(songs_in_play)] = 0.

        # date 제외
        # 미래 곡 제외
        play_CF_vec[song_date_vector > playId2date[play_id]] = 0.

        # non-zero 처리
        non_zero_songs = play_CF_vec.nonzero()[0]
        non_zero_scores = play_CF_vec[non_zero_songs]

        sorted_idx = np.argsort(-non_zero_scores)[:min(non_zero_songs.shape[0], 100)]
        result_songs = non_zero_songs[sorted_idx]
        result_scores = non_zero_scores[sorted_idx]
        
        result_dict[play_id] = (result_songs[:], result_scores[:])


# In[ ]:


CF_K = 10
tag_lambda = 0.2
tag_lambda2 = 0.2
artist_lambda = 0.05
album_lambda = 0.05


# In[ ]:


conditions = [(False, False, False, False), (True, False, False, False), (False, False, True, False),
              (True, False, True, False), (False, False, False, True), (True, False, False, True),
              (False, False, True, True), (False, True, False, False)]
result_per_condition = []


# In[ ]:


for is_tag, is_tag2, is_artist, is_album in conditions:
    print(is_tag, is_tag2, is_artist, is_album)
    
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
    
    result_per_condition.append(result_A_CF.copy())


# In[ ]:


for idx, result_dict in enumerate(result_per_condition):
    with open('./ensemble/song_' + str(idx), 'wb') as f:
        pickle.dump(result_dict, f)


# ### POST processing

# In[ ]:


th = 0.012
len_th = 90
c = 0
post_list = []


# In[ ]:


import copy
for idx, result_dict in enumerate(result_per_condition):
    
    new_result_dict = dict()
    
    # score filtering
    for play_id in tqdm(result_dict):
        results, scores = result_dict[play_id]
        results = results[scores > th]

        if len(list(results)) > len_th:
            results = results[:len_th]
            c+=1

        tmp = play_id_same_album_th(play_id, results)
        tmp2 = []
        tmp3 = []

        if len(list(results) + list(tmp)) < 100:
            tmp2 = play_id_same_artist_th(play_id, results.tolist() + list(tmp)).tolist()

        if len(list(results) + list(tmp) + list(tmp2)) < 100:
            tmp3 = play_id_same_genre_th(play_id, results.tolist() + list(tmp) + list(tmp2)).tolist()

        r = list(results) + list(tmp) + list(tmp2) + list(tmp3)
        assert len(r) >= 100

        new_result_dict[play_id] = r[:min(100, len(r))]
        
    # rerank
    reranked_dict = copy.deepcopy(new_result_dict)

    for play_id in tqdm(new_result_dict):
        result = new_result_dict[play_id]
        new_result = rerank(play_id, result, is_artist=True, is_album=False)
        reranked_dict[play_id] = new_result
        
    post_list.append(copy.deepcopy(reranked_dict))


# In[ ]:


scores = [0.3088, 0.3094, 0.3080, 0.3085, 0.3092, 0.3095, 0.3075, 0.3092]
scores = np.asarray(scores)
scores = np.exp(scores) / np.sum(np.exp(scores))


# In[ ]:


T = 10
ensemble_A_dict = dict()

for idx, play_id in tqdm(enumerate(post_list[0])):
    
    if play_id not in A: continue
    
    total_result = dict()

    CF_recommendation1 = post_list[0][play_id]
    CF_recommendation2 = post_list[1][play_id]
    CF_recommendation3 = post_list[2][play_id]
    CF_recommendation4 = post_list[3][play_id]
    CF_recommendation5 = post_list[4][play_id]
    CF_recommendation6 = post_list[5][play_id]
    CF_recommendation7 = post_list[6][play_id]
    CF_recommendation8 = post_list[7][play_id]
    
    for i, recommendation in enumerate([CF_recommendation1, CF_recommendation2, CF_recommendation3, CF_recommendation4,                                       CF_recommendation5, CF_recommendation6, CF_recommendation7, CF_recommendation8]):
        
        score = scores[i]
        for ranking, song in enumerate(recommendation):
            
            result = np.exp(-ranking / T) * score
            
            if song not in total_result:
                total_result[song] = result
            else:
                total_result[song] += result
    
    new_result = sorted(list(total_result.keys()), key=lambda x: total_result[x], reverse=True)
    ensemble_A_dict[play_id] = new_result[:100]


# In[ ]:


with open('./ensemble/ensemble_A', 'wb') as f:
    pickle.dump(ensemble_A_dict, f)


# ## CD

# In[ ]:


def User_KNN_CD(small_CD, result_dict):
    CF_K = 90
    for play_id in [small_CD]:

        result_dict[play_id] = []
        
        sim_scores = np.zeros((total_playlist_count,))
        
        if play_id in playId2title_tag:
            for play_id2 in playId2title_tag:
                if play_id == play_id2: continue

                tag_in_play1 = set(playId2title_tag[play_id])
                tag_in_play2 = set(playId2title_tag[play_id2])

                num_co_occur = len(tag_in_play1 & tag_in_play2)
                if num_co_occur == 0: continue

                CB_sim_score = num_co_occur / ((np.sqrt(len(tag_in_play1)) * np.sqrt(len(tag_in_play2))) + 20)
                sim_scores[play_id2] += CB_sim_score        
        
        sorted_play_id2s = np.argsort(-sim_scores)[:CF_K]
        sorted_scores = sim_scores[sorted_play_id2s]

        play_mat = np.zeros((CF_K, total_song_count))

        for idx, play_id2 in enumerate(sorted_play_id2s):
            if play_id2 not in playId2songIds: continue
            play_mat[idx, playId2songIds[play_id2]] = 1.

        play_scores = sorted_scores.reshape([CF_K, 1])

        # 1 x total_playlist_count
        play_CF_vec = (play_mat * play_scores).mean(axis=0)

        # date 제외
        # 미래 곡 제외
        play_CF_vec[song_date_vector > playId2date[play_id]] = 0.

        # non-zero 처리
        non_zero_songs = play_CF_vec.nonzero()[0]
        non_zero_scores = play_CF_vec[non_zero_songs]

        sorted_idx = np.argsort(-non_zero_scores)[:min(non_zero_songs.shape[0], 100)]
        result_songs = non_zero_songs[sorted_idx]
        result_scores = non_zero_scores[sorted_idx]
        
        result_dict[play_id] = (result_songs, result_scores)


# In[ ]:


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

result_CD_CF_dict = result_CD_CF.copy()


# In[ ]:


def play_id_same_genre_th(play_id, prev_results, top_result=0):
    
    same_genre_id_set = set()
    
    existing_set = set(prev_results)

    for song_id in prev_results[:top_result]:
        if len(songId2gnrIds[song_id]) < 1: continue
        gnr_id = songId2gnrIds[song_id][0]
        same_genre_id_set.update(gnrId2songId[gnr_id])
    
    same_genre_id_set = list(same_genre_id_set - existing_set)
    
    candidates = np.zeros((total_song_count, ))
    candidates[same_genre_id_set] = 1.
    candidates[prev_results] = 0.
    candidates[song_date_vector > playId2date[play_id]] = 0.
    
    candidate_ids = candidates.nonzero()[0]
    candidate_pops = []

    for candidate in candidate_ids:
        if candidate not in songId2playIds:
            #candidate_pops.append(0)
            
            final_pop = meta_only_song_pop_dict[candidate]
            candidate_pops.append(final_pop)
            
        else:
            candidate_pops.append(len(songId2playIds[candidate]))

    results = candidate_ids[np.argsort(-np.asarray(candidate_pops))]
    return results

def play_id_most_pop(play_id, prev_results):

    songs_in_play = prev_results

    candidates = np.ones((total_song_count, ))
    candidates[prev_results] = 0.
    candidates[song_date_vector > playId2date[play_id]] = 0.

    candidate_ids = candidates.nonzero()[0]
    candidate_pops = []

    for candidate in candidate_ids:
        if candidate not in songId2playIds:
            #candidate_pops.append(0)
            final_pop = meta_only_song_pop_dict[candidate]
            candidate_pops.append(final_pop)
            
        else:
            candidate_pops.append(len(songId2playIds[candidate]))

    results = candidate_ids[np.argsort(-np.asarray(candidate_pops))]
    return results  


# In[ ]:


# 100미만 + threshold 미만 채우기
result_CD_post_dict = dict()

th = 0.0
c1, c2, c3, c4 = 0, 0, 0, 0
for play_id in tqdm(result_CD_CF_dict):
    results, scores = result_CD_CF_dict[play_id]
    results = results[scores > th]
    
    if len(list(results)) == 100:
        result_CD_post_dict[play_id] = results.tolist()
        
    else:
        tmp = play_id_same_album_th(play_id, results, 5)
        c1 += 1
        
        tmp2 = []
        tmp3 = []
        tmp4 = []

        if len(list(results) + list(tmp)) < 100:
            c2 += 1
            tmp2 = play_id_same_artist_th(play_id, results.tolist() + list(tmp), 5).tolist()
            
        if len(list(results) + list(tmp) + list(tmp2)) < 100:
            c3 += 1
            tmp3 = play_id_same_genre_th(play_id, results.tolist() + list(tmp) + list(tmp2), 5).tolist()
            
        if len(list(results) + list(tmp) + list(tmp2) + list(tmp3)) < 100:
            c4 += 1
            tmp4 = play_id_most_pop(play_id, results.tolist() + list(tmp) + list(tmp2) + list(tmp3)).tolist()        
            
        r = list(results) + list(tmp) + list(tmp2) + list(tmp3)  + list(tmp4)
        assert len(r) >= 100
        
        result_CD_post_dict[play_id] = r[:min(100, len(r))]


# In[ ]:


with open('./ensemble/ensemble_CD', 'wb') as f:
    pickle.dump(result_CD_post_dict, f)

