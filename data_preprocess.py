from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import os
import re

import numpy as np
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

train = pd.read_json('data_kakao/data/arena3_train.json', typ = 'frame')
val = pd.read_json('data_kakao/data/arena3_val.json',typ = 'frame')
test = pd.read_json('data_kakao/data/arena3_test.json',typ = 'frame')

#1D numpy array
train_playId_set = train['id'].unique()
val_playId_set = val['id'].unique()
test_playId_set = test['id'].unique()

train_songId_set = []
val_songId_set = []
test_songId_set = []

train_songs_temp = train['songs']
val_songs_temp = val['songs']
test_songs_temp = test['songs']

for i in train_songs_temp:
    train_songId_set += i
for i in val_songs_temp:
    val_songId_set += i
for i in test_songs_temp:
    test_songId_set += i
    
train_songId_set = np.unique(np.array(train_songId_set))
val_songId_set = np.unique(np.array(val_songId_set))
test_songId_set = np.unique(np.array(test_songId_set))

val_songcold_cnt = 0
test_songcold_cnt = 0

for songId in val_songId_set:
    if songId not in train_songId_set:
        val_songcold_cnt+=1
for songId in test_songId_set:
    if songId not in train_songId_set:
        test_songcold_cnt+=1
        


train_tags_temp = train['tags']
val_tags_temp = val['tags']
test_tags_temp = test['tags']

train_tags_set = []
val_tags_set = []
test_tags_set = []

tag2tagId = {}
tagId2tag = {}

val_tagcold_cnt = 0
test_tagcold_cnt = 0
for i in train_tags_temp:
    train_tags_set += i
for i in val_tags_temp:
    val_tags_set += i
for i in test_tags_temp:
    test_tags_set += i

train_tags_set = np.unique(np.array(train_tags_set))
val_tags_set = np.unique(np.array(val_tags_set))
test_tags_set = np.unique(np.array(test_tags_set))

cnt=0
for tag in train_tags_set:
    if tag not in tag2tagId:
        tag2tagId[tag] = cnt
        tagId2tag[cnt] = tag
        cnt+=1

for tag in val_tags_set:
    if tag not in train_tags_set:
        val_tagcold_cnt+=1
    if tag not in tag2tagId:
        tag2tagId[tag] = cnt
        tagId2tag[cnt] = tag
        cnt+=1

for tag in test_tags_set:
    if tag not in train_tags_set:
        test_tagcold_cnt+=1
    if tag not in tag2tagId:
        tag2tagId[tag] = cnt
        tagId2tag[cnt] = tag
        cnt+=1
print("The number of tags : %s"%(cnt))

print("########## Playlist ##########")
print(" Unique:: [Train]:%s  [Val]:%s  [Test]:%s  "%(len(train_playId_set), len(val_playId_set), len(test_playId_set) ))
print(" Max Id:: [Train]:%s  [Val]:%s  [Test]:%s  "%(max(train_playId_set), max(val_playId_set), max(test_playId_set) ))
print(" Min Id:: [Train]:%s  [Val]:%s  [Test]:%s  "%(min(train_playId_set), min(val_playId_set), min(test_playId_set) ))

print("\n########## Songs ##########")
print(" Unique:: [Train]:%s  [Val]:%s  [Test]:%s  "%(len(train_songId_set), len(val_songId_set), len(test_songId_set) ))
print(" Max Id:: [Train]:%s  [Val]:%s  [Test]:%s  "%(max(train_songId_set), max(val_songId_set), max(test_songId_set) ))
print(" Min Id:: [Train]:%s  [Val]:%s  [Test]:%s  "%(min(train_songId_set), min(val_songId_set), min(test_songId_set) ))

print("\n########## Tags ##########")
print(" Unique:: [Train]:%s  [Val]:%s  [Test]:%s  "%(len(train_tags_set), len(val_tags_set), len(test_tags_set) ))

print("\n########## New songs ##########")
print(" Unique:: [Train]:%s  [Public]:%s  [Final]:%s  "%(0, val_songcold_cnt, test_songcold_cnt))

print("\n########## New tags ##########")
print(" Unique:: [Train]:%s  [Public]:%s  [Final]:%s  "%(0, val_tagcold_cnt, test_tagcold_cnt))


playId2songIds = {}
playId2tags = {}
playId2tagIds = {}
playId2title = {}
playId2updt = {} 
playId2like = {}

songId2playIds = {}
tag2playIds = {}

train = pd.read_json('data_kakao/data/arena3_train.json', typ = 'frame')
val = pd.read_json('data_kakao/data/arena3_val.json',typ = 'frame')
test = pd.read_json('data_kakao/data/arena3_test.json',typ = 'frame')

train_temp = train[['id','songs']].to_numpy()
val_temp = val[['id','songs']].to_numpy()
test_temp = test[['id','songs']].to_numpy()
for playId, songIds in train_temp:
    playId2songIds[playId] = songIds
for playId, songIds in val_temp:
    playId2songIds[playId] = songIds
for playId, songIds in test_temp:
    playId2songIds[playId] = songIds
    

train_temp = train[['id','tags']].to_numpy()
val_temp = val[['id','tags']].to_numpy()
test_temp = test[['id','tags']].to_numpy()

for playId, tags in train_temp:
    playId2tags[playId] = tags
    
    if playId not in playId2tagIds:
        playId2tagIds[playId] = []
        
    for tag in tags:
        playId2tagIds[playId].append(tag2tagId[tag])
        
for playId, tags in val_temp:
    playId2tags[playId] = tags
    
    if playId not in playId2tagIds:
        playId2tagIds[playId] = []
    
    for tag in tags:
        playId2tagIds[playId].append(tag2tagId[tag])
        
for playId, tags in test_temp:
    playId2tags[playId] = tags
    
    if playId not in playId2tagIds:
        playId2tagIds[playId] = []
    
    for tag in tags:
        playId2tagIds[playId].append(tag2tagId[tag])

train_temp = train[['id','plylst_title' ]].to_numpy()
val_temp = val[['id','plylst_title']].to_numpy()
test_temp = test[['id','plylst_title']].to_numpy()

for playId, title in train_temp:
    playId2title[playId] = title
for playId, title in val_temp:
    playId2title[playId] = title
for playId, title in test_temp:
    playId2title[playId] = title


for playId in playId2songIds:
    for songId in playId2songIds[playId]:
        if songId in songId2playIds:
            songId2playIds[songId].append(playId)
        else:
            songId2playIds[songId] = [playId]
    
    for tag in playId2tags[playId]:
        if tag in tag2playIds:
            tag2playIds[tag].append(playId)
        else:
            tag2playIds[tag] = [playId]
            
train_temp = train[['id', 'updt_date']].to_numpy()
val_temp = val[['id','updt_date']].to_numpy()
test_temp = test[['id','updt_date']].to_numpy()

for playId, updt in train_temp:
    playId2updt[playId] = updt
for playId, updt in val_temp:
    playId2updt[playId] = updt
for playId, updt in test_temp:
    playId2updt[playId] = updt

train_temp = train[['id', 'like_cnt']].to_numpy()
val_temp = val[['id','like_cnt']].to_numpy()
test_temp = test[['id','like_cnt']].to_numpy()

for playId, like in train_temp:
    playId2like[playId] = like
for playId, like in val_temp:
    playId2like[playId] = like
for playId, like in test_temp:
    playId2like[playId] = like


songId2albumId = {}
songId2artistIds = {}

songId2artists = {} # 새로넣음

songId2name = {}
songId2gnrs = {}
songId2dtlgnrs = {}
songId2gnrIds = {}
songId2date = {}
songId2year = {}
songId2month = {}

albumId_check={}
artistId_check={}
gnrId_check={}
cnt=0

song_meta= pd.read_json('data_kakao/data/arena3_song_meta.json', typ = 'frame')

meta_temp = song_meta[['id','album_id']].to_numpy()
for songId,albumId in meta_temp:
    if albumId in albumId_check:
        songId2albumId[songId] = albumId_check[albumId]
    else:
        albumId_check[albumId] = cnt
        songId2albumId[songId] = albumId_check[albumId]
        cnt+=1
print("albumCnt:%s"%(cnt))

cnt=0
meta_temp = song_meta[['id','artist_id_basket']].to_numpy()
for songId,artistIds in meta_temp:
    for artistId in artistIds:
        if artistId in artistId_check:
            if songId in songId2artistIds:
                songId2artistIds[songId].append(artistId_check[artistId])
            else:
                songId2artistIds[songId] = [artistId_check[artistId]]
        else:
            artistId_check[artistId] = cnt
            if songId in songId2artistIds:
                songId2artistIds[songId].append(artistId_check[artistId])
            else:
                songId2artistIds[songId] = [artistId_check[artistId]]
            
            cnt+=1

print("artistCnt:%s"%(cnt))

meta_temp = song_meta[['id','song_name']].to_numpy()
for songId,song_name in meta_temp:
    songId2name[songId] = song_name
    
meta_temp = song_meta[['id','song_gn_gnr_basket']].to_numpy()
cnt=0
for songId,gnrs in meta_temp:
    songId2gnrs[songId] = gnrs
    for gnr in gnrs:
        if gnr not in gnrId_check:
            gnrId_check[gnr] = cnt
            cnt+=1
            
print("gnrCnt:%s"%(cnt))
for songId,gnrs in meta_temp:
    songId2gnrIds[songId] = []
    for gnr in gnrs:
        songId2gnrIds[songId].append(gnrId_check[gnr])

meta_temp = song_meta[['id','song_gn_dtl_gnr_basket']].to_numpy()
for songId,dtlgnrs in meta_temp:
    songId2dtlgnrs[songId] = dtlgnrs

meta_temp = song_meta[['id','issue_date']].to_numpy()
for songId,date in meta_temp:
    songId2date[songId] = date
    songId2year[songId] = int(date/10000)
    songId2month[songId] = int((date%10000)/100)
    
#
meta_temp = song_meta[['id', 'artist_name_basket']].to_numpy()
for songId,artists in meta_temp:
    songId2artists[songId] = artists
    
#mis
playId_w_tags = []
for tag in tag2playIds:
    for playId in tag2playIds[tag]:
        playId_w_tags.append(playId)
        
playId_w_tags = list(np.unique(np.array(playId_w_tags)))



#index set
with open('data_kakao/data/train_playId_set', 'wb') as f:
    pickle.dump(train_playId_set, f)
with open('data_kakao/data/public_playId_set', 'wb') as f:
    pickle.dump(val_playId_set, f)
with open('data_kakao/data/final_playId_set', 'wb') as f:
    pickle.dump(test_playId_set, f)
with open('data_kakao/data/train_songId_set', 'wb') as f:
    pickle.dump(train_songId_set, f)
with open('data_kakao/data/public_songId_set', 'wb') as f:
    pickle.dump(val_songId_set, f)
with open('data_kakao/data/final_songId_set', 'wb') as f:
    pickle.dump(test_songId_set, f)

    
'''
playId2songIds
playId2tags
playId2tagIds
playId2title
playId2updt
playId2like
songId2playIds
tag2playIds
tag2tagId
tagId2tag
'''

#Dictionary for data
with open('data_kakao/data/playId2songIds', 'wb') as f:
    pickle.dump(playId2songIds, f)
with open('data_kakao/data/playId2tags', 'wb') as f:
    pickle.dump(playId2tags, f)
with open('data_kakao/data/playId2tagIds','wb') as f:
    pickle.dump(playId2tagIds,f)
with open('data_kakao/data/playId2title','wb') as f:
    pickle.dump(playId2title,f)
with open('data_kakao/data/playId2updt','wb') as f:
    pickle.dump(playId2updt,f)
with open('data_kakao/data/playId2like','wb') as f:
    pickle.dump(playId2like,f)
    
with open('data_kakao/data/songId2playIds', 'wb') as f:
    pickle.dump(songId2playIds, f)
with open('data_kakao/data/tag2playIds', 'wb') as f:
    pickle.dump(tag2playIds, f)
with open('data_kakao/data/tag2tagId','wb') as f:
    pickle.dump(tag2tagId,f)
with open('data_kakao/data/tagId2tag','wb') as f:
    pickle.dump(tagId2tag,f)

'''
songId2albumId
songId2artistIds
songId2artists
songId2name
songId2gnrs
songId2dtlgnrs
songId2gnrIds
songId2date
songId2year
songId2month

'''
#Dictionary for meta data (Song)
with open('data_kakao/data/songId2albumId', 'wb') as f:
    pickle.dump(songId2albumId, f)
with open('data_kakao/data/songId2artistIds', 'wb') as f:
    pickle.dump(songId2artistIds, f)
with open('data_kakao/data/songId2artists','wb') as f:
    pickle.dump(songId2artists, f)
with open('data_kakao/data/songId2name', 'wb') as f:
    pickle.dump(songId2name, f)
with open('data_kakao/data/songId2gnrs', 'wb') as f:
    pickle.dump(songId2gnrs, f)
with open('data_kakao/data/songId2dtlgnrs', 'wb') as f:
    pickle.dump(songId2dtlgnrs, f)
with open('data_kakao/data/songId2gnrIds','wb') as f:
    pickle.dump(songId2gnrIds,f)
    
with open('data_kakao/data/songId2date', 'wb') as f:
    pickle.dump(songId2date, f)

with open('data_kakao/data/songId2year','wb') as f:
    pickle.dump(songId2year, f)
with open('data_kakao/data/songId2month','wb') as f:
    pickle.dump(songId2month,f)
    
    
# mis
with open('data_kakao/data/playId_w_tags','wb') as f:
    pickle.dump(playId_w_tags,f)
