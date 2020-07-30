# for CIKMCup raw data process

import os
import gzip
import pandas as pd
import numpy as np
import os,sys
import random
import numpy as np
import json
import operator
import pandas as pd

datapath = '../Data/CIKMCup_raw/'
data_records = 'train-queries.csv'
data_items_attribute = 'products.csv'
data_items_category = 'product-categories.csv'

df_train_queries = pd.read_csv(datapath+data_records, header=0, sep=';')
df_train_queries.dropna(subset=['userId'],inplace=True)
df_train_queries.dropna(subset=['searchstring.tokens'],inplace=True)
query_map = {}
also_view = {}
with open(datapath+data_records+'.processed', 'w') as fout:
    for index, row in df_train_queries.iterrows():
        userId = row['userId']
        queryId = row['queryId']
        searchsring = row['searchstring.tokens']
        eventdate = row['eventdate']
        items = row['items'].strip().split(',')
        if queryId not in query_map:
            query_map[queryId] = searchsring
        items_ = items.copy()
        for item in items:
            if item not in also_view:
                also_view[item] = set()
            items_ = set(items) - set(item)
            also_view[item] = also_view[item] | items_
            list_write = [str(userId), str(queryId), str(eventdate), str(item)]
            #print(list_write)
            fout.write(';'.join(list_write) + '\n')

with open(datapath+data_records+'.queries', 'w') as fout:
    for key, item in query_map.items():
        fout.write(str(key) + ';' + str(item) + '\n')


df_records = pd.read_csv(datapath+data_records+'.processed', header=None, names=['userId', 'queryId', 'eventdate', 'itemId'], sep=';')
df_queries = pd.read_csv(datapath+data_records+'.queries', header=None, names=['queryId', 'searchstring'], sep=';')
df_items_attributes = pd.read_csv(datapath+data_items_attribute, header=0, sep=';')

df_items_category = pd.read_csv(datapath+data_items_category, header=0, sep=';')
df_items_attributes = pd.merge(df_items_attributes, df_items_category, on='itemId')

df_records['itemId'] = df_records['itemId'].astype('int')
df_items_attributes['itemId'] = df_items_attributes['itemId'].astype('int')

df_items_attributes = df_items_attributes[df_items_attributes['itemId'].isin(df_records['itemId'])]
df_records = df_records[df_records['itemId'].isin(df_items_attributes['itemId'])]

df_items_attributes = df_items_attributes[['itemId', 'product.name.tokens', 'categoryId']]
df_items_attributes.to_csv(datapath+data_items_attribute+'.attributes', header=True, index=False, encoding='utf-8', sep=';')

# index and filter
user_list = df_records['userId'].unique().tolist()
item_list = df_records['itemId'].unique().tolist()
query_list = df_queries['queryId'].unique().tolist()

searchstring_term = df_queries['searchstring'].unique().tolist()

productname_term = df_items_attributes['product.name.tokens'].unique().tolist()

terms = searchstring_term+productname_term

term_string = ','.join(terms)
words_list = list(set(term_string.split(',')))

# output also review i-i
also_view_ = {}
for key, value in also_view.items():
    if int(key) in item_list:
        value_ = []
        for i in value:
            if int(i) in item_list:
                value_.append(i)
        also_view_[int(key)] = value_

# output word, user, product indexes
with open(datapath + 'vocabs.txt', 'w') as fout:
    fout.write('\n'.join(list(map(str, words_list))))

with open(datapath + 'users.txt', 'w') as fout:
    fout.write('\n'.join(list(map(str, user_list))))

with open(datapath + 'items.txt', 'w') as fout:
    fout.write('\n'.join(list(map(str, item_list))))

with open(datapath + 'queries_id.txt', 'w') as fout:
    fout.write('\n'.join(list(map(str, query_list))))

with open(datapath + 'also_view_i_i.txt', 'w') as fout:
    for i in item_list:
        items = also_view_[i]
        fout.write(','.join(items) + '\n')

#read and output indexed interaction

def index_set(s):
    i = 0
    s_map = {}
    for key in s:
        s_map[key] = str(i)
        i += 1
    return s_map

def map_fun(x):
    return word_map[x]

word_map = index_set(words_list)
user_map = index_set(user_list)
query_map = index_set(query_list)
item_map = index_set(item_list)
user_interaction_seq = {} # recording the sequence of user interaction in time


with open(datapath + 'interaction_text.txt', 'w') as fout_text, open(datapath + 'interaction_u_q_i.txt', 'w') as fout_u_q_i:
    with open(datapath + 'interaction_id.txt', 'w') as fout_id:
        index = 0
        for index, row in df_records.iterrows():
            userId = row['userId']
            itemId = row['itemId']
            queryId = row['queryId']
            eventdate = row['eventdate']
            relevant_text = df_items_attributes.loc[df_items_attributes['itemId']==itemId, 'product.name.tokens'].tolist()[0] + ','+ df_queries.loc[df_queries['queryId']==queryId, 'searchstring'].tolist()[0]
            relevent_text_list = relevant_text.strip().split(',')
            relevent_text_list_map = list(map(map_fun, relevent_text_list))
            fout_text.write(','.join(relevent_text_list_map))
            if userId not in user_interaction_seq:
                user_interaction_seq[userId] = []
            user_interaction_seq[userId].append((index, eventdate))
            fout_text.write('\n')
            fout_u_q_i.write(user_map[userId] + ',' + query_map[queryId] + ',' + item_map[itemId] + '\n')
            fout_id.write('line_' + str(index) + '\n')
            index += 1

# Sort each user's interactions according to time and output to files

interaction_loc_time_list = [[] for _ in range(index)]
with open(datapath + 'u_qi_seq.txt', 'w') as fout:
    for userId in user_list:
        interaction_time_list = user_interaction_seq[userId]
        user_interaction_seq[userId] = sorted(interaction_time_list, key=operator.itemgetter(1))
        fout.write(','.join([str(x[0]) for x in user_interaction_seq[userId]]) + '\n')
        for i in range(len(user_interaction_seq[userId])):
            interaction_id = user_interaction_seq[userId][i][0]
            time = user_interaction_seq[userId][i][1]
            interaction_loc_time_list[interaction_id] = [i, time]

with open(datapath + 'interaction_loc_and_time.txt', 'w') as fout:
    for t_l in interaction_loc_time_list:
        fout.write(' '.join([str(x) for x in t_l]) + '\n')


#MataData
df_queries = pd.read_csv(datapath+data_records+'.queries', header=None, names=['queryId', 'searchstring'], sep=';')
df_items_attributes = pd.read_csv(datapath+data_items_attribute+'.attributes', header=0, sep=';')

with open(datapath + 'vocabs.txt', 'r') as fin:
    words_list = fin.read().splitlines()

with open(datapath + 'items.txt', 'r') as fin:
    item_list = fin.read().splitlines()
item_list = list(map(int, item_list))
with open(datapath + 'queries_id.txt', 'r') as fin:
    query_list = fin.read().splitlines()
query_list = list(map(int, query_list))
word_map = dict([(words_list[i], str(i)) for i in range(len(words_list))])

def map_fun(x):
    return word_map[x]

with open(datapath + 'item_des.txt', 'w') as fout:
    for i in item_list:
        des = df_items_attributes.loc[df_items_attributes['itemId']==i, 'product.name.tokens'].tolist()[0]
        des_list = des.strip().split(',')
        des_map = list(map(map_fun, des_list))
        fout.write(','.join(des_map) + '\n')

with open(datapath+'queries.txt', 'w') as fout:
    for i in query_list:
        que = df_queries.loc[df_queries['queryId']==i, 'searchstring'].tolist()[0]
        que_list = que.strip().split(',')
        que_map = list(map(map_fun, que_list))
        fout.write(','.join(que_map) + '\n')


# knowledge
category_list = df_items_attributes['categoryId'].unique().tolist()
category_indexes = dict([(category_list[i], str(i)) for i in range(len(category_list))])

with open(datapath + 'item_category.txt', 'w') as fout:
    for i in item_list:
        i_category = df_items_attributes.loc[df_items_attributes['itemId']==i, 'categoryId'].tolist()[0]
        fout.write(category_indexes[i_category] + '\n')

with open(datapath + 'category.txt', 'w') as f:
    f.write('\n'.join(list(map(str, category_list))))

# sequentially split

output_path = datapath + 'seq_query_split/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

#read user-interaction sequence
user_interaction_seq = []
with open(datapath + 'u_qi_seq.txt', 'r') as fin:
    for line in fin:
        arr = line.strip().split(',')
        user_interaction_seq.append([int(x) for x in arr])

#Genrate train/valid/test sets
interaction_sample_rate_test = 0.2
interaction_sample_rate_valid = 0.1
test_interaction_idx = set()
valid_interaction_idx = set()
for interaction_seq in user_interaction_seq:
    if len(interaction_seq) > 20:
        test_sample_num = int(interaction_sample_rate_test * len(interaction_seq))
        valid_sample_num = int(interaction_sample_rate_valid * len(interaction_seq))
        test_interaction_idx = test_interaction_idx.union(set(interaction_seq[-test_sample_num:]))
        valid_interaction_idx = valid_interaction_idx.union(set(interaction_seq[-test_sample_num-valid_sample_num:-test_sample_num]))

print('test size: %d, valid size: %d'%(len(test_interaction_idx), len(valid_interaction_idx)))
#output train/test interaction data
with open(output_path+'train_1.txt', 'w') as train_fout, open(output_path+'test.txt', 'w') as test_fout, open(output_path+'valid.txt', 'w') as valid_fout, open(output_path+'train.txt', 'w') as tain_2_fout:
    with open(datapath+'interaction_u_q_i.txt', 'r') as info_fin, open(datapath+'interaction_text.txt', 'r') as text_fin, open(datapath+'interaction_id.txt', 'r') as id_fin:
        info_line = info_fin.readline()
        text_line = text_fin.readline()
        id_line = id_fin.readline()
        index = 0
        while info_line:
            arr = info_line.strip().split(',')
            if index in test_interaction_idx:
                test_fout.write(arr[0] + ';' + arr[1] + ';' + arr[2] + ';' + text_line.strip() + ';' + str(id_line.strip()) + '\n')
            elif index in valid_interaction_idx:
                valid_fout.write(arr[0] + ';' + arr[1] + ';' + arr[2] + ';' + text_line.strip() + ';' + str(id_line.strip()) + '\n')
            else:
                train_fout.write(arr[0] + ';' + arr[1] + ';' + arr[2] + ';' + text_line.strip() + ';' + str(id_line.strip()) + '\n')
            if index not in test_interaction_idx:
                tain_2_fout.write(arr[0] + ';' + arr[1] + ';' + arr[2] + ';' + text_line.strip() + ';' + str(id_line.strip()) + '\n')
            index += 1
            info_line = info_fin.readline()
            text_line = text_fin.readline()
            id_line = id_fin.readline()
