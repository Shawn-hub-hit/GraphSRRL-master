import os, sys
import gzip
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
#from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import json
import re
import string

datapath = '../Data/amazon_electronics_dataset/'
amazon_term = 'Electronics'
interaction_file = datapath + 'reviews_'+amazon_term+'_5.json.gz'

def PreProcess_text(t):
    t = t.lower()
    # 去除数字
    t = re.sub(r'\d+', '', t)
    # 去除标点和一些符号,末尾空格
    regex = re.compile('[\s+\.\-\\\!\/_,$%^*(+\"\')]+|[+——()?=:;|【】“”！，。？、~@#￥%……&*（）]+')
    t = regex.sub(' ', t)
    t = t.strip()

    # 去除stopwords
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words | set(' ')
    # print(t)
    tokens = word_tokenize(t)
    stemmer_porter = SnowballStemmer("english")
    result = [stemmer_porter.stem(i) for i in tokens if not i in stop_words]
    #print(result)
    return ' '.join(result)
    #return t


with gzip.open(interaction_file, 'r') as g, open(datapath+'interactions_'+amazon_term+'.processed', 'w') as f:  # 对每个interactions的句子进行预处理
    for l in g:
        l = eval(l)
        l['userId'] = l.pop('reviewerID')
        l['itemId'] = l.pop('asin')
        #l['userName'] = l.pop('reviewerName')
        l['interactionText'] = l.pop('reviewText')
        l['eventdate'] = l.pop('unixReviewTime')
        l['interactionTime'] = l.pop('reviewTime')
        interaction_text = PreProcess_text(l['interactionText'])
        summary = PreProcess_text(l['summary'])
        l['interactionText'] = interaction_text
        l['summary'] = summary
        f.write(str(l['userId']) + ';' + str(l['itemId']) + ';' + str(l['eventdate']) + ';' + l['interactionText'] + ';' + l['summary'] + '\n')
        
        
## item attributes
# 提取query,然后将query和产品匹配，得到u-q-p对应 Done
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
#from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import json
import re
import string
import gzip


def PreProcess_text(t):
    t = t.lower()
    # 去除数字
    t = re.sub(r'\d+', '', t)
    # 去除标点和一些符号,末尾空格
    regex = re.compile('[\s+\.\-\\\!\/_,$%^*(+\"\')]+|=[+——()?:;|【】“”！，。？、~@#￥%……&*（）]+')
    t = regex.sub(' ', t)
    t = t.strip()

    # 去除stopwords
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words | set(' ')
    # print(t)
    tokens = word_tokenize(t)
    stemmer_porter = SnowballStemmer("english")
    result = [stemmer_porter.stem(i) for i in tokens if not i in stop_words]
    # print(result)
    return ' '.join(result)
    #return t


datapath = '../Data/amazon_electronics_dataset/'
amazon_term = 'Electronics'
meta_file = datapath+'meta_'+amazon_term+'.json.gz'
indexed_review_path = datapath+'min_count5/'


#./meta_Electronics.json.gz ./amazon_Electronics_index_dataset/min_count5/
with gzip.open(meta_file, 'r') as g, open(datapath+'item_attribtes.preprocess', 'w') as fout: 
    for l in g:
        l = eval(l)
        des = ''
        if 'title' in l.keys():
            des = des + l['title']
            des = des + " "
        # print(des)
        l['Des'] = PreProcess_text(des)
        # print(getIndexesString(des))
        query_ = ' '.join([' '.join(i) for i in l['categories']])
        query_ = PreProcess_text(query_)
        query_ = ' '.join(list(set(query_.split(' '))))
        l['Query'] = query_
        l['Categories'] = query_.split(' ')[0]
        if 'related' in l.keys():
            if 'also_viewed' in l['related']: 
                l['Also_viewed'] = ','.join(l['related']['also_viewed'])
            else:
                l['Also_viewed'] = ''
        else:
            l['Also_viewed'] = ''
        fout.write(str(l['asin']) + ';' + l['Query'] + ';' + l['Des'] + ';' + l['Categories'] + ';' + l['Also_viewed'] + '\n')

        
        
import pandas as pd
datapath = '../Data/amazon_electronics_dataset/'
amazon_term = 'Electronics'

df_records = pd.read_csv(datapath+'interactions_'+amazon_term+'.processed', header=None, names=['userId', 'itemId', 'eventdate', 'interactionText', 'summary'], sep=';')

df_items_attributes = pd.read_csv(datapath+'item_attribtes.preprocess', header=None, names=['itemId', 'searchstring', 'des', 'categories', 'also_viewed'], sep=';')

#print(df_items_attributes.head())
#df_items_category = pd.read_csv(datapath+data_items_category, header=0, sep=';')
#print(df_items_category.head())
#df_items_attributes = pd.merge(df_items_attributes, df_items_category, on='itemId')
#print(df_items_attributes.head())

df_records['itemId'] = df_records['itemId'].astype('str')
df_items_attributes['itemId'] = df_items_attributes['itemId'].astype('str')

print(df_items_attributes.shape, df_records.shape)
df_items_attributes = df_items_attributes[df_items_attributes['itemId'].isin(df_records['itemId'])]
df_records = df_records[df_records['itemId'].isin(df_items_attributes['itemId'])]
df_items_attributes.drop_duplicates(subset=['itemId'], inplace=True)
print(df_items_attributes.shape, df_records.shape)

#print(df_items_attributes.head(10), df_records.head())

df_items_attributes.to_csv(datapath+amazon_term+'.attributes', header=True, index=False, encoding='utf-8', sep=';')
df_records.to_csv(datapath+amazon_term+'.records', header=True, index=False, encoding='utf-8', sep=';')

# output the queries
df_i_q = df_items_attributes[['itemId', 'searchstring']]
#print(df_i_q.head(1))
df_records = pd.merge(df_records, df_i_q, on='itemId')

query_string_list = df_items_attributes['searchstring'].unique().tolist()

query_map = dict([(query_string_list[i], i) for i in range(len(query_string_list))])

def mapfun(x):
    return query_map[x]

querykey = df_records['searchstring'].apply(mapfun)
df_records['queryId'] = querykey
df_records = df_records[['userId', 'queryId', 'itemId', 'eventdate', 'interactionText', 'summary']]
df_items_attributes = df_items_attributes[['itemId', 'des', 'categories', 'also_viewed']]
with open(datapath+amazon_term+'.queries', 'w') as fout:
    for string, key in query_map.items():
        fout.write(str(key) + ';' + str(string) + '\n')


import numpy as np
df_queries = pd.read_csv(datapath+amazon_term+'.queries', header=None, names=['queryId', 'searchstring'], sep=';')
#print(df_queries.head())
user_list = df_records['userId'].unique().tolist()
item_list = df_records['itemId'].unique().tolist()
query_list = df_queries['queryId'].unique().tolist()

searchstring_term = df_queries['searchstring'].unique().tolist()

df_items_attributes['des'] = df_items_attributes['des'].fillna(' ')
item_term = df_items_attributes['des'].unique().tolist()
item_term = list(filter((' ').__ne__, item_term))
df_records['interactionText'] = df_records['interactionText'].fillna(' ')
interaction_text = df_records['interactionText'].unique().tolist()
interaction_text = list(filter((' ').__ne__, interaction_text))
terms = searchstring_term+item_term+interaction_text
#print(terms[:10])

term_string = ' '.join(terms)
words_list = list(set(term_string.split(' ')))
print(words_list[:5])


# output also review i-i

also_view_df = df_items_attributes[['itemId', 'also_viewed']]
also_view = also_view_df.groupby('itemId')['also_viewed'].apply(lambda x: str(x.tolist()[0])).to_dict()
also_view_ = {}
for key, value in also_view.items():
    if key in item_list:
        value_ = []
        for i in value.split(','):
            if i in item_list:
                value_.append(i)
        also_view_[key] = value_
        
        
with open(datapath + 'vocabs.txt','w') as fout:
    fout.write('\n'.join(list(map(str, words_list))))

with open(datapath + 'users.txt','w') as fout:
    fout.write('\n'.join(list(map(str, user_list))))

with open(datapath + 'items.txt','w') as fout:
    fout.write('\n'.join(list(map(str, item_list))))
    
with open(datapath + 'queries_id.txt','w') as fout:
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
            #relevant_text = df_items_attributes.loc[df_items_attributes['itemId']==itemId, 'des'].tolist()[0] + ','+ df_queries.loc[df_queries['queryId']==queryId, 'searchstring'].tolist()[0]
            relevant_text = row['interactionText']
            relevent_text_list = relevant_text.strip().split(' ')
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
import operator
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
import pandas as pd
datapath = '../Data/amazon_electronics_dataset/'
amazon_term = 'Electronics'
df_queries = pd.read_csv(datapath+amazon_term+'.queries', header=None, names=['queryId', 'searchstring'], sep=';')
df_items_attributes = pd.read_csv(datapath+amazon_term+'.attributes', header=0, sep=';')
#print(df_items_attributes.head())

with open(datapath + 'vocabs.txt', 'r') as fin:
    words_list = fin.read().splitlines()

with open(datapath + 'items.txt', 'r') as fin:
    item_list = fin.read().splitlines()

with open(datapath + 'queries_id.txt', 'r') as fin:
    query_list = fin.read().splitlines()
query_list = list(map(int, query_list))
word_map = dict([(words_list[i], str(i)) for i in range(len(words_list))])
def map_fun(x):
    return word_map[x]

with open(datapath + 'item_des.txt', 'w') as fout:
    for i in item_list:
        des = df_items_attributes.loc[df_items_attributes['itemId']==i, 'des'].tolist()[0]
        des_list = str(des).strip().split(' ')
        des_map = list(map(map_fun, des_list))
        fout.write(','.join(des_map) + '\n')

with open(datapath+'queries.txt', 'w') as fout:
    for i in query_list:
        que = df_queries.loc[df_queries['queryId']==i, 'searchstring'].tolist()[0]
        que_list = que.strip().split(' ')
        que_map = list(map(map_fun, que_list))
        fout.write(','.join(que_map) + '\n')
        
        
# knowledge
#print(df_items_attributes.head())
category_list = df_items_attributes['categories'].unique().tolist()
category_indexes = dict([(category_list[i], str(i)) for i in range(len(category_list))])

with open(datapath + 'item_category.txt', 'w') as fout:
    for i in item_list:
        i_category = df_items_attributes.loc[df_items_attributes['itemId']==i, 'categories'].tolist()[0]
        fout.write(category_indexes[i_category] + '\n')

with open(datapath + 'category.txt', 'w') as f:
    f.write('\n'.join(list(map(str, category_list))))
    
    
# sequentially split
import os,sys
import random
import numpy as np
import json
datapath = '../Data/amazon_electronics_dataset/'
amazon_term = 'Electronics'


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
    if len(interaction_seq) > 10:
        test_sample_num = int(interaction_sample_rate_test * len(interaction_seq))
        valid_sample_num = int(interaction_sample_rate_valid * len(interaction_seq))
        test_interaction_idx = test_interaction_idx.union(set(interaction_seq[-test_sample_num:]))
        valid_interaction_idx = valid_interaction_idx.union(set(interaction_seq[-test_sample_num-valid_sample_num:-test_sample_num]))
        
print(len(test_interaction_idx))
print(len(valid_interaction_idx))
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
        
