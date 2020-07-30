import numpy as np
import json
import random
import gzip
import math
from six.moves import xrange
import os

class GENPSDataset:
    def __init__(self, datapath, input_dir, set_name='train'):
        #get item/user/vocabulary information
        with open(datapath + 'items.txt', 'r') as fin:
            self.item_ids = fin.read().splitlines()
        self.item_size = len(self.item_ids)

        with open(datapath + 'users.txt', 'r') as fin:
            self.user_ids = fin.read().splitlines()
        self.user_size = len(self.user_ids)

        with open(datapath + 'vocabs.txt', 'r') as fin:
            self.words = fin.read().splitlines()

        with open(datapath + 'queries_id.txt', 'r') as fin:
            self.query_list = fin.read().splitlines()
        self.query_size = len(self.query_list)

        self.vocab_size = len(self.words)
        self.query_words = []
        self.query_max_length = 0
        with open(datapath + 'queries.txt', 'r') as fin:
            for line in fin:
                words = [int(i) for i in line.strip().split(',')]
                if len(words) > self.query_max_length:
                    self.query_max_length = len(words)
                self.query_words.append(words)
        # pad
        for i in xrange(len(self.query_words)):
            self.query_words[i] = [self.vocab_size for j in xrange(self.query_max_length - len(self.query_words[i]))] + \
                                  self.query_words[i]

        self.word_map = dict([(self.words[i], str(i)) for i in range(len(self.words))])
        self.user_map = dict([(self.user_ids[i], str(i)) for i in range(len(self.user_ids))])
        self.item_map = dict([(self.item_ids[i], str(i)) for i in range(len(self.item_ids))])

        #get interaction sets
        self.word_count = 0
        self.vocab_distribute = np.zeros(self.vocab_size)
        self.interaction_info = []
        self.interaction_text = []
        with open(input_dir + '/' + set_name + '.txt', 'r') as fin:
            for line in fin:
                arr = line.strip().split(';')
                self.interaction_info.append((int(arr[0]), int(arr[1]), int(arr[2])))  # (user_idx, query_idx, item_idx)
                self.interaction_text.append([int(i) for i in arr[3].split(',')])
                for idx in self.interaction_text[-1]:
                    self.vocab_distribute[idx] += 1
                self.word_count += len(self.interaction_text[-1])
        self.interaction_size = len(self.interaction_info)
        self.vocab_distribute = self.vocab_distribute.tolist()
        self.sub_sampling_rate = None
        self.interaction_distribute = np.ones(self.interaction_size).tolist()
        self.item_distribute = np.ones(self.item_size).tolist()

        self.user_interaction_idxs = []
        self.user_qi_idxs = []
        with open(os.path.join(datapath, "u_qi_seq.txt"), 'r') as fin:
            for line in fin:
                interactions = [int(_) for _ in line.strip().split(",")]
                self.user_interaction_idxs.append(interactions)
                self.user_qi_idxs.append([-1 for _ in range(len(self.user_interaction_idxs[-1]))])


        with open(os.path.join(datapath, "interaction_u_q_i.txt"), 'rt') as fin:
            interaction_idx = 0
            for line in fin:
                triple_ = line.rstrip().split(",")
                user_idx, query_idx, item_idx = int(triple_[0]), int(triple_[1]), int(triple_[2])
                if interaction_idx in self.user_interaction_idxs[user_idx]:
                    pos = self.user_interaction_idxs[user_idx].index(interaction_idx)
                    self.user_qi_idxs[user_idx][pos] = (query_idx, item_idx)
                else:
                    print("Error: We cannot find the corresponding items")
                    #print(interaction_idx, self.user_interaction_idxs[user_idx])
                    pass
                interaction_idx += 1

        self.max_history_length = 0
        for qi_idxs in self.user_qi_idxs:
            if self.max_history_length < len(qi_idxs):
                self.max_history_length = len(qi_idxs)

        print("Data statistic: vocab %d, user %d, item %d, query %d, max history length %d, train_size %d\n" % (self.vocab_size,
                    self.user_size, self.item_size, len(self.query_words), self.max_history_length, len(self.interaction_info)))

    def get_test_dataset(self, input_dir):
        self.word_count_test = 0
        self.vocab_distribute_test = np.zeros(self.vocab_size)
        self.interaction_info_test = []
        self.interaction_text_test = []
        with open(input_dir + '/' + 'test.txt', 'r') as fin:
            for line in fin:
                arr = line.strip().split(';')
                self.interaction_info_test.append((int(arr[0]), int(arr[1]), int(arr[2]))) # (user_idx, item_idx)
                self.interaction_text_test.append([int(i) for i in arr[3].split(',')])
                for idx in self.interaction_text_test[-1]:
                    self.vocab_distribute_test[idx] += 1
                self.word_count_test += len(self.interaction_text_test[-1])
        self.interaction_size_test = len(self.interaction_info_test)
        self.vocab_distribute_test = self.vocab_distribute_test.tolist()
        self.sub_sampling_rate_test = None
        self.interaction_distribute_test = np.ones(self.interaction_size_test).tolist()
        self.item_distribute = np.ones(self.item_size).tolist()

        print("Test data statistic: test_size %d\n" % (len(self.interaction_info_test)))
