#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import random

from torch.utils.data import Dataset


class TrainDatasetPS(Dataset):
    def __init__(self, triples, nuser, nitem, nquery, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nuser = nuser
        self.nitem = nitem
        self.nquery = nquery
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(triples)
        self.true_tail = self.get_true_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, query, tail = positive_sample
        sample = []
        sample.append([head, query, tail, 1])

        subsampling_weight = self.count[(head, query)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            # negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            negative_sample = np.random.randint(self.nitem, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.true_tail[(head, query)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        for i in negative_sample:
            sample.append([head, query, i, 0])

        sample = torch.LongTensor(sample)
        uids = sample[:, 0]
        queries = sample[:, 1]
        items = sample[:, 2]
        labels = sample[:, 3]
        return uids, queries, items, labels, subsampling_weight

    @staticmethod
    def collate_fn(data):
        uids = torch.stack([_[0] for _ in data], dim=0)
        queries = torch.stack([_[1] for _ in data], dim=0)
        items = torch.stack([_[2] for _ in data], dim=0)
        labels = torch.stack([_[3] for _ in data], dim=0)
        subsampling_weight = torch.stack([_[4] for _ in data], dim=0)
        return uids, queries, items, labels, subsampling_weight

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, query) or (query, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, query, tail in triples:
            if (head, query) not in count:
                count[(head, query)] = start
            else:
                count[(head, query)] += 1

        return count

    @staticmethod
    def get_true_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_tail = {}

        for head, query, tail in triples:
            if (head, query) not in true_tail:
                true_tail[(head, query)] = []
            true_tail[(head, query)].append(tail)

        for head, query in true_tail:
            true_tail[(head, query)] = np.array(list(set(true_tail[(head, query)])))

        return true_tail


class TrainDatasetKG(Dataset):
    def __init__(self, triples, nuser, nitem, nquery, negative_sample_size, mode='tail_batch'):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nuser = nuser
        self.nitem = nitem
        self.nquery = nquery
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_tail, self.true_head, self.true_query = self.get_true_head_query_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, query, tail = positive_sample


        subsampling_weight = self.count[(head, query)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nitem, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(query, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == self.mode == 'tail-company-batch' or self.mode == 'head-company-batch' or self.mode == 'query-company-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, query)], 
                    assume_unique=True, 
                    invert=True
                )

            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
        #print(self.true_tail[(head, query)])
        if len(list(self.true_tail[(head, query)])) > 0:
            true_tail_company = torch.LongTensor([random.choice(list(self.true_tail[(head, query)]))])
        else:
            true_tail_company = torch.LongTensor([tail])
        if len(list(self.true_head[(query, tail)])) > 0:
            true_head_company = torch.LongTensor([random.choice(list(self.true_head[(query, tail)]))])
        else:
            true_head_company = torch.LongTensor([head])
        if len(list(self.true_query[(head, tail)])) > 0:
            true_query_company = torch.LongTensor([random.choice(list(self.true_query[(head, tail)]))])
        else:
            true_query_company = torch.LongTensor([query])


        return positive_sample, negative_sample, subsampling_weight, self.mode, true_tail_company, true_head_company, true_query_company
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        true_tail_company = torch.stack([_[4] for _ in data], dim=0)
        true_head_company = torch.stack([_[5] for _ in data], dim=0)
        true_query_company = torch.stack([_[6] for _ in data], dim=0)
        return positive_sample, negative_sample, subsample_weight, mode, true_tail_company, true_head_company, true_query_company
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, query) or (query, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, query, tail in triples:
            if (head, query) not in count:
                count[(head, query)] = start
            else:
                count[(head, query)] += 1

        return count
    
    @staticmethod
    def get_true_head_query_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}
        true_query = {}

        for head, query, tail in triples:
            if (head, query) not in true_tail:
                true_tail[(head, query)] = []
            true_tail[(head, query)].append(tail)
            if (query, tail) not in true_head:
                true_head[(query, tail)] = []
            true_head[(query, tail)].append(head)
            if (head, tail) not in true_query:
                true_query[(head, tail)] = []
            true_query[(head, tail)].append(query)

        for query, tail in true_head:
            true_head[(query, tail)] = np.array(list(set(true_head[(query, tail)])))
        for head, query in true_tail:
            true_tail[(head, query)] = np.array(list(set(true_tail[(head, query)])))
        for head, tail in true_query:
            true_query[(head, tail)] = np.array(list(set(true_query[(head, tail)])))

        return true_tail, true_head, true_query

    
class TestDatasetPS(Dataset):
    def __init__(self, triples, all_true_triples, nuser, nitem, nquery):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nuser = nuser
        self.nitem = nitem
        self.nquery = nquery
        self.true_tail = self.get_true_tail(self.triples)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, query, tail = self.triples[idx]

        tmp1 = range(self.nitem)
        tmp2 = np.zeros(self.nitem)
        for t in self.true_tail[(head, query)]:
            idx = tmp1.index(t)
            tmp2[idx] = -1
        idx = tmp1.index(tail)
        tmp2[idx] = 0

        tmp1 = torch.LongTensor(tmp1)
        tmp2 = torch.LongTensor(tmp2)
        filter_bias = tmp2.float()
        negative_tails = tmp1
        uids = torch.LongTensor([head]*self.nitem)
        quries = torch.LongTensor([query]*self.nitem)
        positive_sample = torch.LongTensor((head, query, tail))

        return positive_sample, uids, quries, negative_tails, filter_bias
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        uids = torch.stack([_[1] for _ in data], dim=0)
        quries = torch.stack([_[2] for _ in data], dim=0)
        negative_tails = torch.stack([_[3] for _ in data], dim=0)
        filter_bias = torch.stack([_[4] for _ in data], dim=0)

        return positive_sample, uids, quries, negative_tails, filter_bias

    @staticmethod
    def get_true_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_tail = {}

        for head, query, tail in triples:
            if (head, query) not in true_tail:
                true_tail[(head, query)] = []
            true_tail[(head, query)].append(tail)

        for head, query in true_tail:
            true_tail[(head, query)] = np.array(list(set(true_tail[(head, query)])))

        return true_tail


    
class OneShotIterator(object):
    def __init__(self, dataloader_tail_company, dataloader_head_company, dataloader_query_company):

        self.iterator_tail_company = self.one_shot_iterator(dataloader_tail_company)
        self.iterator_head_company = self.one_shot_iterator(dataloader_head_company)
        self.iterator_query_company = self.one_shot_iterator(dataloader_query_company)
        self.step = 0
        
    def next(self):

        if self.step % 3 == 0:
            data = next(self.iterator_tail_company)
        elif self.step % 3 == 1:
            data = next(self.iterator_head_company)
        else:
            data = next(self.iterator_query_company)
        self.step += 1
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class OneShotIterator2(object):
    def __init__(self, dataloader_tail_company, dataloader_query_company):

        self.iterator_tail_company = self.one_shot_iterator(dataloader_tail_company)
        self.iterator_query_company = self.one_shot_iterator(dataloader_query_company)
        self.step = 0

    def next(self):

        if self.step % 2 == 0:
            data = next(self.iterator_tail_company)
        else:
            data = next(self.iterator_query_company)
        self.step += 1
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data