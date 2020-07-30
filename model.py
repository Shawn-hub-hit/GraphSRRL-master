#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, output_dim, bias=True),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        return self.aggre(x)


def init_embedding(n_vectors, dim):
    """Create a torch.nn.Embedding object with `n_vectors` samples and `dim` dimensions.
    """
    entity_embeddings = nn.Embedding(n_vectors, dim)
    entity_embeddings.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(n_vectors, dim))), requires_grad=True)
    return entity_embeddings

def normalize(embeddings):

    embeddings.weight.data = F.normalize(embeddings.weight.data, p=2, dim=1)


class KGPSModel(nn.Module):
    def __init__(self, model_name, nuser, nitem, nquery, hidden_dim, dataset, device):
        super(KGPSModel, self).__init__()
        self.model_name = model_name
        self.nuser = nuser
        self.nitem = nitem
        self.nquery = nquery
        self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.device = device
        self.vocab_size = self.dataset.vocab_size
        self.query_max_length = self.dataset.query_max_length
        self.query_words_idx = torch.LongTensor(self.dataset.query_words)
        self.query_net_struct = 'RNN'

        self.word_embedding = init_embedding(self.vocab_size+1, self.hidden_dim)
        normalize(self.word_embedding)
        #for PS

        self.user_embedding_PS = init_embedding(nuser, self.hidden_dim)
        normalize(self.user_embedding_PS)
        self.item_embedding_PS = init_embedding(nitem, self.hidden_dim)
        normalize(self.item_embedding_PS)

        #for KG
        self.user_embedding_KG = init_embedding(nuser, self.hidden_dim)
        normalize(self.user_embedding_KG)
        self.item_embedding_KG = init_embedding(nitem, self.hidden_dim)
        normalize(self.item_embedding_KG)

        self.query_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.query_rnn = nn.RNN(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.query_aggre = _Aggregation(self.hidden_dim * 2, self.hidden_dim)

        self.kg_aggre_head = _Aggregation(self.hidden_dim * 2, self.hidden_dim)
        self.kg_aggre_query = _Aggregation(self.hidden_dim * 2, self.hidden_dim)
        self.kg_aggre_tail = _Aggregation(self.hidden_dim * 2, self.hidden_dim)
        self.kg_mlp_pre = _MultiLayerPercep(self.hidden_dim * 2, self.hidden_dim)

        self.g_u = _Aggregation(self.hidden_dim * 2, self.hidden_dim)
        self.g_q_1 = _Aggregation(self.hidden_dim * 2, self.hidden_dim)
        self.g_q_2 = _Aggregation(self.hidden_dim * 2, self.hidden_dim)
        self.g_i = _Aggregation(self.hidden_dim * 2, self.hidden_dim)

        self.ps_mlp_uq = _MultiLayerPercep(self.hidden_dim * 2, self.hidden_dim)
        self.ps_mlp_ui = _MultiLayerPercep(self.hidden_dim * 2, self.hidden_dim)

        self.ps_mlp_pred = _MultiLayerPercep(self.hidden_dim*2, 1)

        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['transE', 'inner_product']:
            raise ValueError('model %s not supported' % model_name)


    def trainkg(self, sample, mode, true_mode):
        if true_mode == True:
            triple, tail_true_company, head_true_company, query_true_company = sample
            batch_size, negative_sample_size = triple.size(0), 1

            head = self.user_embedding_KG(triple[:, 0]).unsqueeze(1)
            tail = self.item_embedding_KG(triple[:, 2]).unsqueeze(1)
            tail_true_company = self.item_embedding_KG(tail_true_company[:, 0]).unsqueeze(1)
            head_true_company = self.user_embedding_KG(head_true_company[:, 0]).unsqueeze(1)

            query_words = self.query_words_idx[triple[:, 1]].to(self.device)
            query_words_emb = self.word_embedding(query_words)
            qw_embs = self.get_query_embedding(query_words_emb).unsqueeze(1)

            query_words_company = self.query_words_idx[query_true_company[:, 0]].to(self.device)
            query_words_emb_company = self.word_embedding(query_words_company)
            query_true_company = self.get_query_embedding(query_words_emb_company).unsqueeze(1)

            score = self.compat_fun(head, qw_embs, tail, None, tail_true_company, head_true_company, query_true_company, mode)

        elif true_mode == False:

            head_part, tail_part, tail_true_company, head_true_company, query_true_company = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = self.user_embedding_KG(head_part[:, 0]).unsqueeze(1)
            tail_neg = self.item_embedding_KG(tail_part.long())
            tail = self.item_embedding_KG(head_part[:, 2]).unsqueeze(1)
            tail_true_company = self.item_embedding_KG(tail_true_company[:, 0]).unsqueeze(1)
            head_true_company = self.user_embedding_KG(head_true_company[:, 0]).unsqueeze(1)

            query_words = self.query_words_idx[head_part[:, 1]].to(self.device)
            query_words_emb = self.word_embedding(query_words)
            qw_embs = self.get_query_embedding(query_words_emb).unsqueeze(1)


            query_words_company = self.query_words_idx[query_true_company[:, 0]].to(self.device)
            query_words_emb_company = self.word_embedding(query_words_company)
            query_true_company = self.get_query_embedding(query_words_emb_company).unsqueeze(1)

            score = self.compat_fun(head, qw_embs, tail, tail_neg, tail_true_company, head_true_company, query_true_company, mode)
        else:
            raise ValueError('mode %s not supported' % mode)
            score = 0

        return score
    def get_query_embedding(self, query_words_emb):
        if 'mean' in self.query_net_struct:  # mean vector
            #print('Query model: mean')
            return self.get_addition_from_words(query_words_emb)
        elif 'fs' in self.query_net_struct:  # LSE f(s)
            #print('Query model: LSE f(s)')
            return self.get_fs_from_words(query_words_emb)
        elif 'RNN' in self.query_net_struct:  # RNN
            #print('Query model: RNN')
            return self.get_RNN_from_words(query_words_emb)
        else:
            #print('Query model: Attention')
            #return self.get_attention_from_words(query_words_emb)
            return self.get_fs_from_words(query_words_emb)

    def get_addition_from_words(self, query_words_emb):
        mean_word_vec = torch.mean(query_words_emb, dim=-2)
        return mean_word_vec

    def get_fs_from_words(self, query_words_emb):
        mean_word_vec = torch.mean(query_words_emb, dim=-2)
        f_s = self.query_linear(mean_word_vec)
        f_s = F.tanh(f_s)
        return f_s

    def get_RNN_from_words(self, query_words_emb):
        if len(query_words_emb.size()) > 3:
            query_words_emb_ = query_words_emb.view(-1, self.query_max_length, self.hidden_dim)
            out, _ = self.query_rnn(query_words_emb_, None)
            out_ = out[:, -1, :]
            out_ = out_.view(query_words_emb.size(0), query_words_emb.size(1), self.hidden_dim)
            return out_
        else:
            out, _ = self.query_rnn(query_words_emb, None)
            return out[:, -1, :]
        
    def forward(self, uids, queries, items):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'tail-batch', 'head-company-batch' or 'tail-company-batch' mode, sample consists four part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        the third part is the tail true company.
        the fourth part is the head true company.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, query) or (query, tail)).
        '''

        #[batch_size, dim]
        kg_user_embed = self.user_embedding_KG(uids).clone().detach()
        kg_item_embed = self.item_embedding_KG(items).clone().detach()


        query_words = self.query_words_idx[queries].to(self.device)
        query_words_emb = self.word_embedding(query_words)
        qw_embs = self.get_query_embedding(query_words_emb)


        u_latent = self.g_u(F.normalize(torch.cat([self.user_embedding_PS(uids), kg_user_embed], dim=-1), dim=-1))
        i_latent = self.g_i(F.normalize(torch.cat([self.item_embedding_PS(items), kg_item_embed], dim=-1), dim=-1))

        uq_latent = self.ps_mlp_uq(F.normalize(torch.cat([u_latent, qw_embs], dim=-1), dim=-1))
        ui_latent = self.ps_mlp_ui(F.normalize(torch.cat([u_latent, i_latent], dim=-1), dim=-1))

        scores = self.ps_mlp_pred(F.normalize(torch.cat([uq_latent, ui_latent], dim=-1), dim=-1)).squeeze(-1)

        return scores
    
    def TransE(self, head, query, tail, mode):
        if mode == 'head-batch':
            score = head + (query - tail)
        else:
            score = (head + query) - tail

        score = torch.norm(score, p=1, dim=2)
        return score

    def TransTailCompany(self, head, query, tail, tail_true):
        if tail_true.size()[1] == tail.size()[1]:
            tails_emb_cont = torch.cat((tail, tail_true), -1)
        else:
            tail_true = tail_true.expand(-1, tail.size()[1], -1)
            tails_emb_cont = torch.cat((tail, tail_true), -1)
        tails_emb_cont = F.normalize(tails_emb_cont, dim=-1)
        tail_ = self.kg_aggre_tail(tails_emb_cont)
        head_query_cat = torch.cat([head, query], -1)
        head_query_cat = F.normalize(head_query_cat, dim=-1)
        tail_pred = self.kg_mlp_pre(head_query_cat)
        score = torch.sum(tail_ * tail_pred, dim=2)
        return score

    def TransHeadCompany(self, head, query, tail, head_true_company):

        heads_emb_cont = torch.cat((head, head_true_company), -1)
        heads_emb_cont = F.normalize(heads_emb_cont, dim=-1)
        head_query_cat = torch.cat([self.kg_aggre_head(heads_emb_cont), query], -1)
        head_query_cat = F.normalize(head_query_cat, dim=-1)
        tail_pred = self.kg_mlp_pre(head_query_cat)
        score = torch.sum(tail*tail_pred, dim=2)
        return score

    def TransQueryCompany(self, head, query, tail, query_true_company):

        queries_emb_cont = torch.cat((query, query_true_company), -1)
        queries_emb_cont = F.normalize(queries_emb_cont, dim=-1)
        head_query_cat = torch.cat([head, self.kg_aggre_query(queries_emb_cont)], -1)
        head_query_cat = F.normalize(head_query_cat, dim=-1)
        tail_pred = self.kg_mlp_pre(head_query_cat)
        score = torch.sum(tail*tail_pred, dim=2)
        return score


    def compat_fun(self, head, query, tail, tail_neg, tail_true_company, head_true_company, query_true_company, mode):

        if tail_neg is None:
            if mode == 'tail-company-batch':
                score = self.TransTailCompany(head, query, tail, tail_true_company)
            elif mode == 'head-company-batch':
                score = self.TransHeadCompany(head, query, tail, head_true_company)
            elif mode == 'query-company-batch':
                score = self.TransQueryCompany(head, query, tail, query_true_company)
        else:
            if mode == 'tail-company-batch':
                score = self.TransTailCompany(head, query, tail_neg, tail)
            elif mode == 'head-company-batch':
                score = self.TransHeadCompany(head, query, tail_neg, head_true_company)
            elif mode == 'query-company-batch':
                score = self.TransQueryCompany(head, query, tail_neg, query_true_company)

        return score


