#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import time
import math

import numpy as np
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from model import GraphSRRLModel

from GENPSDataset import GENPSDataset
from dataloader import TrainDatasetKG
from dataloader import TrainDatasetPS
from dataloader import TestDatasetPS
#from dataloader import TestDataset
from dataloader import OneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing graph embedding for personalized search',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--dataset', type=str, default='CIKMCup')
    parser.add_argument('--data_dir', type=str, default='./Data/CIKMCup_raw/')
    parser.add_argument('--input_dir', type=str, default='./Data/CIKMCup_raw/seq_query_split')
    parser.add_argument('--model', default='inner_product', type=str, help='inner_product')

    parser.add_argument('-n', '--negative_sample_size', default=32, type=int)
    parser.add_argument('-d', '--hidden_dim', default=50, type=int)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-r', '--regularization', default=0.00001, type=float)
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-init', '--init_checkpoint', default='./models', type=str)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', type=float, default=5, help='Weight decay.')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-save', '--save_path', default='models', type=str)

    parser.add_argument('--epoch', type=int, default=15, help='the number of epochs to train for')
    parser.add_argument('--nuser', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nitem', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nquery', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--device', type=str, default='cuda', help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, epoch, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode, metric, epoch, metrics[metric]))


def main(args):
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True    #assure the reproduction

    dataset = ['MeituanData_raw']   #'MeituanData_raw', 'CIKMCup_raw', 'amazon_electronics_dataset', 'amazon_kindle_dataset'
    #proportion_train = [20, 40, 60]
    grid_search_hidden_dim = [50]  #10, 50, 100, 150, 200

    for ds in dataset:
        args.data_dir = '../../../Data/%s/' % ds
        args.dataset = ds
        metrics_list = []
        #for pt in proportion_train:
        for hd in grid_search_hidden_dim:
            args.input_dir = './Data/%s/seq_query_split' % ds
            args.hidden_dim = hd
            #print('current dataset: %s, current train_proportion %d' % (ds, pt))
            print('current dataset: %s, current hidden_dim %d' % (ds, hd))
            print("Loading data ...")
            data_set = GENPSDataset(args.data_dir, args.input_dir, 'train')
            data_set.get_test_dataset(input_dir=args.input_dir)

            nuser = data_set.user_size
            nitem = data_set.item_size
            nquery = len(data_set.query_words)

            args.nuser = nuser
            args.nitem = nitem
            args.nquery = nquery
            # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            train_triples = data_set.interaction_info
            test_triples = data_set.interaction_info_test

            # All true triples
            # all_true_triples = train_triples + valid_triples + test_triples
            all_true_triples = train_triples + test_triples
            device = torch.device(args.device)
            GraphSRRL_model = GraphSRRLModel(
                model_name=args.model,
                nuser=nuser,
                nitem=nitem,
                nquery=nquery,
                hidden_dim=args.hidden_dim,
                dataset=data_set,
                device=device
            )

            logging.info('Model Parameter Configuration:')
            for name, param in GraphSRRL_model.named_parameters():
                logging.info(
                    'Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

            GraphSRRL_model.to(device)
            init_network(GraphSRRL_model)
            train(GraphSRRL_model, args, train_triples, nuser, nitem, nquery, device)
            metric = test(GraphSRRL_model, test_triples, all_true_triples, args, True)
            metrics_list.append(metric)

        with open(args.save_path+'/results_meric.txt', 'a+') as f:
            #for i in range(len(proportion_train)):
            for i in range(len(grid_search_hidden_dim)):
                #f.write("current dataset: %s, current train_proportion: %d" % (ds, proportion_train[i]) + '\n')
                f.write("current dataset: %s, current hidden_dim: %d" % (ds, grid_search_hidden_dim[i]) + '\n')
                for metric in metrics_list[i]:
                    f.write('%s: %f' % (metric, metrics_list[i][metric]) + '\n')


def get_data_load(args, train_triples, nuser, nitem, nquery, device):
    train_dataloader_PS = DataLoader(
        TrainDatasetPS(train_triples, nuser, nitem, nquery, args.negative_sample_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TrainDatasetPS.collate_fn
    )

    train_dataloader_tailcompany = DataLoader(
        TrainDatasetKG(train_triples, nuser, nitem, nquery, args.negative_sample_size, 'tail-company-batch'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TrainDatasetKG.collate_fn
    )

    train_dataloader_headcompany = DataLoader(
        TrainDatasetKG(train_triples, nuser, nitem, nquery, args.negative_sample_size, 'head-company-batch'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TrainDatasetKG.collate_fn
    )

    train_dataloader_querycompany = DataLoader(
        TrainDatasetKG(train_triples, nuser, nitem, nquery, args.negative_sample_size, 'query-company-batch'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TrainDatasetKG.collate_fn
    )

    train_iterator_KG = OneShotIterator(train_dataloader_tailcompany, train_dataloader_headcompany,
                                        train_dataloader_querycompany)

    return train_iterator_KG, train_dataloader_PS

def train(GraphSRRL_model, args, train_triples, nuser, nitem, nquery, device, init_model=False):

    train_iterator_KG, train_dataloader_PS = get_data_load(args, train_triples, nuser, nitem, nquery, device)
    # Set training configuration
    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, GraphSRRL_model.parameters()),
        lr=current_learning_rate
    )
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)

    if init_model:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(args.init_checkpoint)
        init_epoch = checkpoint['epoch']
        GraphSRRL_model.load_state_dict(checkpoint['state_dict'])

        current_learning_rate.load_state_dict(checkpoint['current_learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_epoch = 0

    # Training Loop
    logging.info("Start Training...")
    logging.info('init_epoch = %d' % init_epoch)
    logging.info('current_learning_rate = %f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)

    steps = len(train_triples) // args.batch_size
    if steps * args.batch_size < len(train_triples):
        steps += 1
    warm_up_epoch = 3
    for e in range(init_epoch, args.epoch):

        # KG training
        trainForKG(GraphSRRL_model, optimizer, train_iterator_KG, args, steps, e)

        # PS training
        trainForPS(GraphSRRL_model, optimizer, train_dataloader_PS, args, criterion, e)

        current_learning_rate = current_learning_rate / args.weight_decay
        if e > warm_up_epoch:
            current_learning_rate = 0.001
            logging.info('current_learning_rate = %f' % current_learning_rate)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, GraphSRRL_model.parameters()),
                                         lr=current_learning_rate)
            warm_up_epoch = warm_up_epoch + 3
            train_iterator_KG, train_dataloader_PS = get_data_load(args, train_triples, nuser, nitem, nquery, device)

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': e + 1,
            'state_dict': GraphSRRL_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'current_learning_rate': current_learning_rate
        }

        torch.save(ckpt_dict, os.path.join(args.save_path, 'checkpoint_latest_%d_%s_%s'% (args.hidden_dim, args.dataset, args.input_dir.split('_')[-1])))


def test(GraphSRRL_model, test_triples, all_true_triples, args, init_model=True):

    if init_model:
        # Restore model from checkpoint directory
        path_init = os.path.join(args.save_path, 'checkpoint_latest_%d_%s_%s'% (args.hidden_dim, args.dataset, args.input_dir.split('_')[-1]))
        logging.info('Loading checkpoint %s...' % path_init)
        checkpoint = torch.load(path_init)
        init_epoch = checkpoint['epoch']
        GraphSRRL_model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_epoch = 0

    logging.info('Evaluating on Valid Dataset...')
    metrics = testforPS(GraphSRRL_model, test_triples, all_true_triples, args)
    log_metrics('Valid', init_epoch, metrics)
    return metrics


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def trainForPS(GraphSRRL_model, optimizer, train_dataloader_PS, args, criterion, epoch):
    GraphSRRL_model.train()
    sum_epoch_loss = 0
    start = time.time()

    for i, (uids, queries, items, labels, subsampling_weight) in enumerate(train_dataloader_PS):

        subsampling_weight = subsampling_weight.to(torch.device(args.device))
        uids = uids.to(torch.device(args.device))
        queries = queries.to(torch.device(args.device))
        items = items.to(torch.device(args.device))
        labels = labels.to(torch.device(args.device))

        optimizer.zero_grad()
        score_normalized = GraphSRRL_model(uids, queries, items)

        loss = criterion(score_normalized.float(), labels.float())
        if args.regularization != 0.0:
            # Use L2 regularization for ComplEx and DistMult
            regulariza= args.regularization * (
                    GraphSRRL_model.user_embedding_PS.weight.data.norm(p=2) ** 2 +
                    GraphSRRL_model.word_embedding.weight.data.norm(p=2) ** 2 +
                    GraphSRRL_model.item_embedding_PS.weight.data.norm(p=2) ** 2
            )
            loss = loss + args.regularization*regulariza
            regularization_log = {'regularization': regulariza.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        if i % 1000 == 0:
            logging.info('[TRAIN PS EPOCH %d] step %d/%d batch loss: %.4f (avg %.4f) (%.2f s)'
                % (epoch, i + 1, len(train_dataloader_PS), loss, sum_epoch_loss / (i + 1),
                   (time.time() - start)))

        start = time.time()


def trainForKG(GraphSRRL_model, optimizer, train_iterator_KG, args, steps, epoch):
    GraphSRRL_model.train()
    start = time.time()
    optimizer.zero_grad()
    sum_step_loss = 0
    for s in range(steps):
        positive_sample, negative_sample, subsampling_weight, mode, true_tail_company, true_head_company, true_query_company = train_iterator_KG.next()
        positive_sample = positive_sample.to(torch.device(args.device))
        negative_sample = negative_sample.to(torch.device(args.device))
        subsampling_weight = subsampling_weight.to(torch.device(args.device))

        true_tail_company = true_tail_company.to(torch.device(args.device))
        true_head_company = true_head_company.to(torch.device(args.device))
        true_query_company = true_query_company.to(torch.device(args.device))

        optimizer.zero_grad()
        negative_score = GraphSRRL_model.trainkg(
            (positive_sample, negative_sample, true_tail_company, true_head_company, true_query_company), mode, False)


        negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = GraphSRRL_model.trainkg((positive_sample, true_tail_company, true_head_company, true_query_company), mode, True)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L2 regularization
            regulariza = args.regularization * (
                    GraphSRRL_model.user_embedding_KG.weight.data.norm(p=2) ** 2 +
                    GraphSRRL_model.word_embedding.weight.data.norm(p=2) ** 2 +
                    GraphSRRL_model.item_embedding_KG.weight.data.norm(p=2) ** 2
            )
            loss = loss + args.regularization*regulariza
            regularization_log = {'regularization': regulariza.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_step_loss += loss_val


        if s % 1000 == 0:
            logging.info('[TRAIN KG EPOCH %d] step %d/%d batch loss: %.4f (avg %.4f) (%.2f s)'
                % (epoch, s + 1, steps, loss, sum_step_loss / (s + 1),
                  (time.time() - start)))
        start = time.time()


def testforPS(model, test_triples, all_true_triples, args):
    '''
    Evaluate the model on test or valid datasets
    '''

    model.eval()

    test_dataloader = DataLoader(
        TestDatasetPS(
            test_triples,
            all_true_triples,
            args.nuser,
            args.nitem,
            args.nquery,
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDatasetPS.collate_fn
    )

    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataloader])
    with torch.no_grad():
        for positive_sample, uids, queries, negative_tails, filter_bias in test_dataloader:
            device = torch.device(args.device)
            positive_sample = positive_sample.to(device)
            uids = uids.to(device)
            queries = queries.to(device)
            negative_tails = negative_tails.to(device)
            filter_bias = filter_bias.to(torch.device(args.device))

            score_neg = model(uids, queries, negative_tails)
            score_neg += filter_bias

            sorted_, indices = torch.sort(score_neg, dim=1, descending=True)

            positive_arg = positive_sample[:, 2]
            batch_size = positive_sample.size()[0]
            for i in range(batch_size):
                # Notice that argsort is not ranking
                ranking = (indices[i, :] == positive_arg[i]).nonzero()
                assert ranking.size(0) == 1

                # ranking + 1 is the true ranking used in evaluation metrics
                ranking = 1 + ranking.item()
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'ndcg@10': math.log(2) / math.log(ranking + 1) if ranking <= 10 else 0.0,
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@5': 1.0 if ranking <= 5 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })

            if step % 1000 == 0:
                logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

            step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    return metrics
        
if __name__ == '__main__':
    main(parse_args())
