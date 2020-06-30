# -*- coding: utf-8 -*-

import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
import csv

from dataloader import data_loader
from evaluation import evaluation_metrics
from model import Net

'''
- 참가자는 모델의 결과 파일(Ex> prediction.txt)을 write가 가능한 폴더에 저장되도록 적절 한 path를 입력해야합니다. (tf/notebooks)
'''

import time

# TODO: 0002 로그 패키지 임포트

import logging as log
log.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO)

try:
    # print( e )
    # TODO: 0001 데이터 경로롤 문제마다 다르게 설정
    DATASET_PATH = os.path.join('/tf/notebooks/datasets/19_forecast_traffic')
    if os.name is 'nt':  # windows operating system 인 경우
        log.info("os.name : %s" % os.name)
        DATASET_PATH = os.path.join('./datasets')
    pass
    log.info("DATASET_PATH : %s" % DATASET_PATH)
except Exception as e :
    print( e )
pass


def _infer(model, cuda, data_loader):
    res_pred = None
    for idx, (input_data, _) in enumerate(data_loader):
        if cuda:
            input_data = input_data.cuda()
        pass

        pred = model(input_data)
        pred = pred.detach().cpu().numpy()

        if idx == 0:
            res_pred = pred
        else:
            res_pred = np.concatenate((res_pred, pred), axis=0)
        pass

    return list(res_pred)
pass

def feed_infer(output_file, infer_func):
    prediction = infer_func()

    log.info('Write output ....')

    if 1 : # txt save
        output = open(output_file, 'w', encoding='utf-8-sig', newline='')

        delim = " "

        for pred in prediction :
            output.write( delim.join([str(x) for x in pred]) + '\n')
        pass

        output.close()

        if os.stat(output_file).st_size == 0:
            raise AssertionError('output result of inference is nothing')
        pass
    pass

    if 1 : # csv save
        output = open("predicition.csv", 'w', encoding='utf-8-sig', newline='')

        delim = ", "
        for pred in prediction :
            output.write( delim.join([str(x) for x in pred]) + '\n')
        pass

        output.close()

        if os.stat(output_file).st_size == 0:
            raise AssertionError('output result of inference is nothing')
        pass
    pass

    log.info('Done. Writing output ....')

pass

def validate(prediction_file, model, validate_dataloader, validate_label_file, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=validate_dataloader))

    metric_result = evaluation_metrics(prediction_file, validate_label_file)
    print( 'Eval result: {:.4f}'.format(metric_result) )
    return metric_result
pass

def test(prediction_file_name, model, test_dataloader, cuda):
    feed_infer(prediction_file_name, lambda : _infer(model, cuda, data_loader=test_dataloader))
pass

def save_model(model_name, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    file_name = os.path.join( "./model_data/", model_name + '.pth')
    
    torch.save(state, file_name )

    log.info('model saved. file_name : %s' % file_name )
pass

def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    log.info('model loaded')
pass


def main_work( ) :
    # TODO: 0003 시작 함수 main
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=int, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--model_name", type=str, default="model.pth")
    args.add_argument("--prediction_file", type=str, default="prediction.txt")
    args.add_argument("--batch", type=int, default=4)
    args.add_argument("--mode", type=str, default="train")

    config = args.parse_args()

    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode

    # create model
    model = Net()

    if mode == 'test':
        load_model(model_name, model)
    pass

    if cuda:
        model = model.cuda()
    pass

    # -- train 
    if mode == 'train':
        # define loss function
        loss_fn = nn.L1Loss()
        if cuda:
            loss_fn = loss_fn.cuda()
        pass

        # set optimizer
        optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

        # get data loader
        #TODO 0004 데이터 로딩
        log.info( "Data loading now ......" )
        then = time.time()

        log.info( "Loading training data ...." )
        train_dataloader, _ = data_loader(root=DATASET_PATH, phase='train', batch_size=batch)
        log.info( "Done. Loading training data ...." )

        now = time.time()
        log.info( "// Training data loading. Duration %d sec(s)" % ( now - then ))

        time_ = datetime.datetime.now()
        num_batches = len(train_dataloader) 

        #check parameter of model
        log.info("----------------- check parameter of model -------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        log.info("num of parameter : %s" % total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("num of trainable_ parameter : %s" % trainable_params)
        log.info("------------------ // check parameter of model ------------------------------------------")

        # train
        for epoch in range( num_epochs ):
            model.train()

            for iter_, (input_data, output_data) in enumerate( train_dataloader ):
                # fetch train data
                if cuda:
                    input_data = input_data.cuda()
                    output_data = output_data.cuda()
                pass

                # update weight
                pred = model(input_data)
                loss = loss_fn(pred, output_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iter_ + 1) % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) elapsed {} expected per epoch {}'.format( _epoch, num_epochs, loss.item(), elapsed, expected))
                    time_ = datetime.datetime.now()
                pass
            pass

            # scheduler update
            scheduler.step()

            # save model
            save_model(str(epoch + 1), model, optimizer, scheduler)

            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {} \n'.format(epoch + 1, elapsed))
        pass # // train

        # validate
        if 1 :
            log.info( "Validating ...." )

            then = time.time()
            log.info( "Loading validation data ...." )
            validate_dataloader, validate_label_file = data_loader(root=DATASET_PATH, phase='validate', batch_size=batch)
            now = time.time()
            log.info( "Done. Validation data loading. Duration %d sec(s) \n" % ( now - then ))

            validate( prediction_file, model, validate_dataloader, validate_label_file, cuda )

            log.info( "Done. Validating. \n" )
        pass
    # // -- train

    elif mode == 'test':
        model.eval()
        # get data loader
        test_dataloader, test_file = data_loader(root=DATASET_PATH, phase='test', batch_size=batch)
        test(prediction_file, model, test_dataloader, cuda)
        # submit test result
    pass
pass

if __name__ == '__main__':
    main_work()
pass