import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image
import pandas as pd
import numpy as np

# TODO: 0002 로그 패키지 임포트

import logging as log
log.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO)

is_corrcoeff_saved = 0 

class CustomDataset(data.Dataset): 

    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.labels = {}
        #self.data_index_pool = [0, 1, 5, 6, 7, 19, 24, 25, 29, 31]
        #self.label_index_pool = [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]
        
        self.label_path = os.path.join(self.root, self.phase + '.csv')
        with open(self.label_path, 'r', encoding='utf-8-sig') as f:
            # skip two lines
            f.readline()
            f.readline()

            input_data = []
            output_data = []

            # 20200330
            # Sample Train Dataset(2020. 01. 01 ~ 05. 01) 중 1일치 데이터(3월 30일)은 기계 오류로 인해 데이터가 수집 되지 않았음

            idx = 0 
            max_idx = 0  #TODO 9000 max idx

            data_prev = []
            hour_prev = -1 

            for line in f.readlines():
                values = line.strip().split(',')

                date = int( values[0] )
                hour = int( values[1] )

                if max_idx and max_idx == idx :
                    log.info( "**** max idx encounterd. %s" % max_idx )
                    break
                pass

                if 20200330 == date and phase == 'train':
                    log.info( "**** data encountered: %s, phase = %s \n" % ( date, phase ) )
                    break 
                pass

                data = values[2:]
                data = np.asfarray(data, np.float32)

                if phase != "train" :
                    input = data.copy()
                    output = data.copy() 
                    input_data.append( input )
                    output_data.append( output )
                    
                    if 0 : 
                        log.info( "[%04d] date: %08d, hour: %02d, data len = %d =================================" % (idx, date, hour, len(input)) )
                        log.info( "[%04d] %08d, %02d, input  : %s" % ( idx, date, hour, input ) )
                        log.info( "[%04d] %08d, %02d, output : %s" % ( idx, date, hour, output ) )
                        pass
                elif 0 == idx : 
                    log.info( "data_prev is null." )
                else : 
                    input = data_prev.copy()
                    output = data.copy() 

                    input_data.append( input )
                    output_data.append( output )

                    if 1 : 
                        log.info( "[%04d] date: %08d, hour: %02d, data len = %d =================================" % (idx, date, hour, len(input)) )
                        log.info( "[%04d] %08d, %02d, input  : %s" % ( idx, date, hour, input ) )
                        log.info( "[%04d] %08d, %02d, output : %s" % ( idx, date, hour, output ) )
                    pass
                pass

                data_prev = data
                idx += 1                
            pass

            global is_corrcoeff_saved 

            if not is_corrcoeff_saved : 
                #TODO 0020 corrcoef matrix 
                covMatrix = np.corrcoef( np.transpose( input_data ) ,bias=True) 
                print( "===== corrcoef matrix ======" )
                print( covMatrix )
                np.savetxt("corrcoef.csv", covMatrix, delimiter=",") 

                is_corrcoeff_saved = 1 
            pass
            
            if 0 : 
                #TODO 0006 시스템 강제 종료 
                log.info( "sys.exit(0)")
                import sys
                sys.exit( 0 ) 
            pass
        pass
        
        self.labels['input'] = input_data
        self.labels['output'] = output_data
        
    pass # // -- init         

    def __getitem__(self, index):
        if self.phase == 'test' :
            input_data = torch.tensor(self.labels['input'][index]) 
            dummy = []
            return (input_data,  dummy)
        pass

        input_data = torch.tensor(self.labels['input'][index]) 
        output_data = torch.tensor(self.labels['output'][index])
        
        return (input_data, output_data)
    pass

    def __len__(self):
        return len(self.labels['input'])
    pass

    def get_label_file(self):
        return self.label_path
    pass

pass

def data_loader(root, phase='train', batch_size=16): 

    dataset = CustomDataset(root, phase)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader, dataset.get_label_file()
pass 