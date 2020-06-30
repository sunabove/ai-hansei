# -*- coding: utf-8 -*-
# %%
import math
import pandas as pd
import numpy as np
import argparse
#0 미만은 계산하지 않는다.

def RMSE(gt_value, pred_value, length):
    weight = [1,100,200, 1000, 6000]
    sum_error = 0
    pass_num = 0

    for i in range(length):
        if gt_value[i] < 0 :
            pass_num += 1
            continue
        standard = gt_value[i]
        gt_value[i] = math.log(gt_value[i] + 1)
        pred_value[i] = math.log(pred_value[i] + 1)
        if standard < 10:
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[0]
        elif (standard >= 10) & (standard < 100):
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[1]
        elif (standard >= 100) & (standard < 1000):
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[2]
        elif (standard >= 1000) & (standard < 10000):
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[3]
        elif (standard >= 10000):
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[4]
        pass
    pass

    sum_error = float(sum_error / length)
    sum_error = sum_error ** 0.5

    return sum_error
pass

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--gt", type=str, default='./datasets/test/test_proton_admin.csv')
    args.add_argument("--pred", type=str, default='./prediction/predict.csv')
    
    config = args.parse_args()
    
    gt = pd.read_csv(config.gt)['proton']
    pr = pd.read_csv(config.pred)['predict']
    print(len(gt), len(pr))
    result = RMSE(gt, pr, len(gt))
    
    print('evaluation: ', result)
    
if __name__ == '__main__' :
    main()
pass


# %%
