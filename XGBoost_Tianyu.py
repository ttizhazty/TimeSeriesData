# -*- coding: utf-8 -*-
import pickle
import numpy as np
from tqdm import tqdm
import time
import os
from os import listdir
from os.path import isfile, join
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib as mpl
if os.environ.get('DISPLAY','')=='':
    print('no display found.')
    mpl.use('Agg')
from matplotlib import pyplot as plt
import pdb

sensor_dict = {}
sensor_list = ['1.[Accelerator pedal opening degree](%)', '2.[Actual Engine Torque](%)', '3.[Engine Speed](rpm)', '4.[Target fuel injection amount](mm3/st)', '5.[Current Gear]', '6.[Vehicle speed (25 pulses)](km/h)', '7.[CluchSW](MT only)', '8.[Brake SW]', '9.[Cruise Control Status]', '10.[coolant temperature](℃)', '11.[fuel temperature](℃)', '12.[Post injection Q](mm3)', '13.[Common rail pressure](MPa)', '14.[DPF differential pressure](kPa)', '15.[Atmospheric pressure](kPa)', '16.[Intake air temperature](℃)', '17.[Boost pressure](kPa)', '18.[CSF inlet temperature](℃)', '19.[DOC inlet temperature](℃)', '20.[DPF Status]', '21.[DPF error count]', '22.[DPF warning count]', '23.[DPF PM accumulation status]', '24.[DPF mileage status]', '25.[ITH Motor Protect Duty Limit Status]', '26.[EGR Motor Protect Duty Limit Status]', '27.[EGR Motor2 Protect Duty Limit Status]', '28.[DPF mode]', '29.[MAF](g/cyl)', '30.[EGR Duty](%)', '31.[EGR Target Position](%)','32.[EGR Actual Position](%)', '33.[Intake Throttle Duty](%)', '34.[Intake Throttle Target Position](%)', '35.[Intake Throttle Actual Position](%)', '36.[IGN Voltage](V)', '37.[RPCV Duty(medium small)・PCV Close Timing(large)](%・CA)', '38.[RPCV Actual Current(medium small)・PCV F/B Control Quantity(large)](mA・CA)', '39.[RPCV Desired Current(medium small)・EGR BLDC 2 Actual Position(large)](mA・%)', '40.[RPCV Commanded Fuel Flow(medium small)・EGR BLDC 2 Duty](mm3/sec・%)', '41.[Target Rail Pressure](Mpa)', '42.[VNT actual Position](%)', '43.[VNT Target Position](%)', '44.[Target Boost](%)', '45.[Engine Mode]', '46.[Mail SOI](CA)', '47.[Pilot SOI](CA)', '48.[CAM CRANK Synchro Status]', '49.[Cylinder1 Balancing Fuel Compensation](mm3/st)', '50.[Cylinder2 Balancing Fuel Compensation](mm3/st)', '51.[Cylinder3 Balancing Fuel Compensation](mm3/st)', '52.[Cylinder4 Balancing Fuel Compensation](mm3/st)', '53.[Cylinder5 Balancing Fuel Compensation](mm3/st)', '54.[Cylinder6 Balancing Fuel Compensation](mm3/st)', '55.[Target Idle rpm](rpm)', '56.[VGS Magnetic Valve Drive Status 1]', '57.[VGS Magnetic Valve Drive Status 2]', '58.[VGS Magnetic Valve Drive Status 3]', '59.[EGR cooler bypas valve]', '60.[Exhaust pipe INJ ON / OFF state](%)', '61.[Injection amount of exhaust pipe INJ](mm3/st)', '62.[Exhaust pipe INJ fuel pressure](kPa)', '63.[Compressor outlet temperature](℃)', '64.[Rail pressure reducing valve drive duty](%)', '65.[Rail pressure reducing valve target current](mA)', '66.[Rail pressure reducing valve actual current](mA)', '67.[Rail pressure reducing valve target pressure](MPa)', '68.[Turbo EVRV Duty output](%)', '69.[egr_bldc_pid_base_dc_1]', '70.[egr_bldc_pid_base_dc_2]', '71.[egr_bldc_p_term_fnl_1]', '72.[egr_bldc_p_term_fnl_2]', '73.[egr_bldc_i_term_fnl_1]', '74.[egr_bldc_i_term_fnl_2]', '75.[rpcv_dc_p_gain]', '76.[rpcv_dc_i_gain]', '77.[trb_trg_base_pos]', '78.[trb_map_fb_pos]', '79.[trb_map_p_term_fnl]', '80.[trb_map_i_term_fnl]', '81.[ith_dc_p_term]', '82.[ith_dc_i_term]', '83.[ith_dc_ff_fb]', '84.[CAC in sensor output]', '85.[CAC out sensor output]', '86.[Rail pressure sensor 2 output](MPa)', '87.[Sensor value O2](%)','88.[TBD]']

for item in sensor_list:
    sensor_number = int(item.split('.')[0])
    senror_name = item.split('.')[1]
    sensor_dict[sensor_number] = senror_name
'''
raw_correct_idx = [1,2,3,4,5,6,7,8,9,20,23,31,33,34,39,40,41,43,44,45,55,56,57,58,60,65,67,69,70,71,72,73,74,75,76,79,80,81,82,83,88]
raw_problem_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21,22,24,25,26,27,28,29,30,32,35,36,37,38,42,46,47,48,49,50,51,52,53,54,59,61,63,64,66,68,77,78,84,85,86,87]
'''
raw_correct_idx = [1,2,3,4,5,6,7,8,9,20,23,31,33,34,39,40,41,43,44,45,55,60,69,70,71,72,73,74,75,76,79,80,81,82,83] # depends
raw_problem_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21,22,24,25,26,27,28,29,30,32,35,36,37,38,42,46,47,48,49,50,51,52,53,54,61,62,63,68,77,78]
correct_idx = []
problem_idx = []
for idx in raw_correct_idx:
    correct_idx.append(idx - 1)
correct_l = len(correct_idx)

for idx in raw_problem_idx:
    problem_idx.append(idx - 1) 
problem_l = len(problem_idx)

def trainTestSplit(train_input_path, test_input_path):
    train_folders = os.listdir(train_input_path)
    test_folders = os.listdir(test_input_path)
    all_test_files = []
    train_feature = np.zeros((1,35))
    train_label = np.zeros((1,40))
    test_case = []
    
    # Collect the valid testing data:
    print('loading testing data')
    for folder in test_folders:
        if len(folder) > 10: # filter out the invalid folders
            test_files = os.listdir(test_input_path + folder)
            all_test_files += test_files
            test_feature = np.zeros((1,35))
            test_label = np.zeros((1,40))
            for test_file in test_files:
                test_X, test_Y = laodTrainingSample(test_input_path + folder + '/' + test_file)
                test_feature = np.concatenate((test_feature, test_X), axis=0)
                test_label = np.concatenate((test_label, test_Y), axis=0)
                # print(test_feature.shape)
                # print(test_label.shape)
            test_case.append((test_feature, test_label))   
    
    # Collect the training data
    print('loading training data')
    if not os.path.isfile('./../res_data/sample_train_feature_all.pkl'):
        cnt = 0
        for folder in train_folders:
            if len(folder) >= 10:
                train_files = os.listdir(train_input_path + folder)
                np.random.shuffle(train_files)
                for train_file in train_files:
                    if cnt > 100000:
                        break
                    print(cnt)
                    cnt += 1
                    if train_file not in all_test_files: #and train_file[7:11] == '2017':
                        print(train_file)
                        train_X, train_Y = laodTrainingSample(train_input_path + folder + '/' + train_file)
                        # Sampling
                        sample_step = 30
                        train_X, train_Y = train_X[::sample_step,:], train_Y[::sample_step,:]
                        train_feature = np.concatenate((train_feature, train_X), axis=0)
                        train_label = np.concatenate((train_label, train_Y), axis=0)
        print('data loading completed')    
        new_data = np.concatenate((train_feature, train_label), axis=1)
        np.random.shuffle(new_data)
        train_feature = new_data[:,:35]
        train_label = new_data[:,35:]
        pickle.dump(train_feature, open('./../res_data/sample_train_feature_all.pkl', 'wb'))
        pickle.dump(train_label, open('./../res_data/sample_train_label_all.pkl', 'wb'))
    else:
        with open('./../res_data/sample_train_feature_all.pkl', 'rb') as f:
            train_feature = pickle.load(f)
        with open('./../res_data/sample_train_label_all.pkl', 'rb') as f:
            train_label = pickle.load(f)
    return train_feature, train_label, test_case
    

def laodTrainingSample(filepath):
    with open(filepath, 'rb') as f:
        train_one_day = pickle.load(f)
        try:
            temp = []
            for item in train_one_day:
                if len(item) == 88:
                    temp.append(item)
            train_one_day = np.array(temp)
            train_X = train_one_day[:, correct_idx]
            train_Y = train_one_day[:, problem_idx]
        except IndexError:
            train_X, train_Y = np.zeros((1,35)), np.zeros((1,40))
        return train_X, train_Y


def xgbModel(train_X, test_X, test_case):
    print('strat training the linear models ......')
    print('the data size is ......')
    print(train_X.shape)
    print(train_Y.shape)

    params1={
        'booster':'gbtree',
        'objective': 'reg:linear',
        'gamma':0.1,
        'max_depth':7, 
        'lambda':2,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'min_child_weight':1, 
        'silent':0 ,
        'eta': 0.05, 
        'seed':1000,
        #'nthread':7,
        }

    params2={
        'booster':'gbtree',
        'objective': 'reg:linear',
        'gamma':0.1,
        'max_depth':5, 
        'lambda':2,
        'subsample':0.5,
        'colsample_bytree':0.8,
        'min_child_weight':3, 
        'silent':0 ,
        'eta': 0.01, 
        'seed':1000,
        #'nthread':7,
        }

    plst1 = list(params1.items())
    plst2 = list(params2.items())
    # TODO:divided the training set again to check training status.......
    num_rounds = 400
    c = 0
    for item in test_case:
        test_X = item[0]
        test_Y = item[1]
        print('the testing data size in this case is :', test_X.shape, test_Y.shape)
        for i in range(train_Y.shape[1]):
            sensor_idx = raw_problem_idx[i]
            sensor_name = sensor_dict[sensor_idx]
            
            # training .....
            if not os.path.isfile('./../models/xgb_models_sampledata/sensor_%d' %i + '.model'):
                train_label = np.abs(train_Y[:,i]).reshape(-1)
                xgb_train = xgb.DMatrix(train_X, label=train_label)
                watchlist = [(xgb_train, 'train')]
                model = xgb.train(plst1, xgb_train, num_rounds, watchlist)
                #model saving ...
                model.save_model('./../models/xgb_models_sampledata/sensor_%d' %i + '.model')
            else:
                print('loading model......')
                model = xgb.Booster()
                model.load_model('./../models/xgb_models_sampledata/sensor_%d' %i + '.model')
            
            xgb_test = xgb.DMatrix(test_X)
            
            preds = model.predict(xgb_test)
            preds = np.array(preds).reshape(-1)
            test_label = test_Y[:,i].reshape(-1)
            loss = np.sqrt(np.mean(np.subtract(test_label, preds)**2))
            difference = (preds - test_label) / test_label * 100
            difference[np.isinf(difference)] = preds[np.isinf(difference)]
            print('the loss of this model is :', loss)
            print('i am in plot !!!!!!') 
            plt.figure()
            plt.plot(difference)
            #plt.plot(preds[::1000])
            #plt.plot(test_label[::1000])
            # plt.plot(test_label[::2000]*0.85,'r')
            # plt.plot(test_label[::2000]*1.15, 'r')
            #plt.legend(['prediciton', 'label', 'lower_bound', 'upper_bound'])
            plt.xlabel('Samples')
            plt.ylabel('Value')
            plt.ylim((-200,200))
            plt.title(sensor_name + '(loss=%f)' %loss)
            plt.savefig('./../res/day_pred/predictions_xgb_sample2017/case%d_'%c +'sensor_%d_xgb.png' %i)
            plt.close()
        c += 1

if __name__ == '__main__':
    train_input_path = './../processed_data/'
    test_input_path = './../processed_data/abnormal_data/'
    train_X, train_Y, test_case = trainTestSplit(train_input_path, test_input_path)
    xgbModel(train_X, train_Y, test_case)
