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
import xgboost as xgb
from matplotlib import pyplot as plt
from seq2seq_v1 import Seq2seqModel
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
raw_correct_idx = [1,2,3,4,5,6,7,8,9,20,23,31,33,34,39,40,41,43,44,45,55,60,69,70,71,72,73,74,75,76,79,80,81,82,83,88]
raw_problem_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21,22,24,25,26,27,28,29,30,32,35,36,37,38,42,46,47,48,49,50,51,52,53,54,61,62,63,68,77,78]
correct_idx = []
problem_idx = []
for idx in raw_correct_idx:
    correct_idx.append(idx - 1)
correct_l = len(correct_idx)

for idx in raw_problem_idx:
    problem_idx.append(idx - 1) 
problem_l = len(problem_idx)

def laodTrainingSample(filepath):
    with open(filepath, 'rb') as f:
        train_one_day = pickle.load(f)
        train_X = train_one_day[:, correct_idx]
        train_Y = train_one_day[:, problem_idx]
        return train_X, train_Y

def dataDistributionAnalysis(train_X, train_Y):
    print('begin to plot the data on the scale of time......')
    for i in range(train_X.shape[1]):
        senor_idx = raw_correct_idx[i]
        sensor_name = sensor_dict[senor_idx]
        plt.figure()
        plt.plot(train_X[:, i])
        plt.title(sensor_name + ' vs time')
        plt.savefig('./res/plot_data/correct_sensor_%d.png' %i)

    for i in range(train_Y.shape[1]):
        senor_idx = raw_problem_idx[i]
        sensor_name = sensor_dict[senor_idx]
        plt.figure()
        plt.plot(train_Y[:, i])
        plt.title(sensor_name + ' vs time')
        plt.savefig('./res/plot_data/probelm_sensor_%d.png' %i)
    print('plot finished !!!')

def loadData(dir):
    file_list = listdir(dir)
    np.random.shuffle(file_list)
    #train_X_all = np.zeros([len(file_list), 14520, correct_l])
    #train_Y_all = np.zeros([len(file_list), 14520, problem_l])
    train_X_all = np.zeros([200, 14520, correct_l])
    train_Y_all = np.zeros([200, 14520, problem_l])
    #for i in range(1,len(file_list) - 1):
    for i in range(200):
        file_path = dir + file_list[i]
        try:
            train_X_sample, train_Y_sample = laodTrainingSample(file_path)
            train_X_all[i,:,:] = train_X_sample 
            train_Y_all[i,:,:] = train_Y_sample
        except Exception:
            continue
    '''
    #comment out here to chcek the data distribution on the scale of time line.
    train_X, test_X, train_Y, test_Y = train_test_split(train_X_all, train_Y_all, test_size = 0.2, random_state=7)
    ''' 
    train_X = train_X_all.reshape(200*14520, -1)
    train_Y = train_Y_all.reshape(200*14520, -1)
    #print(train_X.shape)
    #print(train_Y.shape)
    print('training data size is: ', train_X.shape)
    print('train label size is: ', train_Y.shape)
    train_X, test_X, train_Y, test_Y = train_test_split(train_X_all, train_Y_all, test_size = 0.2, random_state=7) 
    train_X = train_X.reshape(160, 14520,-1)
    train_Y = train_Y.reshape(160,14520,-1)
    test_X = test_X.reshape(40,14520,-1)
    test_Y = test_Y.reshape(40,14520,-1)
    return train_X, test_X, train_Y, test_Y 
    '''
    return train_X, train_Y
    '''
def loadData_seperate_day(dir):
    file_list = listdir(dir)
    train_X_all = np.zeros([len(file_list), 14520, correct_l])
    train_Y_all = np.zeros([len(file_list), 14520, problem_l])
    print(file_list)
    for i in range(1,len(file_list) - 1):
        file_path = dir + file_list[i]
        train_X_sample, train_Y_sample = laodTrainingSample(file_path)
        train_X_all[i,:,:] = train_X_sample 
        train_Y_all[i,:,:] = train_Y_sample
    train_X, test_X, train_Y, test_Y = train_test_split(train_X_all, train_Y_all, test_size = 0.1)
    print(dir+file_list[-1])
    test_X, test_Y = laodTrainingSample(dir+file_list[-1])
    return train_X_all, test_X, train_Y_all, test_Y

if __name__ == '__main__':
    input_dir = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/New_injector_failure_data/unzip/'
    train_X, test_X, train_Y, test_Y = loadData(input_dir)
    Seq2seqV1 = Seq2seqModel(encode_dim=36,decode_dim=40,input_seq_len=121,output_seq_len=121,rnn_size=40,layer_size=3,dnn_size = [64,128,256,128,64],learning_rate=0.001,dropout=0.7,reg_lambda=0.5,train_X=train_X,train_Y=train_Y,test_X=test_X,test_Y=test_Y,sensor2sensor=True,train_epoch=50)