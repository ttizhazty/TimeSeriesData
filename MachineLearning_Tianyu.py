import pickle
import numpy as np
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pdb

raw_correct_idx = [1,2,3,4,5,6,7,8,9,20,23,31,33,34,39,40,41,43,44,45,55,56,57,58,60,65,67,69,70,71,72,73,74,75,76,79,80,81,82,83,88]
raw_problem_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21,22,24,25,26,27,28,29,30,32,35,36,37,38,42,46,47,48,49,50,51,52,53,54,59,61,62,63,64,66,68,77,78,84,85,86,87]
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

def loadData(dir):
    file_list = listdir(dir)
    train_X_all = np.zeros([len(file_list), 14520, correct_l])
    train_Y_all = np.zeros([len(file_list), 14520, problem_l])

    for i in range(1,len(file_list) - 1):
        file_path = dir + file_list[i]
        train_X_sample, train_Y_sample = laodTrainingSample(file_path)
        train_X_all[i,:,:] = train_X_sample 
        train_Y_all[i,:,:] = train_Y_sample
    train_X, test_X, train_Y, test_Y = train_test_split(train_X_all, train_Y_all, test_size = 0.1)
    return train_X, test_X, train_Y, test_Y 

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
    #rain_X, test_X, train_Y, test_Y = train_test_split(train_X_all, train_Y_all, test_size = 0.1)
    print(dir+file_list[-1])
    test_X, test_Y = laodTrainingSample(dir+file_list[-1])
     
    return train_X_all, test_X, train_Y_all, test_Y 

def linearModel(train_X, test_X, train_Y, test_Y):
    model_list = []
    train_X = train_X.reshape(train_X.shape[0] * 14520, -1)
    train_Y = train_Y.reshape(train_Y.shape[0] * 14520, -1)
    reg = linear_model.LinearRegression()
    for i in tqdm(range(train_X.shape[1])):
        reg.fit(train_X, train_Y[:,i])
        model_list.append(reg)
    return model_list

def linearModelPrediction(model_list, test_X, test_Y):
    #test_X = test_X.reshape(test_X.shape[0] * 14520, -1)
    #test_Y = test_Y.reshape(test_Y.shape[0] * 14520, -1)
    for i in range(len(model_list)):
        #print(model_list[i].score(test_X, test_Y[:,i]))
        pred_Y = model_list[i].predict(test_X)
        plt.plot(pred_Y)
        plt.plot(test_Y[:,i])
        

if __name__ == '__main__':
    input_dir = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/New_injector_failure_data/unzip/'
    train_X, test_X, train_Y, test_Y = loadData_seperate_day(input_dir)
    models = linearModel(train_X, test_X, train_Y, test_Y)
    linearModelPrediction(models, test_X, test_Y) 

