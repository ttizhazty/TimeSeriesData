import pickle
import numpy as np
from tqdm import tqdm
import time
import os
from os import listdir
from os.path import isfile, join
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import xgboost as xgb
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
    file_list = listdir(dir)[1:]
    np.random.shuffle(file_list)
    #train_X_all = np.zeros([len(file_list), 14520, correct_l])
    #train_Y_all = np.zeros([len(file_list), 14520, problem_l])
    train_X_all = np.zeros([50, 14520, correct_l])
    train_Y_all = np.zeros([50, 14520, problem_l])
    #for i in range(1,len(file_list) - 1):
    for i in range(50):
        file_path = dir + file_list[i]
        train_X_sample, train_Y_sample = laodTrainingSample(file_path)
        train_X_all[i,:,:] = train_X_sample 
        train_Y_all[i,:,:] = train_Y_sample
    train_X, test_X, train_Y, test_Y = train_test_split(train_X_all, train_Y_all, test_size = 0.02, random_state=1)
    print('training data size is: ', train_X.shape)
    print('testing data size is: ', test_Y.shape)
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
    test_X = test_X.reshape(test_X.shape[0] * 14520, -1)
    test_Y = test_Y.reshape(test_Y.shape[0] * 14520, -1)
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    reg_linear = linear_model.LinearRegression()
    reg_lasso = linear_model.Lasso()
    reg_ridge = linear_model.Ridge(alpha = 5)
    for i in range(train_X.shape[1]):
        reg_ridge.fit(train_X, train_Y[:,i])
        weights = reg_ridge.coef_
        print('the weight of the sensor_%d' %i + ' is :')
        print(weights)
        model_list.append(reg_ridge)
        preds = reg_ridge.predict(test_X)
        test_label = test_Y[:,i].reshape(-1) 
        loss = np.sqrt(np.mean([(x - a)**2 for x in preds for a in test_label]))
        print('the loss of this model is :', loss)
        plt.figure()
        plt.plot(preds)
        plt.plot(test_label)
        plt.legend(['prediciton', 'label'])
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title('prediciton Vs ground truth')
        plt.savefig('./res/predictions/sensor_%d_linear.png' %i)


def xgbModel(train_X, test_X, train_Y, test_Y):
    print('training the XGBoost models ......')
    model_list = []
    train_X = train_X.reshape(train_X.shape[0] * 14520, -1)
    train_Y = train_Y.reshape(train_Y.shape[0] * 14520, -1)
    test_X = test_X.reshape(test_X.shape[0] * 14520, -1)
    test_Y = test_Y.reshape(test_Y.shape[0] * 14520, -1)
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape) 
    params={
        'booster':'gbtree',
        'objective': 'reg:gamma',
        'gamma':0.1,
        'max_depth':6, 
        'lambda':2,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'min_child_weight':3, 
        'silent':0 ,
        'eta': 0.05, 
        'seed':1000,
        #'nthread':7,
        }
    plst = list(params.items())
    # TODO:divided the training set again to check training status.......
    num_rounds = 200
    for i in range(train_Y.shape[1]):
        train_label = train_Y[:,i].reshape(-1)
        print(train_label.shape)
        xgb_train = xgb.DMatrix(train_X, label=train_label)
        xgb_test = xgb.DMatrix(test_X)
        watchlist = [(xgb_train, 'train')]
        model = xgb.train(plst, xgb_train, num_rounds, watchlist)
        model_list.append(model)
        preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
        test_label = test_Y[:,i].reshape(-1)
        loss = np.sqrt(np.mean([(x - a)**2 for x in preds for a in test_label]))
        print('the weight of the sensor_%d' %i + ' is :', loss) 
        plt.figure()
        plt.plot(preds)
        plt.plot(test_label)
        plt.legend(['prediciton', 'label'])
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title('prediciton Vs ground truth')
        plt.savefig('./res/predictions/sensor_%d_xgb.png' %i)

if __name__ == '__main__':
    input_dir = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/New_injector_failure_data/unzip/'
    train_X, test_X, train_Y, test_Y = loadData(input_dir)
    linearModel(train_X, test_X, train_Y, test_Y)
    xgbModel(train_X, test_X, train_Y, test_Y)

