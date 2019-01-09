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
    
    # day_level_data......
    val_X, val_Y = laodTrainingSample(dir+file_list[102])

    #comment out here to chcek the data distribution on the scale of time line.
    train_X, test_X, train_Y, test_Y = train_test_split(train_X_all.reshape(200*14520, -1), train_Y_all.reshape(200*14520, -1), test_size = 0.1, random_state=7)
    ''' 
    train_X = train_X_all.reshape(96*14520, -1)
    train_Y = train_Y_all.reshape(96*14520, -1)
    print(train_X.shape)
    print(train_Y.shape)
    '''
    print('training data size is: ', train_X.shape)
    print('testing data size is: ', test_Y.shape)
    return train_X, test_X, train_Y, test_Y, val_X, val_Y
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
    #rain_X, test_X, train_Y, test_Y = train_test_split(train_X_all, train_Y_all, test_size = 0.1)
    print(dir+file_list[-1])
    test_X, test_Y = laodTrainingSample(dir+file_list[-1])

    return train_X_all, test_X, train_Y_all, test_Y 

def linearModel(train_X, test_X, train_Y, test_Y, val_X, val_Y):
    print('strat training the linear models ......')
    model_list = []
    '''
    train_X = train_X.reshape(train_X.shape[0] * 14520, -1)
    train_Y = train_Y.reshape(train_Y.shape[0] * 14520, -1)
    test_X = test_X.reshape(test_X.shape[0] * 14520, -1)
    test_Y = test_Y.reshape(test_Y.shape[0] * 14520, -1)
    '''
    print('the data size is ......')
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    reg_linear = linear_model.LinearRegression()
    reg_lasso = linear_model.Lasso()
    reg_ridge = linear_model.Ridge(alpha = 5)
    svr_model = SVR(gamma='scale', C=1.0, epsilon=0.2)

    for i in range(train_Y.shape[1]):
        sensor_idx = raw_problem_idx[i]
        sensor_name = sensor_dict[sensor_idx]
        reg_ridge.fit(train_X, train_Y[:,i])
        #weights = reg_ridge.coef_
        #weights_list = weights.tolist()
        #most_important_sensor_idx = weights_list.sort(reverse=True)[:5]
        #print('the most 5 import features for predicting ' + 'sensor_name: ' + )
        #print('the weight of the sensor_%d' %i + ' is :')
        #print(weights)
        model_list.append(reg_ridge)
        preds = reg_ridge.predict(test_X)
        preds = np.array(preds).reshape(-1)
        test_label = test_Y[:,i].reshape(-1)
        val_pred = np.array(reg_ridge.predict(val_X)).reshape(-1)
        val_label = val_Y[:,i].reshape(-1)
        loss = np.sqrt(np.mean(np.subtract(test_label, preds)**2))
        val_loss = np.sqrt(np.mean(np.subtract(val_label, val_pred)**2))
        print('the loss of this model is :', loss)
        print('the loss of this model on validation set :', val_loss)
        print('i am in plot !!!!!!') 
        plt.figure()
        plt.plot(preds[::1000])
        plt.plot(test_label[::1000])
        #plt.plot(test_label[::2000]*0.85,'r')
        #plt.plot(test_label[::2000]*1.15, 'r')
        plt.legend(['prediciton', 'label', 'lower_bound', 'upper_bound'])
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title(sensor_name + '(loss=%f)' %loss)
        plt.savefig('./res/day_pred/predictions/sensor_%d_linear.png' %i)
        plt.close()
        
        #plt.figure()
        val_pred = reg_ridge.predict(val_X)
        val_pred = np.array(val_pred).reshape(-1)
        val_label = val_Y[:,i].reshape(-1) 
        plt.plot(val_pred[::200])
        plt.plot(val_label[::200])
        plt.plot(val_label[::200]*0.85,'r')
        plt.plot(val_label[::200]*1.15, 'r')
        plt.legend(['prediciton', 'label', 'lower_bound', 'upper_bound'])
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title(sensor_name + '(loss=%f)' %val_loss)
        plt.savefig('./res/day_pred/predictions/oneDay_sensor_%d_linear.png' %i)
        plt.close()
        
    '''
    for j in range(val_Y.shape[1]):
        sensor_idx = raw_problem_idx[j]
        sensor_name = sensor_dict[sensor_idx]
        reg_ridge = model_list[j]
        val_pred = reg_ridge.predict(val_X)
        val_pred = np.array(val_pred).reshape(-1)
        val_label = val_Y[:,j].reshape(-1)
        val_loss = np.sqrt(np.mean(np.subtract(val_label, val_pred)**2))
        #pdb.set_trace()
        print('the loss of this model on validation set :', val_loss)
        plt.figure()
        plt.plot(val_pred[:200])
        plt.plot(val_label[:200])
        plt.legend(['prediciton', 'label'])
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title(sensor_name + '(loss=%f)' %val_loss)
        plt.savefig('./res/day_pred/predictions/oneDay_sensor_%d_linear.png' %j)
        plt.close()
    '''

def xgbModel(train_X, test_X, train_Y, test_Y, val_X, val_Y):
    print('training the XGBoost models ......')
    model_list = []
    #train_X = train_X.reshape(train_X.shape[0] * 14520, -1)
    #train_Y = train_Y.reshape(train_Y.shape[0] * 14520, -1)
    #test_X = test_X.reshape(test_X.shape[0] * 14520, -1)
    #test_Y = test_Y.reshape(test_Y.shape[0] * 14520, -1)
    print('the data size is ......')
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape) 
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
    num_rounds = 150
    for i in range(train_Y.shape[1]):
        sensor_idx = raw_problem_idx[i]
        sensor_name = sensor_dict[sensor_idx]
        train_label = np.abs(train_Y[:,i]).reshape(-1)
        print(train_label.shape)
        xgb_train = xgb.DMatrix(train_X, label=train_label)
        xgb_test = xgb.DMatrix(test_X)
        xgb_val = xgb.DMatrix(val_X)
        watchlist = [(xgb_train, 'train')]
        #model = xgb.train(plst1, xgb_train, num_rounds, watchlist)
        #model.save_model('./models/tree_models/sensor_%d' %i + '.mdoel')
        #model_list.append(model)
        print('loading model......')
        model = xgb.Booster()
        model.load_model('./models/tree_models/sensor_%d' %i + '.mdoel')
        preds = model.predict(xgb_test)
        preds = np.array(preds).reshape(-1)
        test_label = test_Y[:,i].reshape(-1)
        val_pred = model.predict(xgb_val)
        val_pred = np.array(val_pred).reshape(-1)
        val_label = val_Y[:,i].reshape(-1)
        loss = np.sqrt(np.mean(np.subtract(test_label, preds)**2))
        val_loss = np.sqrt(np.mean(np.subtract(val_label, val_pred)**2))
        print('the loss of the sensor_%d' %i + ' is :', loss) 
        print('the loss of this model on validation ser : ', val_loss)
        plt.figure()
        #plt.plot(preds[::1000])
        plt.plot(test_label-preds)
        #plt.plot(test_label[::1000]*0.85,'r')
        #plt.plot(test_label[::1000]*1.15, 'r')
        plt.legend(['difference'])
        #plt.legend(['prediciton', 'label', 'lower_bound', 'upper_bound']) 
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title(sensor_name + '(loss=%f)' %loss)
        plt.savefig('./res/day_pred/predictions/sensor_%d_xgb.png' %i)
        plt.close()

        plt.plot(val_pred[::200])
        plt.plot(val_label[::200])
        plt.plot(val_label[::200]*0.85,'r')
        plt.plot(val_label[::200]*1.15, 'r')
        plt.legend(['prediciton', 'label', 'lower_bound', 'upper_bound'])
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title(sensor_name + '(loss=%f)' %val_loss)
        plt.savefig('./res/day_pred/predictions/oneDay_sensor_%d_xgb.png' %i)
        plt.close() 
    
def randomForestModel(train_X, test_X, train_Y, test_Y):
    print('start training random forest model ...')
    model_list = []
    print('the data size is ......')
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    rf_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=4,
           oob_score=False, random_state=0, verbose=1, warm_start=False)
    for i in range(train_Y.shape[1]):
        sensor_idx = raw_problem_idx[i]
        sensor_name = sensor_dict[sensor_idx]
        rf_model.fit(train_X, train_Y[:,i])
        model_list.append(rf_model)
        preds = rf_model.predict(test_X)
        preds = np.array(preds).reshape(-1)
        test_label = test_Y[:,i].reshape(-1)
        loss = np.sqrt(np.mean(np.subtract(test_label, preds)**2)) 
        print('the loss of this model is :', loss)
        plt.figure()
        plt.plot(preds[:200])
        plt.plot(test_label[:200])
        plt.plot(test_label[:200]*0.9,'r.')
        plt.plot(test_label[:200]*1.1, 'r.')
        plt.legend(['prediciton', 'label', 'lower_bound', 'upper_bound'])
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title(sensor_name + '(loss=%f)' %loss)
        plt.savefig('./res/predictions/sensor_%d_rf.png' %i)
        print('tsaining complete for ' + sensor_name + ' !!!')
    
def encodeData(line):
    line = line.strip('\n').split(',')
    line = line[3:]
    sample = []
    if line[0] != '--':
        for i in range(len(line)):
            try:
                sample.append(float(line[i]))
            except ValueError:
                continue
        sample.append(0)
        return sample
    else:
        return [0] * 88


if __name__ == '__main__':
    input_dir = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/New_injector_failure_data/unzip/'
    train_X, test_X, train_Y, test_Y, val_X, val_Y = loadData(input_dir)
    #randomForestModel(train_X, test_X, train_Y, test_Y)
    #linearModel(train_X, test_X, train_Y, test_Y, val_X, val_Y)
    #xgbModel(train_X, test_X, train_Y, test_Y)
    #train_X, train_Y = loadData(input_dir)
    #dataDistributionAnalysis(train_X, train_Y)
    test_data_raw = []
    with open('/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/validation_data/201501_201703/output_7000618_2016_8_2.txt', 'r', encoding = 'Shift-JIS') as f:
        for line in f:
            try:
                sample = encodeData(line)
                if len(sample) == 88:
                    test_data_raw.append(sample)
            except (ValueError, IndexError):
                continue
    with open('/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/validation_data/201501_201703/output_7000618_2016_8_3.txt', 'r', encoding = 'Shift-JIS') as f:
        for line in f:
            try:
                sample = encodeData(line)
                if len(sample) == 88:
                    test_data_raw.append(sample)
            except (ValueError, IndexError):
                continue
    with open('/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/validation_data/201501_201703/output_7000618_2016_8_4.txt', 'r', encoding = 'Shift-JIS') as f:
        for line in f:
            try:
                sample = encodeData(line)
                if len(sample) == 88:
                    test_data_raw.append(sample)
            except (ValueError, IndexError):
                continue
    print(len(test_data_raw))
    test_data = np.array(test_data_raw)
    test_bad_X = test_data[:, correct_idx]
    test_bad_Y = test_data[:, problem_idx]
    #linearModel(train_X, test_bad_X, train_Y, test_bad_Y, val_X, val_Y)
    xgbModel(train_X, test_bad_X, train_Y, test_bad_Y, val_X, val_Y)


