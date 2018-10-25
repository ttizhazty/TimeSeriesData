import pickle
import numpy as np
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import pdb

# sample data: ['2017/03/07 07:15:34.50', '0', '16', '561.125', '26.234375', '1st Gear', '3.28125', 'Clutch Pedal Depressed (踏んだ)', 'ブレーキペダル踏んだ', 'OFF(クルーズ制御終了もしくは不可)', '88', '48', '0', '31', '0', '99.5', '16', '4', '270', '290', 'DPFステータス0', '0', '0', '2', '0', 'ITHモーター過電流防止Duty制限なし', 'EGRモーター過電流防止Duty制限なし', 'EGRモーター2過電流防止Duty制限なし', '30', '0.882', '18', '41.5', '38', '-4', '55', '59.5', '28', '35', '1012', '1002.5', '1510', '29', '92.4', '92', '298', '3', '-12', '0', '16', '0.75', '-0.75', '-0.5', '1', '0', '0', '575', 'M/V OFF', 'M/V OFF', 'M/V OFF', 'ON', '0', '0', '493.3', '101', '255', '65535', '65535', '255', '0', '13', '0', '-3', '0', '13', '0', '0', '-0.07275390625', '8', '0', '0', '0', '0', '-1', '-9', '215', '215', '255', '25.5', '0']
class TimeSeriesData:
    def __init__(self, input_file_path, unzip_file_dir_path, unzip_file_path):
        self.input_file_path = input_file_path
        self.unzip_file_dir_path = unzip_file_dir_path
        self.unzip_file_path = unzip_file_path
        
        self.time_series_data = []

    def unzipFile(self):
        os.system('7z x' + ' ' + self.input_file_path + ' -o' +  self.unzip_file_dir_path)

    def encodeData(self, line):
        line = line.strip('\n').split(',')
        encoded_data = []
        '''
        if line[1] != '--':
            # Encode the gear level into number
            if line[5] = 'Neutral':
               encoded_data[5] = 0
            else:
                encoded_data[5] = line[5][0]
            # Encode the Clutch Pedal Depressed into number:
        '''
        if line[1] != '--':
            for data in line:
                try:
                    data = float(data) 
                except ValueError:
                    continue
                encoded_data.append(data)
            return encoded_data
        else:
            return [0] * 76

    def loadData(self):
        #output_file_path = self.unzip_file_path[:-3] + '.pkl'
        with open(self.unzip_file_path[:-3] + '.csv', 'r', encoding = 'Shift-JIS') as fr:
            cnt = 0
            for line in tqdm(fr):
                if cnt > 10000:
                    break
                cnt += 1
                encoded_sample = self.encodeData(line)
                self.time_series_data.append(encoded_sample)
            #pickle.dump(time_series_data, open(output_file_path, 'wb'))

    def dataSegement_by_1_Hour(self):             # one hour collecter 121 * 5 = 605(sample size) 
        #print('I am in a segement function-----------------')
        #print(len(self.time_series_data))
        for i in tqdm(range(0, len(self.time_series_data) // 121 - 3, 3)):    # the setp size of the silding window
            output_file_path = self.unzip_file_path[:-3] +'_%d_ hour' %i + '.pkl'
            #print('I am wirtting hourly file --------------------------------------')
            start = i
            end = i + 605
            one_hour_sample = np.array(self.time_series_data[start:end])
            pickle.dump(one_hour_sample, open(output_file_path, 'wb'))
    
    def dataSegement_by_2_Hour(self):
        #TODO: later make more wide slice
        pass
        
if __name__ == '__main__':
    input_file_dir_path = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/New_injector_failure_data/'
    unzip_file_dir_path = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/New_injector_failure_data/unzip/'
    zip_files = listdir(input_file_dir_path)
    cnt = 0
    for files in zip_files:
        if cnt > 2:
            break
        cnt += 1 
        input_file_path = input_file_dir_path + files
        unzip_file_path = unzip_file_dir_path + files
        time_series_data = TimeSeriesData(input_file_path, unzip_file_dir_path, unzip_file_path)
        time_series_data.unzipFile()
        try:
            time_series_data.loadData()
            time_series_data.dataSegement_by_1_Hour()
            os.system('rm ' + unzip_file_path[:-3]+ '.csv')
        except FileNotFoundError:
            continue

