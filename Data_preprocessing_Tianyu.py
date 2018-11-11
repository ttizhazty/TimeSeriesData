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
        print('All the files are unzipped......')

    def encodeData(self, line):
        line = line.strip('\n').split(',')
        line = line[1:]
        encoded_data = [0] * 88
        if line[0] != '--':
            idx = [0] * 88
            # Encode the gear level into number
            if line[4] == 'Neutral':
               encoded_data[4] = 0
            elif line[4][0] == '1':
                encoded_data[4] = 1
            elif line[4][0] == '2':
                encoded_data[4] = 2
            elif line[4][0] == '3':
                encoded_data[4] = 3
            elif line[4][0] == '4':
                encoded_data[4] = 4
            elif line[4][0] == '5':
                encoded_data[4] = 5
            else:
                encoded_data[4] = 0
            idx[4] = 1

            # Encode the Clutch Pedal Depressed into number
            if line[6] == 'Clutch Pedal Depressed (踏んだ)':
                encoded_data[6] = 1
            else:
                encoded_data[6] = 0
            idx[6] = 1

            if line[7] == 'ブレーキペダル踏んだ':
                encoded_data[7] = 1
            else:
                encoded_data[7] = 0
            idx[7] = 1

            if line[8] == 'OFF(クルーズ制御終了もしくは不可)':
                encoded_data[8] = 0
            else:
                encoded_data[8] = 1
            idx[8] = 1

            if line[19] == 'DPFステータス0':
                encoded_data[19] = 1
            else:
                encoded_data[19] = 0
            idx[19] = 1

            if line[24] == 'ITHモーター過電流防止Duty制限なし':
                encoded_data[24] = 0
            else:
                encoded_data[24] = 1
            idx[24] = 1

            if line[25] == 'EGRモーター過電流防止Duty制限なし':
                encoded_data[25] = 0
            else:
                encoded_data[25] = 1
            idx[25] = 1

            if line[26] == 'EGRモーター2過電流防止Duty制限なし':
                encoded_data[26] = 0
            else:
                encoded_data[26] = 1 
            idx[26] = 1

            if line[55] == 'M/V OFF':
                encoded_data[55] = 0
            else:
                encoded_data[55] = 1
            idx[55] = 1

            if line[56] == 'M/V OFF':
                encoded_data[56] = 0
            else:
                encoded_data[56] = 1 
            idx[56] = 1

            if line[57] == 'M/V OFF':
                encoded_data[57] = 0
            else:
                encoded_data[57] = 1
            idx[57] = 1

            if line[58] == 'ON':
                encoded_data[58] = 1
            else:
                encoded_data[58] = 0
            idx[58] = 1

            for i in range(len(idx)):
                if idx[i] == 0:
                    encoded_data[i] = float(line[i])
            return encoded_data
        else:
            return encoded_data

    def loadData(self):
        print(self.unzip_file_path[:-3] + '.csv')
        with open(self.unzip_file_path[:-3] + '.csv', 'r', encoding = 'Shift-JIS') as fr:
            cnt = 0
            for line in tqdm(fr):
                if cnt == 0:
                    print(line.split(','))
                if cnt > 10000000:
                    break
                cnt += 1
                try:
                    encoded_sample = self.encodeData(line)
                except ValueError:
                    continue
                self.time_series_data.append(encoded_sample)

    def dataSegement_by_1_Hour(self):             # one hour collecter 121 * 5 = 605(sample size)  
        start = 0 
        end = 14520
        print(len(self.time_series_data))
        for i in tqdm(range(len(self.time_series_data) // 14520)):    # the setp size of the silding window
            output_file_path = self.unzip_file_path[:-3] +'_%d_ day' %i + '.pkl'
            one_hour_sample = np.array(self.time_series_data[start:end])
            pickle.dump(one_hour_sample, open(output_file_path, 'wb'))
            start += 14520
            end += 14520

    def dataSegement_by_2_Hour(self):
        #TODO: later make more wide slice
        pass
        
if __name__ == '__main__':
    input_file_dir_path = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/New_injector_failure_data/'
    unzip_file_dir_path = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/New_injector_failure_data/unzip/'
    zip_files = listdir(input_file_dir_path)

    for files in zip_files:
        input_file_path = input_file_dir_path + files
        unzip_file_path = unzip_file_dir_path + files
        time_series_data = TimeSeriesData(input_file_path, unzip_file_dir_path, unzip_file_path)
        time_series_data.unzipFile()
        try:
            time_series_data.loadData()
            time_series_data.dataSegement_by_1_Hour()
            #os.system('rm ' + unzip_file_path[:-3]+ '.csv')
        except FileNotFoundError:
            continue

