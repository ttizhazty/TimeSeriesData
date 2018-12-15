import numpy as np 
import pickle
import pdb
import os


def encodeData(line):
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

def data_filtering(input_path, output_path, input_file, output_file):
    truck_ID = input_file[7:14]
    with open(input_path + input_file, 'r', encoding = 'Shift-JIS') as f:
        print('i am in !!!')
        print(truck_ID)
        f.readline() # removing the title
        for line in f:
            print('processing line .......')
            try:
                encode_sample = encodeData(line)
                print(encode_sample)
            except ValueError:
                continue

            

if __name__ == '__main__':
    input_path = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/validation_data/201501_201703/val_data/'
    output_path = '/Users/tianyuzhang/Desktop/Intern/intern/ISUZU/OBD_tianyu/validation_data/201501_201703/output/'
    file_list = os.listdir(input_path)
    for failure_file in file_list:
        data_filtering(input_path, output_path, failure_file, failure_file)
