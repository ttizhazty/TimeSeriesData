import os
import numpy as np 
import pickle
import pdb
import os

def encodeData(line):
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

def dataParseByDay(input_path, input_file, output_path):
	ID = input_file[16:23]
	suffix = input_file[-13:-4]
	print('Truck ID is : ',ID)
	with open(input_path + input_file, 'r', encoding = 'Shift-JIS') as f:
		all_data = f.readlines()
		one_day_data = []
		for i in range(1,len(all_data)-1):
			line_1 = all_data[i].strip('\n').split(',')
			line_2 = all_data[i+1].strip('\n').split(',')
			date_1 = line_1[0].split(' ')[0]
			date_2 = line_2[0].split(' ')[0]
			if date_1 == date_2:
				sample = encodeData(line_1)
				if sample != [0] * 88:
					one_day_data.append(sample)
			else:
				if os.path.isdir(output_path + '3002LV234N3-----' + ID + 'output'+ suffix + '/'):
					date = '_'.join(date_1.split('/'))
					pickle.dump(one_day_data, open(output_path + '3002LV234N3-----' + ID + 'output'+ suffix + '/' + ID + date + '.pkl', 'wb'))
					one_day_data = []
				else:
					os.mkdir(output_path + '3002LV234N3-----' + ID + 'output'+ suffix + '/')
					date = '_'.join(date_1.split('/'))
					pickle.dump(one_day_data, open(output_path + '3002LV234N3-----' + ID + 'output' + suffix + '/' + ID + date + '.pkl', 'wb'))
					one_day_data = []

if __name__ == '__main__':
    input_path = './../raw_data/'
    output_path = './../processed_data/'
    file_list = os.listdir(input_path)
    for _file in file_list:
        dataParseByDay(input_path, _file, output_path)