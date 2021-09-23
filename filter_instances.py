from algorithm_analysis import *
import logging
import pandas as pd
from datetime import *
import os

folder_name = 'collection'
file_name = 'results/filter_'+folder_name+'_log'
logging.basicConfig(level=logging.INFO, filename=file_name)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
logging.info('>>>>>>>>>>>>>>>> Running testbed ' + folder_name + " "+ dt_string + "<<<<<<<<<<<<<<<<<<<")
testbed = read_all_test_instances(folder_name)
print('Filtering %i problems'%len(testbed))
error_probs = []

for idx, model_name in enumerate(testbed):
    if idx<1007:
        pass
    else:
        logging.info('testing problem ' + model_name)
        print('Testing model %s'%model_name)
        print('... Which is Number ' + str(idx+1) + ' of ' + str(len(testbed)))
        path_name = folder_name+'/'+ model_name
        try:
            m = test_heur(path_name, model_name)
            del m
        except:
            print('unexpected error occurred')
            print(model_name[:-4])
            logging.info('Error in problem ' + model_name)
            error_probs.append(model_name[:-4])

os.rename('temp_results.pickle',file_name+'.pickle')
convert_dict_to_dataframe(file_name+'.pickle')
print('Problems with errors:',error_probs)

