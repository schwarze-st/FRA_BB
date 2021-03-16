from fra_heur import *
from algorithm_analysis import *
import logging
from datetime import *
import os

globalpath_name = ''
folder_name = 'collection'
testbed_name = 'granularSCIPsmall'

logging.basicConfig(level=logging.INFO, filename='results/test_paperversion'+folder_name)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
logging.info('>>>>>>>>>>>>>>>> Running testbed ' + testbed_name + " "+ dt_string + "<<<<<<<<<<<<<<<<<<<")

with open('testbed/'+testbed_name+'.txt', "r") as to_read:
	testbed = to_read.readlines()

print('Testing %i problems'%len(testbed))
error_probs = []

for idx, model_name in enumerate(testbed[6:9]):
    model_name = model_name[:-5]
    logging.info('Testing model ' + model_name)
    print('Testing model ' + model_name)
    print('Testing problem ' + str(idx+1) + ' of ' + str(len(testbed)))
    path_name = folder_name+'/'+ model_name+'.mps'
    try:
        test_heur(path_name, model_name)
    except:
        print('unexpected error occurred')
        error_probs.append(model_name)
os.rename('temp_results.pickle','results/FRA_Scip.pickle')
convert_dict_to_dataframe()
print('Problems with errors:',error_probs)


