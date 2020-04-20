from fra_heur import *
from algorithm_analysis import *
import logging
from datetime import *
import os
import tracemalloc

tracemalloc.start()
folder_name = 'collection'
logging.basicConfig(level=logging.INFO, filename='results/root_node_analysis_log_'+folder_name)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
logging.info('>>>>>>>>>>>>>>>> Running testbed ' + folder_name + " "+ dt_string + "<<<<<<<<<<<<<<<<<<<")
testbed = read_all_test_instances(folder_name)
print('Testing %i problems'%len(testbed))
error_probs = []

for idx, model_name in enumerate(testbed):
    logging.info('testing problem ' + model_name)
    print('Testing model %s'%model_name)
    path_name = folder_name+'/'+ model_name
    try:
        test_heur(path_name, model_name)
    except:
        print('unexpected error occurred')
        error_probs.append(model_name[:-4])
    current, peak = tracemalloc.get_traced_memory()
    logging.INFO(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
os.rename('temp_results.pickle','results/FRA_Scip.pickle')
convert_dict_to_dataframe()
print('Problems with errors:',error_probs)
tracemalloc.stop()


