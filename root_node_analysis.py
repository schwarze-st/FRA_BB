from fra_heur import *
from algorithm_analysis import *
import logging
import pandas as pd
from datetime import *
import os


folder_name = 'benchmark2'
logging.basicConfig(level=logging.INFO, filename='results/root_node_analysis_log_'+folder_name)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
logging.info('>>>>>>>>>>>>>>>> Running testbed ' + folder_name + " "+ dt_string + "<<<<<<<<<<<<<<<<<<<")
testbed = read_all_test_instances(folder_name)
print('Testing %i problems'%len(testbed))
results_list = []
problem_names = []
error_probs = []

for idx, model_name in enumerate(testbed):
    logging.info('testing problem ' + model_name)
    print('Testing model %s'%model_name)
    path_name = folder_name+'/'+ model_name
    try:
        m = test_heur(path_name, model_name)
        m.writeStatistics("statistics")
        with open('statistics') as f:
            content = f.readlines()
        for line in content:
            if 'PyHeur' in line:
                line_of_interest = line.split()
                line_of_interest.remove(':')
                results_list.append(line_of_interest)
        del m
        print(results_list)
        os.remove('statistics')
        problem_names.append(model_name[:-4])
    except:
        print('unexpected error occurred')
        results_list.append([float('inf')]*6)
        error_probs.append(model_name[:-4])
    results_frame = pd.DataFrame(results_list, columns = ['Heuristic','ExecTime','SetupTime','Calls','Found','Best'])
    results_frame.index = testbed[:idx+1]
    results_frame.to_pickle('results/'+folder_name+'results')
    print(results_frame)

os.rename('temp_results.pickle','results/FRA_Scip.pickle')
convert_dict_to_dataframe(problem_names)
print(error_probs)

