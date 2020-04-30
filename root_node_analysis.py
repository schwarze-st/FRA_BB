from fra_heur import *
from algorithm_analysis import *
import logging
from datetime import *
import os
import tracemalloc

tracemalloc.start()
globalpath_name = '/home/stefan/Dokumente/opt_problems/'
folder_name = 'collection'
logging.basicConfig(level=logging.INFO, filename='results/root_node_analysis_log_'+folder_name)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
logging.info('>>>>>>>>>>>>>>>> Running testbed ' + folder_name + " "+ dt_string + "<<<<<<<<<<<<<<<<<<<")
testbed = read_all_test_instances(globalpath_name+folder_name)
#testbed = ['set3-09', 'supportcase20', 'set3-20', 'iis-hc-cov', 'sp150x300d', 'b1c1s1', 'mik-250-20-75-3', 'neos-1445743', 'n13-3', 'khb05250', 'mik-250-20-75-5', 'gen-ip021', 'opm2-z6-s1', 'seymour', 'beasleyC2', 'gsvm2rl3', 'ger50-17-trans-dfn-3t', 'beasleyC3', 'mik-250-20-75-2', 'neos-4954672-berkel', 'k16x240b', 'neos-3118745-obra', 'mc11', 'ger50-17-ptp-pop-6t', 'ger50-17-trans-pop-3t', 'markshare2', 'neos-3611689-kaihu', 'set3-10', 'mc7', 'n6-3', 'gen-ip054', 'ns4-pr6', 'bg512142', 'ger50-17-ptp-pop-3t', 'neos-1445765', 'gen-ip002']
testbed = ['set3-09', 'supportcase20']
print('Testing %i problems'%len(testbed))
error_probs = []

for idx, model_name in enumerate(testbed):
    logging.info('testing problem ' + model_name)
    print('Testing model %s'%model_name)
    path_name = globalpath_name + '/' + folder_name+'/'+ model_name+'.mps'
    try:
        test_heur(path_name, model_name)
    except:
        print('unexpected error occurred')
        error_probs.append(model_name[:-4])
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    if idx > 4:
        break
os.rename('temp_results.pickle','results/FRA_Scip.pickle')
convert_dict_to_dataframe()
print('Problems with errors:',error_probs)
tracemalloc.stop()


