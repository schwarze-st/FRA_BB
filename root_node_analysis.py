from fra_heur import *
from algorithm_analysis import *
import logging
from datetime import *
import os

globalpath_name = ''
folder_name = 'collection'

logging.basicConfig(level=logging.INFO, filename='results/root_node_analysis_log_'+folder_name)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
logging.info('>>>>>>>>>>>>>>>> Running testbed ' + folder_name + " "+ dt_string + "<<<<<<<<<<<<<<<<<<<")

#testbed = read_all_test_instances(globalpath_name+folder_name)
testbed = ['set3-09',
 'set3-20',
# 'buildingenergy',
 'set3-10',
 'neos-1367061',
 'bg512142',
 'gsvm2rl9',
 'stein9inf',
 'gsvm2rl12',
 'seymour1',
 'set3-16',
 'p2m2p1m1p0n100',
 'mushroom-best',
 'gsvm2rl11',
 'neos-1112787',
 'dg012142',
 'neos-787933',
 'supportcase20',
 'ns4-pr6',
 'manna81',
 'app2-1',
 'gr4x6',
 'haprp',
 '30_70_45_095_100',
 'markshare_5_0',
 'iis-hc-cov',
 'b1c1s1',
 'n13-3',
 'khb05250',
 'gen-ip021',
 'seymour',
 'gsvm2rl3',
 #'bmocbd3',
 'neos-3118745-obra',
 'ger50-17-ptp-pop-6t',
 'markshare2',
 'neos-3611689-kaihu',
 'gen-ip054',
 'adult-regularized',
 'ger50-17-ptp-pop-3t',
 'gen-ip002',
 'neos-3611447-jijia',
 'p500x2988',
 'b2c1s1',
 'iis-glass-cov',
 'breastcancer-regularized',
 'app2-2',
 'g200x740',
 'scpj4scip',
 'n7-3',
 'supportcase39',
 'v150d30-2hopcds',
 'stockholm',
 'a1c1s1',
 'n9-3',
 'ran12x21',
 'neos-848198',
 'neos-3610051-istra',
 'scpk4',
 'scpm1',
 'stein45inf',
 'qiu',
 'gsvm2rl5',
 'markshare1',
 'neos5',
 'scpl4',
 'germany50-UUM',
 'stein15inf',
 'supportcase42',
 'gen-ip036',
 'npmv07']
print('Testing %i problems'%len(testbed))
error_probs = []

for idx, model_name in enumerate(testbed):
    logging.info('Testing model ' + model_name)
    print('Testing model ' + model_name)
    print('Testing problem ' + str(idx+1) + ' of ' + str(len(testbed)))
    path_name = folder_name+'/'+ model_name+'.mps'
    try:
        test_heur(path_name, model_name)
    except:
        print('unexpected error occurred')
        error_probs.append(model_name[:-4])
os.rename('temp_results.pickle','results/FRA_Scip.pickle')
convert_dict_to_dataframe()
print('Problems with errors:',error_probs)


