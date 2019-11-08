from fra_heur import *
from algorithm_analysis import *
import logging

logging.basicConfig(level=logging.INFO)
folder_name = 'selectedTestbed'
testbed = read_all_test_instances(folder_name)
print('Testing %i problems'%len(testbed))
for model_name in testbed:
    print('Testing model %s'%model_name)
    path_name = folder_name+'/'+ model_name
    test_heur(path_name)
print(testbed)