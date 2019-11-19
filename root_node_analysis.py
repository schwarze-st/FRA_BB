from fra_heur import *
from algorithm_analysis import *
import logging

logging.basicConfig(level=logging.INFO)
folder_name = 'testbed2'
testbed = read_all_test_instances(folder_name)
print('Testing %i problems'%len(testbed))
problems_with_eq_constrs = []
for model_name in testbed:
    print('Testing model %s'%model_name)
    path_name = folder_name+'/'+ model_name
    m = test_heur(path_name)
    m.printStatistics()
    del m

print(testbed)