from fra_heur import *
from algorithm_analysis import *

folder_name = 'benchmark'
testbed = read_all_test_instances(folder_name)
for model_name in testbed:
    path_name = folder_name+'/'+ model_name
    test_heur(path_name)
print(testbed)