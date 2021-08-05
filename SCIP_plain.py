import sys
from pyscipopt import Model

globalpath_name = ''
folder_name = 'collection'
testbed_name = 'high_diving_performance_new'

#with open('testbed/'+testbed_name+'.txt', "r") as to_read:
#	testbed = to_read.readlines()

f = open('results/' +testbed_name+ '.txt', 'w')
sys.stdout = f

testbed = ['b1c1s1', 'b2c1s1', 'dg012142', 'gsvm2rl11', 'gsvm2rl12', 'gsvm2rl9',
       'mushroom-best', 'neos-983171', 'opm2-z10-s4', 'opm2-z8-s0', 'sorrell7',
       'sorrell8']


for idx, model_name in enumerate(testbed):
    #model_name = model_name[:-5]
    k = 0
    while k<5:
        print('>>>>> Testing model ',model_name)
        print('>>>>>>>>>>>>>>> Run ',k+1)
        m = Model(problemName=model_name)
        m.redirectOutput()
        model_path = folder_name+'/'+ model_name+'.mps'
        m.readProblem(model_path)
        m.setIntParam('randomization/lpseed', (k+5)*100)
        m.setRealParam("limits/time", 1800)
        #m.hideOutput(True)
        m.optimize()
        m.printStatistics()
        m.freeProb()
        del m
        k = k+1


sys.stdout.close()