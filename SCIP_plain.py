import sys
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE


def SCIP_test():

    folder_name = 'collection'
    testbed_name = '12_instances_with_multiple_seeds'

    f = open('results/' +testbed_name+ '.txt', 'w')
    sys.stdout = f

    global testbed, best_sols, idx

    testbed = ['b1c1s1', 'b2c1s1', 'dg012142', 'gsvm2rl11', 'gsvm2rl12', 'gsvm2rl9',
               'mushroom-best', 'neos-983171', 'opm2-z10-s4', 'opm2-z8-s0', 'sorrell7',
               'sorrell8']
    best_sols = [69071.5, 68701.5, 14373382.6, 39792.8, 34.4, 13611.9, 2072.9, 8747, -22681, -11328, -160, -324]

    for idx, model_name in enumerate(testbed):
        k = 0
        while k<5:
            print('>>>>> Testing model ',model_name)
            print('>>>>>>>>>>>>>>> Run ',k+1)
            m = Model(problemName=model_name)
            m.redirectOutput()
            model_path = folder_name+'/'+ model_name+'.mps'
            m.readProblem(model_path)
            if k==0:
                print('>>>>>>>>>> Default random seed')
                pass
            else:
                print('>>>>>>>>>> Seed ',k)
                m.setIntParam('randomization/lpseed', ((k-1)+5)*100)
            m.setRealParam("limits/time", 1800)

            eventhdlr = FoundBetterSol()
            m.includeEventhdlr(eventhdlr, "FoundBetterSol", "python event handler to stop process, when sol_SCIP < sol_diving")
            #m.hideOutput(True)
            m.optimize()
            m.printStatistics()
            m.freeProb()
            del m
            k = k+1


    sys.stdout.close()


class FoundBetterSol(Eventhdlr):

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        assert event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND
        obj_SCIP = self.model.getSolObjVal(self.model.getBestSol(), original=True)
        if obj_SCIP < best_sols[idx]:
            print('>>>>>>>>>>>>>>>>>>>>>>> Obj SCIP: ', obj_SCIP)
            print('>>>>>>>>>>>>>>>>>>>>>>> Obj Div : ', best_sols[idx])
            self.model.interruptSolve()



if __name__ == "__main__":
    SCIP_test()