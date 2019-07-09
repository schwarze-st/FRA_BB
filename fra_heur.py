
import gc, weakref, pytest
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_PARAMSETTING, SCIP_HEURTIMING, quicksum
from pyscipopt.scip import is_memory_freed
#from util import is_optimized_mode

#
# heuristic
#

class RoundingHeur(Heur):

    def heurinitsol(self):
        print(">>>> call heurinitsol()")

    def heurexitsol(self):
        print(">>>> call heurexitsol()")

    # execution method of the heuristic
    def heurexec(self, heurtiming, nodeinfeasible):

        print(">>>> Call feasible rounding heuristisc at a node with depth %d" % (self.model.getDepth()))

        # TODO:(Wichtig, damit IPS größer wird) Fixierte Variablen müssen aus dem LP rausgenommen werden / oder ein neues LP ohne die Variablen erstellen

        # Modell für innere Parallelmenge erstellen
        ips_model = Model()
        variables = self.model.getVars()
        for v in variables:
            ips_model.addVar(name=v.name,
                             vtype=v.vtype(),
                             lb = v.getLbLocal(),
                             ub = v.getUbLocal(),
                             obj = v.getObj())

        # TODO: Objective hinzufügen

        linear_rows_data = self.model.getLPRowsData()  # linear rows of current LP-solution in one List: Methods from 'Row'-class can be applied to them
        print('>>> Execution of Rows')
        # iterate through all rows and update the RHS
        for lrow in linear_rows_data:
            vlist = [i.getVar() for i in lrow.getCols()]
            clist = lrow.getVals()
            beta = 0
            for i in range(len(clist)):
                if (vlist[i].vtype()!='CONTINUOUS'):
                    beta += abs(clist[i])
            ips_model.addCons(quicksum([vlist[i]*clist[i] for i in range(len(vlist))]) <= lrow.getRhs()-0.5*beta)

        ips_model.optimize()

        print('>>> Done with Rows')
        # create solution
        sol = self.model.createSol()
        #for v in vars:
        #   self.model.setSolVal(sol, v, 0.0)
        accepted = self.model.trySol(sol)
        print(">>>> accepted solution? %s" % ("yes" if accepted == 1 else "no"))

        if accepted:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}


def test_heur():

    # create model
    m = Model()

    # create and add heuristic to SCIP
    heuristic = RoundingHeur()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE, freq=0)

    # read problem from file
    m.readProblem('/home/stefan/Downloads/10teams.mps')

    # optimize problem
    m.optimize()

    # free model explicitly
    del m

if __name__ == "__main__":
    test_heur()
