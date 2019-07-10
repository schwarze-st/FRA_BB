
import gc, weakref, pytest
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum

#
# heuristic: feasibe rounding approach
#

class feasiblerounding(Heur):

    def heurinitsol(self):
        print(">>>> call heurinitsol()")

    def heurexitsol(self):
        print(">>>> call heurexitsol()")

    # execution method of the heuristic
    def heurexec(self, heurtiming, nodeinfeasible):

        print(">>>> Call feasible rounding heuristic at a node with depth %d" % (self.model.getDepth()))

        # TODO:(Wichtig, damit IPS größer wird) Fixierte Variablen müssen aus dem LP rausgenommen werden / oder ein neues LP ohne die Variablen erstellen

        # Modell für innere Parallelmenge erstellen
        print(">>>> Build inner parallel set")
        ips_model = Model("ips")
        # Variablen zum Modell hinzufügen
        variables = self.model.getVars()
        for v in variables:
            ips_model.addVar(name=v.name,
                             vtype=v.vtype(),
                             lb=v.getLbLocal(),
                             ub=v.getUbLocal(),
                             obj=v.getObj())

        # Zielfunktion
        ips_model.setObjective(self.model.getObjective())

        # Durch Rows iterieren und modifizierte Ungleichungen hinzufügen
        linear_rows = self.model.getLPRowsData()
        for lrow in linear_rows:
            vlist = [i.getVar() for i in lrow.getCols()]
            clist = lrow.getVals()
            beta = 0
            for i in range(len(clist)):
                if (vlist[i].vtype()!='CONTINUOUS'):
                    beta += abs(clist[i])

            ips_model.addCons(quicksum(vlist[i]*clist[i] for i in range(len(vlist))) <= lrow.getRhs()-0.5*beta)

        ips_model.optimize()
        print(">>>> Optimized over inner parallel set")

        # Platzhalter: create solution
        sol = self.model.createSol()
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
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE, freq=0)

    # read problem from file
    m.readProblem('/home/stefan/Downloads/10teams.mps')

    # optimize problem
    m.optimize()

    # free model explicitly
    del m

if __name__ == "__main__":
    test_heur()
