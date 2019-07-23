
import gc, weakref, pytest
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum
import math

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
        delta = 0.99
        gran = True

        print(">>>> Call feasible rounding heuristic at a node with depth %d" % (self.model.getDepth()))

        # TODO:(Wichtig, damit IPS größer wird) Fixierte Variablen müssen aus dem LP rausgenommen werden / oder ein neues LP ohne die Variablen erstellen

        # Modell für innere Parallelmenge erstellen
        print(">>>> Build inner parallel set")
        ips_model = Model("ips")

        # Variablen zum Modell hinzufügen + vergrößern der Box-Constraints
        variables = self.model.getVars(transformed=True)
        ips_vars = dict()
        for v in variables:
            if v.vtype!='CONTINUOUS':
                lb = math.ceil(v.getLbLocal()) - delta + 0.5
                ub = math.floor(v.getUbLocal()) + delta - 0.5
                ips_vars['v.name'] = ips_model.addVar(name=v.name , vtype='CONTINUOUS', lb=lb, ub=ub, obj=v.getObj())
            else:
                ips_vars['v.name'] = ips_model.addVar(name=v.name , vtype='CONTINUOUS', lb=v.getLbLocal(), ub=v.getUbLocal(), obj=v.getObj())

        # Zielfunktion
        # ips_model.setObjective(self.model.getObjective())

        # Durch Rows iterieren und modifizierte Ungleichungen hinzufügen
        linear_rows = self.model.getLPRowsData()
        for lrow in linear_rows:
            vlist = [i.getVar() for i in lrow.getCols()]
            clist = lrow.getVals()

            beta = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype != 'CONTINUOUS')
            # enlarged inner parallel set
            current_delta = 0
            if all(vlist[i].vtype != 'CONTINUOUS' for i in range(len(clist))) and all(clist[i].is_integer() for i in range(len(clist))):
                current_delta = delta
                print('enlargement possible!')
            print(lrow.getLhs())
            print(lrow.getRhs())
            lhs = lrow.getLhs() + 0.5*beta - current_delta
            rhs = lrow.getRhs() - 0.5*beta + current_delta

            if lhs > (rhs + 10**(-6)):
                print('(Sub-)Problem is not granular')
                gran = False
                break

            ips_model.addCons(lhs <= (quicksum(ips_vars[vlist[i].name]*clist[i] for i in range(len(vlist))) <= rhs))

        if gran:

            print(">>>> Optimize over inner parallel set")
            ips_model.optimize()

            # TODO: Rundung vom Optimum als Lösung übergeben
            sol = self.model.createSol()
            accepted = self.model.trySol(sol)
            print(">>>> accepted solution? %s" % ("yes" if accepted == 1 else "no"))

            if accepted:
                return {"result": SCIP_RESULT.FOUNDSOL}
            else:
                return {"result": SCIP_RESULT.DIDNOTFIND}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}


def test_heur():

    # create model
    m = Model()

    # create and add heuristic to SCIP
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE, freq=0)

    # read exemplary problem from file
    m.readProblem('/home/stefan/Dokumente/02_HiWi_IOR/mps-files/50v-10.mps')

    # optimize problem
    m.optimize()

    # free model explicitly
    del m

if __name__ == "__main__":
    test_heur()
