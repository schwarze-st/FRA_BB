import os, glob
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum, Expr
import math


#
# heuristic: feasible rounding approach
#

class feasiblerounding(Heur):

    def heurinitsol(self):
        print(">>>> call heurinitsol()")

    def heurexitsol(self):
        print(">>>> call heurexitsol()")

    # execution method of the heuristic
    def heurexec(self, heurtiming, nodeinfeasible):
        global granular_problems, non_granular_problems, eq_constrained, problem_number

        delta = 0.999
        solvable_ips = True

        print(">>>> Call feasible rounding heuristic at a node with depth %d" % (self.model.getDepth()))

        # TODO:(Wichtig, damit IPS größer wird) Fixierte Variablen müssen aus dem LP rausgenommen werden / oder ein neues LP ohne die Variablen erstellen

        # Modell für innere Parallelmenge erstellen
        print(">>>> Build IPS")
        ips_model = Model("ips")

        # Variablen zum Modell hinzufügen + vergrößern der Box-Constraints
        variables = self.model.getVars(transformed=True)
        ips_vars = dict()
        for v in variables:
            if v.vtype != 'CONTINUOUS':
                lb = math.ceil(v.getLbLocal()) - delta + 0.5
                ub = math.floor(v.getUbLocal()) + delta - 0.5
                ips_vars[v.name] = ips_model.addVar(name=v.name, vtype='CONTINUOUS', lb=lb, ub=ub, obj=v.getObj())
            else:
                ips_vars[v.name] = ips_model.addVar(name=v.name, vtype='CONTINUOUS', lb=v.getLbLocal(),
                                                    ub=v.getUbLocal(), obj=v.getObj())

        # Zielfunktion setzen 
        obj_sub = Expr()
        zf_sense = self.model.getObjectiveSense()
        k = 0
        for v in variables:
            coeff = v.getObj()
            if coeff != 0:
                obj_sub += coeff * ips_vars[v.name]
                k = 1
        assert k == 1, "Objective Function is empty"
        obj_sub.normalize()
        ips_model.setObjective(obj_sub, sense=zf_sense)

        # zf_ex = self.model.getObjective()
        # zf_ex.normalize()
        # zf_items = zf_ex.terms.items()
        # Nur für lineare Zielfunktionen, Konstanten werden rausgelassen
        # new_expr = quicksum(ips_vars['t_'+v[0].name]*coef for v,coef in zf_items if len(v)!=0)
        # ips_model.setObjective(new_expr,sense=zf_sense)

        # Durch Rows iterieren und modifizierte Ungleichungen hinzufügen
        linear_rows = self.model.getLPRowsData()
        for lrow in linear_rows:
            vlist = [i.getVar() for i in lrow.getCols()]
            clist = lrow.getVals()
            beta = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype != 'CONTINUOUS')

            # print(vlist, clist)
            # Vergrößerte IPM
            current_delta = 0
            if all(vlist[i].vtype != 'CONTINUOUS' for i in range(len(clist))) and all(
                    clist[i].is_integer() for i in range(len(clist))):
                current_delta = delta

            # print(lrow.getLhs(), lrow.getRhs())
            lhs = lrow.getLhs() + 0.5 * beta - current_delta
            rhs = lrow.getRhs() - 0.5 * beta + current_delta
            # print(lhs, rhs)

            # Unlösbarkeit abfangen
            if lhs > (rhs + 10 ** (-6)):
                print('>>>> EIPS is empty')
                solvable_ips = False
                break

            # Ungleichung dem Modell hinzufügen
            ips_model.addCons(lhs <= (quicksum(ips_vars[vlist[i].name] * clist[i] for i in range(len(vlist))) <= rhs))

        if solvable_ips:
            print(">>>> Optimize over EIPS")
            ips_model.optimize()

            sol = self.model.createSol()
            for v in variables:
                if v.vtype != 'CONTINUOUS':
                    self.model.setSolVal(sol, v, round(ips_model.getVal(ips_vars[v.name])))
                else:
                    self.model.setSolVal(sol, v, ips_model.getVal(ips_vars[v.name]))

            accepted = self.model.trySol(sol)
            print(">>>> accepted solution? %s" % ("yes" if accepted == 1 else "no"))

            if accepted:
                granular_problems.append(problem_number)
                return {"result": SCIP_RESULT.FOUNDSOL}
            else:
                non_granular_problems.append(problem_number)
                return {"result": SCIP_RESULT.DIDNOTFIND}
        else:
            non_granular_problems.append(problem_number)
            eq_constrained.append(problem_number)
            return {"result": SCIP_RESULT.DIDNOTFIND}


granular_problems = []
non_granular_problems = []
eq_constrained = []
problem_number = 0


def test_granularity():
    path_to_problems = '/home/stefan/Dokumente/02_HiWi_IOR/Paper_BA/franumeric/selectedTestbed/'
    os.chdir(path_to_problems)
    problem_names = glob.glob("*.mps")

    global granular_problems, non_granular_problems, eq_constrained, problem_number
    number_of_instances = len(problem_names)

    for problem in problem_names:
        problem_number += 1

        # create model
        m = Model()

        # create and add heuristic to SCIP
        heuristic = feasiblerounding()
        m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                      freq=0)

        m.setParam("limits/time", 60)

        # read exemplary problem from file
        print('>>>>> Working on Problem: ', problem, ', which is number ', problem_number, 'of ', number_of_instances)
        m.readProblem("".join([path_to_problems, problem]))

        # optimize problem
        m.optimize()

        # free model explicitly
        del m

    print(">>>>> Auswertung")
    print("Tested Instances", problem_names)
    print("Instances in Testbed: ", number_of_instances)
    print("granular instances: ", len(set(granular_problems)))
    print("Non granular Problems: ", len(set(non_granular_problems)))
    print("Equality constrained", len(set(eq_constrained)))
    print("Both (because it restarted)", len(set(granular_problems).intersection(set(non_granular_problems))))
    print("Not tested because of time-limit: ", number_of_instances-len(set(granular_problems))-len(set(non_granular_problems))+len(set(granular_problems).intersection(set(non_granular_problems))))


def test_heur():
    m = Model()
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                  freq=15)
    m.readProblem('/home/stefan/Dokumente/opt_problems/miplib2010_benchmark/dfn-gwin-UUM.mps')
    m.optimize()
    del m


if __name__ == "__main__":
    # test_heur()
    test_granularity()
