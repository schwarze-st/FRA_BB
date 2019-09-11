import glob
import math
import os

from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum, Expr

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

        # TODO:(Wichtig, damit IPS größer wird) Fixierte Variablen müssen aus dem LP rausgenommen werden / oder ein
        #  neues LP ohne die Variablen erstellen

        # Modell für innere Parallelmenge erstellen
        print(">>>> Build IPS")
        ips_model = Model("ips")

        # Variablen zum Modell hinzufügen + vergrößern der Box-Constraints
        variables = self.model.getVars(transformed=True)
        variables_lpval = []
        ips_vars = dict()
        for v in variables:
            lower = v.getLbLocal()
            upper = v.getUbLocal()
            variables_lpval.append(self.model.getVal(v))
            if v.vtype() != 'CONTINUOUS':
                if lower is not None:
                    lower = math.ceil(lower) - delta + 0.5
                if upper is not None:
                    upper = math.floor(upper) + delta - 0.5
            ips_vars[v.name] = ips_model.addVar(name=v.name, vtype='CONTINUOUS', lb=lower, ub=upper, obj=v.getObj())

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

        # Durch Rows iterieren und modifizierte Ungleichungen hinzufügen
        linear_rows = self.model.getLPRowsData()
        number_of_rows = len(linear_rows)

        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()
            if const:
                print('>>>>> lrow Constant: ', const)

            beta = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype() != 'CONTINUOUS')

            lhs = lrow.getLhs()
            rhs = lrow.getRhs()
            # Vergrößerte IPM
            if all(vlist[i].vtype() != 'CONTINUOUS' for i in range(len(clist))) and all(
                    clist[i].is_integer() for i in range(len(clist))):
                lhs = math.ceil(lrow.getLhs()) + 0.5 * beta - delta
                rhs = math.floor(lrow.getRhs()) - 0.5 * beta + delta
            else:
                lhs = lrow.getLhs() + 0.5 * beta
                rhs = lrow.getRhs() - 0.5 * beta

            # Unlösbarkeit abfangen
            if lhs > (rhs + 10 ** (-6)):
                solvable_ips = False
                break

            # Ungleichung dem Modell hinzufügen
            ips_model.addCons(
                lhs <= (quicksum(ips_vars[vlist[i].name] * clist[i] for i in range(len(vlist))) + const <= rhs))

        print('>>>> Total number of LP-Rows: ', number_of_rows)

        if solvable_ips:
            print(">>>> Optimize over EIPS")
            ips_model.optimize()
            if ips_model.getStatus() == 'optimal':
                ips_optimal_solved.append(problem_number)

            sol = self.model.createSol()
            for v in variables:
                val_ips = ips_model.getVal(ips_vars[v.name])
                if v.vtype() != 'CONTINUOUS':
                    val_ips = round(val_ips)
                self.model.setSolVal(sol, v, val_ips)

            accepted = self.model.trySol(sol, printreason=True, completely=False, checkbounds=True,
                                         checkintegrality=True, checklprows=True, free=True)

            print(">>>> accepted solution? %s" % ("yes" if accepted == 1 else "no"))

            if accepted:
                granular_problems.append(problem_number)
                return {"result": SCIP_RESULT.FOUNDSOL}
            else:
                vio = []
                for row in linear_rows:
                    vio.append(self.model.getRowSolFeas(row, sol))
                print('Maximum violation of LP row: ', min(vio))
                max_vio.append((problem_number, min(vio)))
                non_granular_problems.append(problem_number)
                return {"result": SCIP_RESULT.DIDNOTFIND}
        else:
            print('>>>> EIPS is empty')
            non_granular_problems.append(problem_number)
            eq_constrained.append(problem_number)
            return {"result": SCIP_RESULT.DIDNOTFIND}


granular_problems = []
non_granular_problems = []
eq_constrained = []
ips_optimal_solved = []
problem_number = 0
max_vio = []

def test_granularity_rootnode():
    path_to_problems = '/home/stefan/Dokumente/02_HiWi_IOR/Paper_BA/franumeric/selectedTestbed/'
    os.chdir(path_to_problems)
    problem_names = glob.glob("*.mps")

    global granular_problems, non_granular_problems, eq_constrained, problem_number
    number_of_instances = len(problem_names)

    for i in [34, 43, 46, 18, 50, 54, 29, 31]:
        problem_number += i
        problem = problem_names[i-1]

        # create model
        m = Model()

        # create and add heuristic to SCIP
        heuristic = feasiblerounding()
        m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                      freq=0)

        m.setParam("limits/time", 45)

        # read exemplary problem from file
        print('>>>>> Working on Problem: ', problem, ', which is number ', problem_number, 'of ', number_of_instances)
        m.readProblem("".join([path_to_problems, problem]))

        # optimize problem
        m.optimize()

        # free model explicitly
        del m

    print(granular_problems)
    print(non_granular_problems)
    print(eq_constrained)
    print(ips_optimal_solved)
    print(max_vio)

    print(">>>>> Auswertung")
    print("Tested Instances", problem_names)
    print("Instances in Testbed: ", number_of_instances)
    print("granular instances: ", len(set(granular_problems)))
    print("Non granular Problems: ", len(set(non_granular_problems)))
    print("Equality constrained", len(set(eq_constrained)))
    print("Both (because it restarted)", len(set(granular_problems).intersection(set(non_granular_problems))))
    print("Not tested because of time-limit: ",
          number_of_instances - len(set(granular_problems)) - len(set(non_granular_problems)) + len(
              set(granular_problems).intersection(set(non_granular_problems))))


def test_heur():
    m = Model()
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                  freq=0)
    m.readProblem('/home/stefan/Dokumente/02_HiWi_IOR/Paper_BA/franumeric/selectedTestbed/fixnet6.mps')
    m.optimize()
    del m


if __name__ == "__main__":
    # test_heur()
    test_granularity_rootnode()
