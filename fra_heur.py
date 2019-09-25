import glob
import math
import os
import logging

from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum, Expr

#
# heuristic: feasible rounding approach
#


class feasiblerounding(Heur):

    def add_vars_and_bounds(self, local_model, original_vars, enlarged=True, fix_binaries=False, delta=0.999):
        var_dict = dict()
        for var in original_vars:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS' and enlarged:
                if fix_binaries and round(var.getLPSol()) == var.getLPSol() and var.vtype() == 'BINARY':
                    lower = upper = var.getLPSol()
                else:
                    if lower is not None:
                        lower = math.ceil(lower) - delta + 0.5
                    if upper is not None:
                        upper = math.floor(upper) + delta - 0.5
            var_dict[var.name] = local_model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper, obj=var.getObj())

        return local_model, var_dict

    def heurinitsol(self):
        print(">>>> call heurinitsol()")

    def heurexitsol(self):
        print(">>>> call heurexitsol()")

    # execution method of the heuristic
    def heurexec(self, heurtiming, nodeinfeasible):
        delta = 0.999

        logging.info(">>>> Call feasible rounding heuristic at a node with depth %d" % (self.model.getDepth()))
        logging.info(">>>> Build inner parallel set")

        variables = self.model.getVars(transformed=True)
        ips_model = Model("ips")
        ips_model, ips_vars = self.add_vars_and_bounds(ips_model, variables, enlarged=True, fix_binaries=False, delta=0.999)

        # TODO:(Wichtig, damit IPS größer wird) Fixierte Variablen müssen aus dem LP rausgenommen werden / oder ein
        #  neues LP ohne die Variablen erstellen

        # Zielfunktion setzen
        obj_sub = Expr()
        zf_sense = self.model.getObjectiveSense()
        k = 0
        for v in variables:
            coefficient = v.getObj()
            if coefficient != 0:
                obj_sub += coefficient * ips_vars[v.name]
                k = 1
        if k == 0:
            logging.warning("Objective function is empty")
        obj_sub.normalize()
        ips_model.setObjective(obj_sub, sense=zf_sense)

        # Durch Rows iterieren und modifizierte Ungleichungen hinzufügen
        linear_rows = self.model.getLPRowsData()

        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()
            lhs = lrow.getLhs()
            rhs = lrow.getRhs()


            beta = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype() != 'CONTINUOUS')
            # Vergrößerte IPM
            if all(vlist[i].vtype() != 'CONTINUOUS' for i in range(len(clist))) and all(
                    clist[i].is_integer() for i in range(len(clist))):
                lhs = math.ceil(lrow.getLhs()) + 0.5 * beta - delta
                rhs = math.floor(lrow.getRhs()) - 0.5 * beta + delta
            else:
                lhs = lrow.getLhs() + 0.5 * beta
                rhs = lrow.getRhs() - 0.5 * beta

            if lhs > (rhs + 10 ** (-6)):
                break

            # add constraint
            ips_model.addCons(
                lhs <= (quicksum(ips_vars[vlist[i].name] * clist[i] for i in range(len(vlist))) + const <= rhs))

        logging.info(">>>> Optimize over EIPS")
        ips_model.optimize()
        # if ips_model.getStatus() == 'optimal':

        sol = self.model.createSol()
        for v in variables:
            val_ips = ips_model.getVal(ips_vars[v.name])
            if v.vtype() != 'CONTINUOUS':
                val_ips = int(round(val_ips))
            self.model.setSolVal(sol, v, val_ips)

        feasible_rounding = self.model.checkSol(sol)
        accepted_solution = self.model.trySol(sol)

        logging.debug(">>>> feasible solution? %s" % ("yes" if feasible_rounding == 1 else "no"))
        logging.info(">>>> accepted solution? %s" % ("yes" if accepted_solution == 1 else "no"))

        vio = []
        for row in linear_rows:
            vio.append(self.model.getRowSolFeas(row, sol))
        logging.debug('Maximum violation of LP row: ' + str(min(vio)))

        if accepted_solution:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}


def test_granularity_rootnode():
    path_to_problems = '/home/stefan/Dokumente/02_HiWi_IOR/Paper_BA/franumeric/selectedTestbed/'
    os.chdir(path_to_problems)
    problem_names = glob.glob("*.mps")

    for i in [34, 43, 46, 18, 50, 54, 29, 31]:
        problem = problem_names[i-1]

        # create model
        m = Model()

        # create and add heuristic to SCIP
        heuristic = feasiblerounding()
        m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                      freq=0)

        m.setParam("limits/time", 45)

        # read exemplary problem from file
        print('>>>>> Working on Problem: ', problem, ', which is number ', i+1, 'of ', len(problem_names))
        m.readProblem("".join([path_to_problems, problem]))

        # optimize problem
        m.optimize()

        # free model explicitly
        del m


def test_heur():
    m = Model()
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                  freq=0)
    m.readProblem('/home/stefan/Dokumente/02_HiWi_IOR/Paper_BA/franumeric/selectedTestbed/mik.250-1-100.1.mps')
    m.optimize()
    del m


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_heur()
    test_granularity_rootnode()
