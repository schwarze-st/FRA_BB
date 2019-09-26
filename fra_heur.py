import glob
import math
import os
import logging

from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum, Expr

#
# heuristic: feasible rounding approach
#


class feasiblerounding(Heur):

    def __init__(self, options = {}):

        self.options = {'enlargement': True, 'post_processing': True,
                        'fix_integers': True, 'delta' : 0.999}
        for key in options:
            self.options[key] = options[key]

    def add_vars_and_bounds(self, local_model):
        original_vars = self.model.getVars(transformed=True)
        var_dict = dict()
        for var in original_vars:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS' and self.options['enlargement']:
                if self.options['fix_integers'] and round(var.getLPSol()) == var.getLPSol() and var.vtype() == 'BINARY':
                    lower = upper = var.getLPSol()
                else:
                    if lower is not None:
                        lower = math.ceil(lower) - self.options['delta'] + 0.5
                    if upper is not None:
                        upper = math.floor(upper) + self.options['delta'] - 0.5
            var_dict[var.name] = local_model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper, obj=var.getObj())

        return var_dict

    def get_lp_violation(self, sol):
        local_linear_rows = self.model.getLPRowsData()
        vio = []
        for row in local_linear_rows:
            vio.append(self.model.getRowSolFeas(row, sol))
        return min(vio)

    def heurinitsol(self):
        print(">>>> call heurinitsol()")

    def heurexitsol(self):
        print(">>>> call heurexitsol()")

    # execution method of the heuristic
    def heurexec(self, heurtiming, nodeinfeasible):
        delta = self.options['delta']
        fix_integers = self.options['fix_integers']
        enlarged = self.options['enlargement']

        logging.info(">>>> Call feasible rounding heuristic at a node with depth %d" % (self.model.getDepth()))
        logging.info(">>>> Build inner parallel set")

        variables = self.model.getVars(transformed=True)
        ips_model = Model("ips")
        ips_vars = self.add_vars_and_bounds(ips_model)

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

        # modifizierte Ungleichungen hinzufügen
        linear_rows = self.model.getLPRowsData()

        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()
            lhs = lrow.getLhs()
            rhs = lrow.getRhs()

            beta = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype() != 'CONTINUOUS')
            if fix_integers:
                minus = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype() == 'BINARY' and round(vlist[i].getLPSol()) == vlist[i].getLPSol())
                beta = beta - minus
            if enlarged and all(vlist[i].vtype() != 'CONTINUOUS' for i in range(len(clist))) and all(
                    clist[i].is_integer() for i in range(len(clist))):
                lhs = math.ceil(lrow.getLhs()) + 0.5 * beta - delta
                rhs = math.floor(lrow.getRhs()) - 0.5 * beta + delta
            else:
                lhs = lrow.getLhs() + 0.5 * beta
                rhs = lrow.getRhs() - 0.5 * beta
            if lhs > (rhs + 10 ** (-6)):
                logging.warning("Inner parallel set is empty (equality constrained)")
                break

            ips_model.addCons(
                lhs <= (quicksum(ips_vars[vlist[i].name] * clist[i] for i in range(len(vlist))) + const <= rhs))

        logging.info(">>>> Optimize over EIPS")
        ips_model.optimize()

        # Post-Processing -> fix rounded integer values and optimize
        post_model = Model()
        post_var_dict = dict()
        for var in variables:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS':
                local_rounding = int(round(ips_model.getVal(ips_vars[var.name])))
                lower = upper = local_rounding
            post_var_dict[var.name] = post_model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper, obj=var.getObj())
        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()
            lhs = lrow.getLhs()
            rhs = lrow.getRhs()
            # add constraint
            post_model.addCons(
                lhs <= (quicksum(post_var_dict[vlist[i].name] * clist[i] for i in range(len(vlist))) + const <= rhs))

        post_model.optimize()

        # Add Solution
        sol = self.model.createSol()
        for v in variables:
            val_post = post_model.getVal(post_var_dict[v.name])
            self.model.setSolVal(sol, v, val_post)

        feasible_rounding = self.model.checkSol(sol)
        accepted_solution = self.model.trySol(sol)

        logging.debug(">>>> feasible solution? %s" % ("yes" if feasible_rounding == 1 else "no"))
        logging.info(">>>> accepted solution? %s" % ("yes" if accepted_solution == 1 else "no"))

        if ips_model.getStatus() == 'optimal' and feasible_rounding == 0:
            logging.warning("ips feasible, but no feasible rounding")

        logging.debug('Maximum violation of LP row: ' + str(self.get_lp_violation(sol)))

        del ips_model
        del post_model

        if accepted_solution:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}


def test_granularity_rootnode():
    path_to_problems = '/home/stefan/Dokumente/02_HiWi_IOR/Paper_BA/franumeric/selectedTestbed/'
    os.chdir(path_to_problems)
    problem_names = glob.glob("*.mps")
    i = 0

    for problem in problem_names:
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
        i = i+1
        if i > 10:
            break


def test_heur():
    m = Model()
    options = {'enlarged': True, 'post_processing' : True}
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                  freq=5)
    # m.readProblem('/home/stefan/Dokumente/02_HiWi_IOR/Paper_BA/franumeric/selectedTestbed/mik.250-1-100.1.mps') # implicit integer variable
    m.readProblem('50v-10.mps') # ERROR SIGSEGV
    m.optimize()
    del m


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_heur()
    # test_granularity_rootnode()
