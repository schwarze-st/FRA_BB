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

    def add_vars_and_bounds(self, model):

        original_vars = self.model.getVars(transformed=True)
        var_dict = dict()
        for var in original_vars:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS':
                if self.options['fix_integers'] and round(var.getLPSol()) == var.getLPSol() and var.vtype() == 'BINARY':
                    lower = upper = var.getLPSol()
                else:
                    if lower is not None:
                        lower = math.ceil(lower) - self.options['delta'] + 0.5
                    if upper is not None:
                        upper = math.floor(upper) + self.options['delta'] - 0.5
            var_dict[var.name] = model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper, obj=var.getObj())

        return var_dict

    def add_objective(self, model, var_dict):

        obj_sub = Expr()
        variables = self.model.getVars(transformed=True)
        zf_sense = self.model.getObjectiveSense()
        for v in variables:
            coefficient = v.getObj()
            if coefficient != 0:
                obj_sub += coefficient * var_dict[v.name]
        if obj_sub.degree() != 1:
            logging.warning("Objective function is empty")
        obj_sub.normalize()
        model.setObjective(obj_sub, sense=zf_sense)

    def enlargement_possible(self, vlist, clist):
        return all(vlist[i].vtype() != 'CONTINUOUS' for i in range(len(clist))) and all(
            clist[i].is_integer() for i in range(len(clist)))

    def is_fixed(self, var):
        return (var.vtype()=='BINARY' and round(var.getLPSol()) == var.getLPSol())

    def compute_fixing_enlargement(self, vlist, clist):
        fixing_enlargement = 0
        for i, var in enumerate(vlist):
            if self.is_fixed(var):
                fixing_enlargement += abs(clist[i])
        return fixing_enlargement

    def add_ips_constraints(self, ips_model, var_dict):
        linear_rows = self.model.getLPRowsData()

        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()

            beta = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype() != 'CONTINUOUS')
            if self.options['fix_integers']:
                fixing_enlargement = self.compute_fixing_enlargement(vlist, clist)
                beta = beta - fixing_enlargement
            if self.enlargement_possible(vlist, clist):
                lhs = math.ceil(lrow.getLhs()) + 0.5 * beta - self.options['delta']
                rhs = math.floor(lrow.getRhs()) - 0.5 * beta + self.options['delta']
            else:
                lhs = lrow.getLhs() + 0.5 * beta
                rhs = lrow.getRhs() - 0.5 * beta
            if lhs > (rhs + 10 ** (-6)):
                logging.warning("Inner parallel set is empty (equality constrained)")
                break

            ips_model.addCons(
                lhs <= (quicksum(var_dict[vlist[i].name] * clist[i] for i in range(len(vlist))) + const <= rhs))

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

    def add_reduced_vars(self, reduced_model, ips_model, ips_vars):
        reduced_var_dict = {}
        original_vars = self.model.getVars(transformed=True)
        for var in original_vars:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS':
                local_rounding = int(round(ips_model.getVal(ips_vars[var.name])))
                lower = upper = local_rounding
            reduced_var_dict[var.name] = reduced_model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper,
                                                           obj=var.getObj())
        return  reduced_var_dict


    # execution method of the heuristic
    def heurexec(self, heurtiming, nodeinfeasible):
        delta = self.options['delta']
        fix_integers = self.options['fix_integers']

        logging.info(">>>> Call feasible rounding heuristic at a node with depth %d" % (self.model.getDepth()))
        logging.info(">>>> Build inner parallel set")

        variables = self.model.getVars(transformed=True)
        ips_model = Model("ips")
        ips_vars = self.add_vars_and_bounds(ips_model)
        self.add_objective(ips_model,ips_vars)
        self.add_ips_constraints(ips_model, ips_vars)

        logging.info(">>>> Optimize over EIPS")
        ips_model.optimize()

        linear_rows = self.model.getLPRowsData()

        # Post-Processing -> fix rounded integer values and optimize
        reduced_model = Model('reduced_model')
        reduced_model_vars = self.add_reduced_vars(reduced_model, ips_model, ips_vars)

        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()
            lhs = lrow.getLhs()
            rhs = lrow.getRhs()
            # add constraint
            reduced_model.addCons(
                lhs <= (quicksum(reduced_model_vars[vlist[i].name] * clist[i] for i in range(len(vlist))) + const <= rhs))

        reduced_model.optimize()

        # Add Solution
        sol = self.model.createSol()
        for v in variables:
            val_post = reduced_model.getVal(reduced_model_vars[v.name])
            self.model.setSolVal(sol, v, val_post)

        feasible_rounding = self.model.checkSol(sol)
        accepted_solution = self.model.trySol(sol)

        logging.debug(">>>> feasible solution? %s" % ("yes" if feasible_rounding == 1 else "no"))
        logging.info(">>>> accepted solution? %s" % ("yes" if accepted_solution == 1 else "no"))

        if ips_model.getStatus() == 'optimal' and feasible_rounding == 0:
            logging.warning("ips feasible, but no feasible rounding")

        logging.debug('Maximum violation of LP row: ' + str(self.get_lp_violation(sol)))

        del ips_model
        del reduced_model

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
    options = {'post_processing' : True}
    heuristic = feasiblerounding(options)
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
