import glob
import math
import os
import logging

from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum, Expr

#
# heuristic: feasible rounding approach
#



class feasiblerounding(Heur):

    def __init__(self, options={}):

        self.options = {'mode': ['original', 'deep_fixing'], 'delta': 0.999}
        for key in options:
            self.options[key] = options[key]


    def add_vars_and_bounds(self, model, mode):

        original_vars = self.model.getVars(transformed=True)
        var_dict = dict()
        for var in original_vars:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS':
                if mode == 'deep_fixing' and round(var.getLPSol()) == var.getLPSol() and var.vtype() == 'BINARY':
                    lower = upper = var.getLPSol()
                else:
                    if lower is not None:
                        lower = math.ceil(lower) - self.options['delta'] + 0.5
                    if upper is not None:
                        upper = math.floor(upper) + self.options['delta'] - 0.5
            var_dict[var.name] = model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper, obj=var.getObj())

        return var_dict

    def get_obj_value(self, sol_dict):
        variables = self.model.getVars(transformed=True)
        obj_val = sum([v.getObj()*sol_dict[v.name] for v in variables])
        return obj_val

    def add_objective(self, model, var_dict):
        obj_sub = Expr()
        variables = self.model.getVars(transformed=True)
        zf_sense = self.model.getObjectiveSense()
        for v in variables:
            obj_sub += v.getObj() * var_dict[v.name]
        if obj_sub.degree() != 1:
            logging.warning("Objective function is empty")
        obj_sub.normalize()
        model.setObjective(obj_sub, sense=zf_sense)

    def enlargement_possible(self, vlist, clist):
        return all(vlist[i].vtype() != 'CONTINUOUS' for i in range(len(clist))) and all(
            clist[i].is_integer() for i in range(len(clist)))

    def is_fixed(self, var):
        return var.vtype() == 'BINARY' and round(var.getLPSol()) == var.getLPSol()

    def compute_fixing_enlargement(self, vlist, clist):
        fixing_enlargement = 0
        for i, var in enumerate(vlist):
            if self.is_fixed(var):
                fixing_enlargement += abs(clist[i])
        return fixing_enlargement

    def add_model_constraints(self, model, var_dict):

        linear_rows = self.model.getLPRowsData()

        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()
            lhs = lrow.getLhs()
            rhs = lrow.getRhs()
            # add constraint
            model.addCons( lhs <=
                           (quicksum( var_dict[vlist[i].name] * clist[i] for i in range(len(vlist))) + const
                            <= rhs))

    def add_ips_constraints(self, ips_model, var_dict, mode):
        linear_rows = self.model.getLPRowsData()

        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()

            beta = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype() != 'CONTINUOUS')
            if mode == 'deep_fixing':
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
        if vio:
            for row in local_linear_rows:
                vio.append(self.model.getRowSolFeas(row, sol))
            return min(vio)
        else:
            return 0

    def heurinitsol(self):
        print(">>>> call heurinitsol()")

    def heurexitsol(self):
        print(">>>> call heurexitsol()")

    def add_reduced_vars(self, reduced_model, sol_FRA):
        reduced_var_dict = {}
        original_vars = self.model.getVars(transformed=True)
        for var in original_vars:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS':
                lower = upper = sol_FRA[var.name]
            reduced_var_dict[var.name] = reduced_model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper,
                                                           obj=var.getObj())
        return  reduced_var_dict

    def round_sol(self, sol):
        original_vars = self.model.getVars(transformed=True)
        for v in original_vars:
            if v.vtype() != 'CONTINUOUS':
                sol[v.name] = int(round(sol[v.name]))
    
    def get_sol(self, vars, model):
        sol = {}
        original_vars = self.model.getVars(transformed=True)
        for v in original_vars:
            var_value = model.getVal(vars[v.name])
            sol[v.name] = var_value
        return sol

    def try_sol(self, sol_dict):
        original_vars = self.model.getVars(transformed=True)
        sol = self.model.createSol()
        for v in original_vars:
            self.model.setSolVal(sol, v, sol_dict[v.name])

        rounding_feasible = self.model.checkSol(sol)
        solution_accepted = self.model.trySol(sol)

        if rounding_feasible == 0:
            logging.warning(">>>> ips feasible, but no feasible rounding")
        logging.info(">>>> accepted solution? %s" % ("yes" if solution_accepted == 1 else "no"))

        logging.debug('Maximum violation of LP row: ' + str(self.get_lp_violation(sol)))

        return solution_accepted

    def fix_and_optimize(self, sol_FRA):
        reduced_model = Model('reduced_model')
        reduced_model_vars = self.add_reduced_vars(reduced_model, sol_FRA)
        self.add_model_constraints(reduced_model, reduced_model_vars)
        self.add_objective(reduced_model, reduced_model_vars)
        reduced_model.optimize()
        sol_FRA = self.get_sol(reduced_model_vars, reduced_model)
        del reduced_model
        return sol_FRA

    def build_ips(self, mode):
        ips_model = Model("ips")
        ips_vars = self.add_vars_and_bounds(ips_model, mode)
        self.add_ips_constraints(ips_model, ips_vars, mode)
        self.add_objective(ips_model, ips_vars)
        return ips_model, ips_vars

    def get_best_sol(self, sol_dict, val_dict):
        if not sol_dict:
            return {}
        if self.model.getObjectiveSense() == 'minimize':
            best_sol = sol_dict[min(val_dict, key=val_dict.get)]
        elif self.model.getObjectiveSense() == 'maximize':
            best_sol = sol_dict[max(val_dict, key=val_dict.get)]
        else:
            logging.warning('Unknown objective sense. Expected \'minimize\'or \' maximize \' ')
        return best_sol

    # execution method of the heuristic
    def heurexec(self, heurtiming, nodeinfeasible):

        logging.info(">>>> Call feasible rounding heuristic at a node with depth %d" % (self.model.getDepth()))
        logging.info(">>>> Build inner parallel set")
        sol_dict = {}
        val_dict = {}

        for mode in self.options['mode']:
            if not (mode in ['original', 'deep_fixing']):
                logging.warning('Mode must be original or deep fixing, but is ' + mode)
            ips_model, ips_vars = self.build_ips(mode)
            logging.info(">>>> Optimize over EIPS")
            ips_model.optimize()

            if ips_model.getStatus() == 'optimal':
                sol_FRA_current_mode = self.get_sol(ips_vars, ips_model)
                self.round_sol(sol_FRA_current_mode)
                sol_dict[mode] = self.fix_and_optimize(sol_FRA_current_mode)
                val_dict[mode] = self.get_obj_value(sol_dict[mode])
            del ips_model
            del ips_vars
        logging.info(val_dict)
        sol_FRA = self.get_best_sol(sol_dict, val_dict)

        if sol_FRA:
            solution_accepted = self.try_sol(sol_FRA)
            if solution_accepted:
                return {"result": SCIP_RESULT.FOUNDSOL}
            else:
                return {"result": SCIP_RESULT.DIDNOTFIND}
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
    options = {'mode':['original','deep_fixing']}
    heuristic = feasiblerounding(options)
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                  freq=5)
    # m.readProblem('/home/stefan/Dokumente/02_HiWi_IOR/Paper_BA/franumeric/selectedTestbed/mik.250-1-100.1.mps') # implicit integer variable
    m.readProblem('50v-10.mps')
    m.optimize()
    del m


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_heur()
    # test_granularity_rootnode()
