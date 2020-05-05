import math, random, numpy as np, logging, pickle, scipy as sp
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_LPSOLSTAT, quicksum
from time import time
from scipy.sparse import dok_matrix, linalg


FEAS_TOL = 1E-6

#
# heuristic: feasible rounding approach
#

class feasiblerounding(Heur):

    def __init__(self, options=None):
        """
        Construct the feasible rounding heuristic.
        Encoded in options (i.e. options['mode'] etc.):
        :param mode: 'original' or 'deep_fixing'
        :param delta: enlargement (0,1)
        :param line_search: flag for postprocessing
        :return: returns nothing
        """

        if options is None:
            options = {}
        self.options = {'mode': 'original', 'delta': 0.999, 'line_search': False, 'diving': True}
        for key in options:
            self.options[key] = options[key]

        # Set class variables with names as keys and their corresponding values from the dictionary
        for key in self.options:
            setattr(self, key, self.options[key])

        self.statistics = []
        self.timer_start = None

    def heurexec(self, heurtiming, nodeinfeasible):
        """
        executes the heuristic.

        :param heurtiming
        :param nodeinfeasible
        :return: returns SCIP_RESULT to solver
        """

        self.start_run_statistics()
        sol_dict = {}
        val_dict = {}

        if self.model.getLPSolstat() != SCIP_LPSOLSTAT.OPTIMAL:
            logging.info('>>>> Sub-problem is pruned, skip heuristic.')
            self.run_statistics['pruned_prob'] = True
            self.save_run_statistics()
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if self.contains_equality_constrs():
            self.run_statistics['eq_constrs'] = True
            logging.info('>>>> Problem contains equality constraints on int vars, skip heuristic.')
            self.save_run_statistics()
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if not (self.mode in ['original', 'deep_fixing']):
            logging.warning('Mode must be original or deep fixing, but is ' + self.mode)
        elif self.mode == 'deep_fixing' and self.diving == True:
            logging.warning('Deep fixing and diving in combination is not recommended')

        logging.info(">>>> Build inner parallel set")
        ips_model, ips_vars = self.build_ips()
        if self.ips_proven_empty:
            logging.info('>>>> Lefthand side larger than right hand side for some constraint, skip heuristic.')
            self.save_run_statistics()
            return {"result": SCIP_RESULT.DIDNOTRUN}

        logging.info(">>>> Optimize over EIPS")
        timer_ips = time()
        ips_model.hideOutput(True)
        ips_model.optimize()
        self.run_statistics['time_solveips'] = time() - timer_ips
        logging.info("Model status is:" + str(ips_model.getStatus()))

        if ips_model.getStatus() == 'optimal':

            self.run_statistics['ips_nonempty'] = True
            sol_root = self.get_sol_submodel(ips_vars, ips_model)

            if self.diving:
                diving_mode = 'impact'
                obj_diving, sol_diving = self.diving_procedure(diving_mode, sol_root)
                sol_dict['diving'] = sol_diving
                val_dict['diving'] = obj_diving


            if self.line_search:
                timer_pp = time()
                rel_sol_dict = self.get_sol_relaxation()
                line_search_sol = self.get_line_search_rounding(rel_sol_dict, sol_root)
                label_sol = self.mode + '_ls'
                sol_dict[label_sol] = self.fix_and_optimize(line_search_sol)
                val_dict[label_sol] = self.get_obj_value(sol_dict[label_sol])
                self.run_statistics['time_pp'] = time() - timer_pp

            label_sol = self.mode
            self.round_sol(sol_root)
            sol_dict[label_sol] = self.fix_and_optimize(sol_root)
            val_dict[label_sol] = self.get_obj_value(sol_dict[self.mode])

        del ips_model
        del ips_vars

        logging.info(val_dict)
        sol_best = self.get_best_sol(sol_dict, val_dict)

        if sol_best:
            sol_model = self.model.getBestSol()
            solution_accepted = self.sol_is_accepted(sol_best)
            self.run_statistics['obj_best'] = val_dict[min(val_dict, key=val_dict.get)]
            self.run_statistics['obj_root'] = val_dict[self.mode]
            if self.line_search:
                self.run_statistics['obj_ls'] = val_dict[self.mode + '_ls']
            self.run_statistics['obj_SCIP'] = self.model.getSolObjVal(sol_model, original=False)

            if solution_accepted:
                self.run_statistics['accepted'] = True
                self.save_run_statistics()
                return {"result": SCIP_RESULT.FOUNDSOL}
            else:
                self.save_run_statistics()
                return {"result": SCIP_RESULT.DIDNOTFIND}
        else:
            self.save_run_statistics()
            return {"result": SCIP_RESULT.DIDNOTFIND}

    def start_run_statistics(self):
        self.run_statistics = {}
        self.run_statistics = {'name': self.model.getProbName(), 'depth': self.model.getDepth(), 'eq_constrs': False,
                               'pruned_prob': False, 'ips_nonempty': False, 'feasible': False,
                               'accepted': False, 'obj_best': None, 'obj_ls': None, 'obj_root':None, 'obj_SCIP': None,
                               'time_heur': None, 'time_solveips': None, 'time_pp': None, 'time_scip': None,
                               'diving_depth': None, 'diving_best_depth': None, 'obj_diving': None}
        self.ips_proven_empty = False
        self.timer_start = time()

    def save_run_statistics(self):
        if self.timer_start:
            self.run_statistics['time_heur'] = time() - self.timer_start
        self.run_statistics['time_scip'] = self.model.getTotalTime()
        self.statistics.append(self.run_statistics)

    def diving_procedure(self, diving_mode, sol_root):
        sol_diving = sol_diving_new = sol_root
        obj_diving = math.inf
        if diving_mode == 'impact':
            self.computeB()
        candidate = self.get_diving_candidates(sol_diving, diving_mode)
        self.model.startProbing()
        logging.info('>>>> Start Diving')
        dive_itr = 1
        while candidate:
            to_fix = candidate.pop(0)
            self.model.fixVarProbing(to_fix, round(sol_diving_new[to_fix.name]))
            cutoff, numberofreductions = self.model.propagateProbing(-1)
            if cutoff:
                logging.warning("Diving-LP not feasible anymore")
            if (dive_itr%5 == 0) or (not candidate) or (numberofreductions>0):
                ips_model_p, ips_vars_p = self.build_ips()
                ips_model_p.hideOutput(True)
                ips_model_p.optimize()

                sol_diving_new = self.get_sol_submodel(ips_vars_p, ips_model_p)
                self.round_sol(sol_diving_new)
                sol_diving_new = self.fix_and_optimize(sol_diving_new)
                obj_diving_new = self.get_obj_value(sol_diving_new)
                if (not sol_diving) or obj_diving_new < obj_diving:
                    logging.info('>>>> Diving yielded better objective: ' + str(obj_diving_new) + ' in Depth ' + str(dive_itr-1))
                    self.run_statistics['diving_best_depth'] = dive_itr
                    sol_diving = sol_diving_new
                    obj_diving = obj_diving_new
                candidate = self.get_diving_candidates(sol_diving_new, diving_mode)
                ips_model_p.freeProb()
                del ips_model_p
                del ips_vars_p
            dive_itr = dive_itr + 1
        self.model.endProbing()
        logging.info('>>>> End Diving')
        self.run_statistics['obj_diving'] = obj_diving
        self.run_statistics['diving_depth'] = dive_itr - 1
        return obj_diving, sol_diving

    def get_diving_candidates(self, ips_sol, diving_mode):
        if diving_mode=='impact':
            candidates = self.get_impact_candidates(ips_sol)
        elif diving_mode=='simple':
            candidates = self.get_random_candidates(ips_sol)
        else:
            candidates = None

        return candidates

    def get_random_candidates(self, ips_sol):
        original_vars = self.model.getVars(transformed=True)
        variables = {v.name: v for v in original_vars if self.int_and_not_fixed(v)}
        cand = random.shuffle(list(variables.values()))

        return cand


    def get_impact_candidates(self, ips_sol):
        original_vars = self.model.getVars(transformed=True)
        variables = {v.name: v for v in original_vars if self.int_and_not_fixed(v)}

        n = len(self.B_index_dict)
        E = dok_matrix((n,n))
        for v in variables:
            y = ips_sol[v]
            y_check = round(ips_sol[v])
            ind = self.B_index_dict[v]
            E[ind,ind] = (y-y_check)

        E = E.tocsc()
        m1 = np.sum((self.B).dot(E), axis=0)
        m2 = np.sum(self.Babs, axis=0)
        measure = m1+m2
        if np.amax(measure) > 0:
            with_index = [(measure[i],i) for i in range(len(measure))]
            with_index.sort(key=lambda tup: tup[0])
            candidates = [variables[list(self.B_index_dict)[i]] for m,i in with_index if m>0]
        else:
            candidates = []
        print('Completed "get impact candidates"')
        return candidates

    def computeB(self):
        original_vars = self.model.getVars(transformed=True)
        linear_rows = self.model.getLPRowsData()
        B_index_dict = {}
        index = 0
        for v in original_vars:
            if v.vtype() != 'CONTINUOUS':
                B_index_dict[v.name] = index
                index += 1
        B = dok_matrix((len(linear_rows),index))
        Babs = dok_matrix((len(linear_rows),index))
        for i,row in enumerate(linear_rows):
            cols = row.getCols()
            vals = row.getVals()
            for j,col in enumerate(cols):
                v = col.getVar()
                if v.vtype() != 'CONTINUOUS':
                    B[i,B_index_dict[v.name]] = vals[j]
                    Babs[i,B_index_dict[v.name]] = abs(vals[j])

        B = B.tocsc()
        Babs = Babs.tocsc()
        norm_b = linalg.norm(B, axis=1)
        norm_b[norm_b == 0] = 1
        self.B =  (B.transpose()).dot(np.diag(1 / norm_b))
        self.Babs = (Babs.transpose()).dot(np.diag(1 / norm_b))
        self.B_index_dict = B_index_dict
        print('completed computation of B')

    def int_and_not_fixed(self, v):
        return (v.vtype() != 'CONTINUOUS') and (v.getLbLocal() != v.getUbLocal())

    def create_sol(self, sol_dict):
        original_vars = self.model.getVars(transformed=True)
        sol = self.model.createSol()
        for v in original_vars:
            self.model.setSolVal(sol, v, sol_dict[v.name])
        return sol

    def get_line_search_rounding(self, rel_sol_dict, ips_sol_dict):

        original_vars = self.model.getVars(transformed=True)
        rel_int_sol = self.get_value_list_of_int_vars(rel_sol_dict)
        ips_int_sol = self.get_value_list_of_int_vars(ips_sol_dict)
        switching_points = get_switching_points(rel_int_sol, ips_int_sol)
        logging.info(switching_points)
        feasible = False
        i = 0
        while (not feasible) and (i < len(switching_points)):
            t = switching_points[i]
            sol_dict = {}
            for v in original_vars:
                sol_dict[v.name] = rel_sol_dict[v.name] + t * (ips_sol_dict[v.name] - rel_sol_dict[v.name])
            self.round_sol(sol_dict)
            feasible = self.sol_satisfies_constrs(sol_dict)
            i = i + 1

        if not feasible:
            logging.warning("No feasible point found while iterating through switching points")
            logging.warning("len switching points: " + str(len(switching_points)) + "current iteration" + str(i))
            sol = self.create_sol(sol_dict)
            logging.warning("constraint violation is " + str(self.get_lp_violation(sol)))

        else:
            logging.info('Found feasible point for t = ' + str(t))

        return sol_dict

    def contains_equality_constrs(self):
        linear_rows = self.model.getLPRowsData()
        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            vtype_list = [var.vtype() for var in vlist]
            if any([var != 'CONTINUOUS' for var in vtype_list]):
                if lrow.getLhs() == lrow.getRhs():
                    return True
        return False

    def get_value_list_of_int_vars(self, sol_dict):
        int_values = []
        original_vars = self.model.getVars(transformed=True)
        for var in original_vars:
            if var.vtype() != 'CONTINUOUS':
                int_values.append(sol_dict[var.name])
        return int_values

    def add_vars_and_bounds(self, model):
        original_vars = self.model.getVars(transformed=True)
        var_dict = dict()
        for var in original_vars:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS':
                if self.mode == 'deep_fixing' and round(var.getLPSol()) == var.getLPSol() and var.vtype() == 'BINARY':
                    lower = upper = var.getLPSol()
                else:
                    if lower != upper:
                        if lower is not None:
                            lower = math.ceil(lower) - self.delta + 0.5
                        if upper is not None:
                            upper = math.floor(upper) + self.delta - 0.5
            var_dict[var.name] = model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper, obj=var.getObj())
        self.set_objective_sense(model)
        return var_dict

    def get_obj_value(self, sol):
        variables = self.model.getVars(transformed=True)
        obj_val = sum([v.getObj() * sol[v.name] for v in variables])
        return obj_val

    def enlargement_possible(self, vlist, clist):
        return all(vlist[i].vtype() != 'CONTINUOUS' for i in range(len(clist))) and all(
            clist[i].is_integer() for i in range(len(clist)))

    def is_integer(self, var):
        return var.vtype() == 'BINARY' and round(var.getLPSol()) == var.getLPSol()

    def compute_fixing_enlargement(self, vlist, clist):
        fixing_enlargement = 0
        for i, var in enumerate(vlist):
            if self.is_integer(var):
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
            model.addCons(lhs <=
                          (quicksum(var_dict[vlist[i].name] * clist[i] for i in range(len(vlist))) + const
                           <= rhs))

    def add_ips_constraints(self, ips_model, var_dict):
        linear_rows = self.model.getLPRowsData()
        for lrow in linear_rows:
            vlist = [col.getVar() for col in lrow.getCols()]
            clist = lrow.getVals()
            const = lrow.getConstant()

            # only values from integer variables, which are not fixed
            beta = sum(abs(clist[i]) for i in range(len(clist))
                       if ((vlist[i].vtype() != 'CONTINUOUS') and (vlist[i].getLbLocal()) != vlist[i].getUbLocal()))
            if self.mode == 'deep_fixing':
                fixing_enlargement = self.compute_fixing_enlargement(vlist, clist)
                beta = beta - fixing_enlargement
            if self.enlargement_possible(vlist, clist):
                lhs = math.ceil(lrow.getLhs()) + 0.5 * beta - self.delta
                rhs = math.floor(lrow.getRhs()) - 0.5 * beta + self.delta
            else:
                lhs = lrow.getLhs() + 0.5 * beta
                rhs = lrow.getRhs() - 0.5 * beta
            if lhs > (rhs + FEAS_TOL):
                logging.warning("Inner parallel set is empty")
                self.ips_proven_empty = True
                break
            ips_model.addCons(
                lhs <= (quicksum(var_dict[vlist[i].name] * clist[i] for i in range(len(vlist))) + const <= rhs))

    def get_lp_violation(self, sol):
        local_linear_rows = self.model.getLPRowsData()
        vio = []
        if local_linear_rows:
            for row in local_linear_rows:
                vio.append(self.model.getRowSolFeas(row, sol))
            return min(vio)
        return 0

    def heurinitsol(self):
        logging.info(">>>> Call heurinitsol()")

    def heurexitsol(self):
        logging.info(">>>> Call heurexitsol()")
        with open('temp_results.pickle', 'ab') as handle:
            pickle.dump(self.statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def add_reduced_vars(self, reduced_model, sol_fra):
        reduced_var_dict = {}
        original_vars = self.model.getVars(transformed=True)
        for var in original_vars:
            lower = var.getLbLocal()
            upper = var.getUbLocal()
            if var.vtype() != 'CONTINUOUS':
                lower = upper = sol_fra[var.name]
            reduced_var_dict[var.name] = reduced_model.addVar(name=var.name, vtype='CONTINUOUS', lb=lower, ub=upper,
                                                              obj=var.getObj())
        self.set_objective_sense(reduced_model)
        return reduced_var_dict

    def set_objective_sense(self, submodel):
        if self.model.getObjectiveSense() == 'minimize':
            submodel.setMinimize()
        elif self.model.getObjectiveSense() == 'maximize':
            submodel.setMaximize()
        else:
            logging.warning('Objective sense is not \'minimize\' or \'maximize\'')

    def round_sol(self, sol):
        original_vars = self.model.getVars(transformed=True)
        for v in original_vars:
            if v.vtype() != 'CONTINUOUS':
                sol[v.name] = int(round(sol[v.name]))

    def get_sol_submodel(self, model_vars, model):
        sol = {}
        original_vars = self.model.getVars(transformed=True)
        for v in original_vars:
            var_value = model.getVal(model_vars[v.name])
            sol[v.name] = var_value
        return sol

    def get_sol_relaxation(self):
        sol = {}
        original_vars = self.model.getVars(transformed=True)
        for v in original_vars:
            var_value = self.model.getVal(v)
            sol[v.name] = var_value
        return sol

    def sol_satisfies_constrs(self, sol_dict):
        sol = self.create_sol(sol_dict)
        rounding_feasible = (self.get_lp_violation(sol) >= -FEAS_TOL)
        return rounding_feasible

    def sol_is_accepted(self, sol_dict):
        original_vars = self.model.getVars(transformed=True)
        sol = self.model.createSol()
        for v in original_vars:
            self.model.setSolVal(sol, v, sol_dict[v.name])

        rounding_feasible = self.model.checkSol(sol)
        solution_accepted = self.model.trySol(sol)

        if rounding_feasible == 0:
            logging.warning(">>>> ips feasible, but no feasible rounding")
            logging.debug('Maximum violation of LP row: ' + str(self.get_lp_violation(sol)))
        else:
            self.run_statistics['feasible'] = True
        logging.info(">>>> accepted solution? %s" % ("yes" if solution_accepted == 1 else "no"))

        return solution_accepted

    def fix_and_optimize(self, sol_fra):
        reduced_model = Model('reduced_model')
        reduced_model_vars = self.add_reduced_vars(reduced_model, sol_fra)
        self.add_model_constraints(reduced_model, reduced_model_vars)
        reduced_model.hideOutput(True)
        reduced_model.optimize()
        sol_fra = self.get_sol_submodel(reduced_model_vars, reduced_model)
        reduced_model.freeProb()
        del reduced_model
        return sol_fra

    def build_ips(self):
        ips_model = Model("ips")
        ips_vars = self.add_vars_and_bounds(ips_model)
        self.add_ips_constraints(ips_model, ips_vars)
        return ips_model, ips_vars

    def get_best_sol(self, sol_dict, val_dict):
        best_sol = {}
        if not sol_dict:
            pass
        elif self.model.getObjectiveSense() == 'minimize':
            best_sol = sol_dict[min(val_dict, key=val_dict.get)]
        elif self.model.getObjectiveSense() == 'maximize':
            best_sol = sol_dict[max(val_dict, key=val_dict.get)]
        else:
            logging.warning('Unknown objective sense. Expected \'minimize\'or \' maximize \' ')
        return best_sol


# static methods

def get_switching_points(int_sol1, int_sol2):
    switching_points = [0, 1]
    for j in range(len(int_sol1)):
        eta_j = int_sol2[j] - int_sol1[j]
        if eta_j != 0:
            lower = math.ceil(int_sol1[j] + min(0, eta_j) - 1 / 2)
            upper = math.floor(int_sol1[j] + max(0, eta_j) - 1 / 2)
            for l in range(lower, upper + 1):
                switching_points.append((1 / 2 - int_sol1[j] + l) / eta_j)
    logging.info("Computed %i switching points" % len(switching_points))
    return sorted(set(switching_points))