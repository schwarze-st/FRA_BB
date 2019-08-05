
import os, glob
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum, Expr
#from pyscipopt  import Constant, SumExpr, Expr, VarExpr, ProdExpr
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
        delta = 0.999
        gran = True

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
        zf_ex = self.model.getObjective()
        zf_sense = self.model.getObjectiveSense()
        zf_ex.normalize()
        zf_items = zf_ex.terms.items()
        new_expr = quicksum(ips_vars[v[0].name]*coef for v,coef in zf_items if len(v)!=0)
        
        #ips_model.setObjective(quicksum(ips_vars[v.name]*coef for v , coef in zf_ex.terms.items()),sense=zf_sense)
        
        '''
        if Expr._is_number(zf_ex):
            zf = Constant(zf_ex)
        elif isinstance(zf_ex, Expr):
            # loop over terms and create a sumexpr with the sum of each term
            # each term is either a variable (which gets transformed into varexpr)
            # or a product of variables (which gets tranformed into a prod)
            sumexpr = SumExpr()
            for vars, coef in zf_ex.terms.items():
                 if len(vars) == 0:
                     sumexpr += coef
                 elif len(vars) == 1:
                     # add the 'new" Variables in the Goalfunction
                     varexpr = VarExpr(ips_vars[vars[0].name])
                     sumexpr += coef * varexpr
                 # unused, because function is linear
                 else:
                     prodexpr = ProdExpr()
                     for v in vars:
                          varexpr = VarExpr(ips_vars[v.name])
                          prodexpr *= varexpr
                     sumexpr += coef * prodexpr
            zf = sumexpr
        '''

        # Durch Rows iterieren und modifizierte Ungleichungen hinzufügen
        linear_rows = self.model.getLPRowsData()
        for lrow in linear_rows:
            vlist = [i.getVar() for i in lrow.getCols()]
            clist = lrow.getVals()

            beta = sum(abs(clist[i]) for i in range(len(clist)) if vlist[i].vtype != 'CONTINUOUS')
            # enlarged inner parallel set
            current_delta = 0
            if all(vlist[i].vtype != 'CONTINUOUS' for i in range(len(clist))) and all(
                    clist[i].is_integer() for i in range(len(clist))):
                current_delta = delta

            lhs = lrow.getLhs() + 0.5 * beta - current_delta
            rhs = lrow.getRhs() - 0.5 * beta + current_delta

            if lhs > (rhs + 10 ** (-6)):
                print('>>>> EIPS is empty')
                gran = False
                break

            ips_model.addCons(lhs <= (quicksum(ips_vars[vlist[i].name] * clist[i] for i in range(len(vlist))) <= rhs))

        if gran:
            print(">>>> Optimize over EIPS")
            ips_model.optimize()
            
            sol = self.model.createSol()
            for v in variables:
                if v.vtype != 'CONTINUOUS':
                    self.model.setSolVal(sol,v,round(ips_model.getVal(ips_vars[v.name])))
                else:
                    self.model.setSolVal(sol,v,ips_model.getVal(ips_vars[v.name]))
            
            accepted = self.model.trySol(sol)
            print(">>>> accepted solution? %s" % ("yes" if accepted == 1 else "no"))

            if accepted:
                return {"result": SCIP_RESULT.FOUNDSOL}
            else:
                return {"result": SCIP_RESULT.DIDNOTFIND}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}


def big_test_heur(testruns = 4):
    path_to_problems = '/home/stefan/Dokumente/opt_problems/miplib2010_benchmark/small_subset/'
    os.chdir(path_to_problems)
    problem_names = glob.glob("*.mps")
    
    iteration = 0
    
    for problem in problem_names:
        
        iteration += 1
        # create model
        m = Model()
    
        # create and add heuristic to SCIP
        heuristic = feasiblerounding()
        m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                      freq=0)

        # read exemplary problem from file
        print('>>>>> Working on Problem: ',problem)
        m.readProblem("".join([path_to_problems,problem]))
    
        # optimize problem
        m.optimize()

        # free model explicitly
        del m
        
        if iteration >= testruns:
            break

def test_single_heur():
    m = Model()
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE, freq=4)
    m.readProblem('/home/stefan/Dokumente/opt_problems/miplib2017_benchmark/50v-10.mps')
    m.optimize()
    del m
    

if __name__ == "__main__":
    test_single_heur()
