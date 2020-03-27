import os
from fra_heur import *
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum
FEAS_TOL = 1E-6

def read_all_test_instances(folder):
    files = os.listdir('./'+folder)
    suffix = ".mps"
    pyfiles = [file for file in files if file.endswith(suffix)]
    testset = [file for file in pyfiles]
    return testset

def test_heur(model_path):
    m = Model()
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                  freq=5)
    m.readProblem(model_path)
    m.setIntParam('limits/restarts',0)
    m.setRealParam("limits/time",720)
    m.setLongintParam("limits/nodes",1)
    m.optimize()
    return m

