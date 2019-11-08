import os
from fra_heur import *
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum

def read_all_test_instances(folder):
    files = os.listdir('./'+folder)
    suffix = ".mps"
    pyfiles = [file for file in files if file.endswith(suffix)]
    testset = [file for file in pyfiles]
    return testset

def test_heur(model_path):
    m = Model()
    options = {'mode':['original','deep_fixing']}
    heuristic = feasiblerounding(options)
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.DURINGLPLOOP,
                  freq=5)
    m.readProblem(model_path)
    m.setRealParam("limits/time",10.0)
    m.setLongintParam("limits/nodes",10)
    m.optimize()
    del m