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

def test_heur(model_path, model_name):
    m = Model(problemName=model_name)
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                  freq=1)
    m.readProblem(model_path)
    m.setIntParam('limits/restarts',0)
    m.setRealParam("limits/time",3600)
    m.setLongintParam("limits/nodes",10)
    m.optimize()
    return m

def convert_dict_to_dataframe(problemnames):
    data = []
    with open('results/FRA_Scip.pickle', 'rb') as handle:
        try:
            while True:
                data.append(pickle.load(handle))
        except EOFError:
            pass
    results_frame = pd.DataFrame(data)
    results_frame.index = problemnames
    results_frame.to_pickle('results/FRA_Scip_dataframe')
    print(results_frame.to_string())
