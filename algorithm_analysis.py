import os
from fra_heur import *
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum
import pandas as pd
FEAS_TOL = 1E-6

def read_all_test_instances(folder):
    os.chdir(folder)
    files = os.listdir('./')
    suffix = ".mps"
    pyfiles = [file for file in files if file.endswith(suffix)]
    testset = [file for file in pyfiles]
    os.chdir("..")
    return testset

def test_heur(model_path, model_name):
    m = Model(problemName=model_name)
    heuristic = feasiblerounding()
    m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE,
                  freq=1)
    m.readProblem(model_path)
    m.setIntParam('limits/restarts',0)
    m.setRealParam("limits/time",1800)
    m.setLongintParam("limits/nodes",1)
    m.hideOutput(True)
    m.optimize()
    m.freeProb()
    del m

def convert_dict_to_dataframe(name='results/FRA_Scip.pickle'):
    data = []
    with open(name, 'rb') as handle:
        try:
            while True:
                data.append(pickle.load(handle))
        except EOFError:
            pass
    results_frame = pd.DataFrame()
    for i in range(0,len(data)):
        temp_frame = pd.DataFrame(data[i])
        if i==0:
            results_frame = temp_frame
        else:
            results_frame = pd.concat([results_frame,temp_frame], ignore_index = True)
    results_frame = results_frame.reindex(columns=['name', 'depth', 'eq_constrs', 'pruned_prob','ips_nonempty', 'feasible', 'accepted',
                                                   'obj_best', 'obj_SCIP', 'obj_root', 'obj_ls',
                                                   'time_heur', 'time_solveips', 'time_pp', 'time_scip','time_diving_lp', 'time_diving_prop',
                                                   'diving_lp_solves', 'diving_depth', 'diving_best_depth', 'obj_diving'])
    results_frame.to_pickle('results/filter_collection_log_dataframe')
    print(results_frame.to_string())
