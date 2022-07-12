import pickle, pandas as pd

results_frame = pd.DataFrame()

for i in range(8):
    path = 'results/filter_instances/raw_results/temp_results'+str(i+1)+'.pickle'
    print('Append temp_results'+str(i+1)+' to DataFrame.')
    data = []
    with open(path, 'rb') as handle:
        try:
            while True:
                data.append(pickle.load(handle))
        except EOFError:
            pass
    print('There are '+str(len(data))+ ' entries.')
    for j in range(0,len(data)):
        temp_frame = pd.DataFrame(data[j])
        if j==0 and i==0:
            results_frame = temp_frame
        else:
            results_frame = pd.concat([results_frame,temp_frame], ignore_index = True)
            a = len(results_frame.index)
            results_frame = results_frame.drop_duplicates(subset=[results_frame.columns[0]])
            if (a-len(results_frame.index))>0:
                print('There is a duplicate')
                print(temp_frame)

results_frame = results_frame.reindex(columns=['name', 'depth', 'eq_constrs', 'pruned_prob','ips_nonempty', 'feasible', 'accepted',
                                                   'obj_best', 'obj_SCIP', 'obj_root', 'obj_ls',
                                                   'time_heur', 'time_solveips', 'time_pp', 'time_scip','time_diving_lp', 'time_diving_prop',
                                                   'diving_lp_solves', 'diving_depth', 'diving_best_depth', 'obj_diving'])
results_frame.to_pickle('results/filter_instances/filter_collection_dataframe')
print(results_frame.to_string())