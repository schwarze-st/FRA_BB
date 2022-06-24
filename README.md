# FRA_BB: Numerical tests for Section 5.3 in "Feasible rounding based strategies in branch-and-bound methods for mixed-integer optimization"
We integrate feasible rounding approaches and diving ideas into the SCIP framework via pyscipopt. The following versions were used for our experiments:
- scipoptsuite7.0.0
- pycharm_community2020.1
- PySCIPOpt version: https://bitbucket.org/schwarzestefan/pyscipopt/src/master/
- version gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Please follow the instructions 
## The method
The diving procedure is implemented in <b>fra_heur.py</b> as instance of the class Heur in PySCIPOpt.
## Reproducing our results
There are scripts for all experiments and data we generated for the publication. 
The jupyter notebook in <b>results/notebook/results_nb.ipynb</b> extracts all relevant data from the generated output-files 
### 1. Filtering the test bed: <b>filter_instances.py</b>
See results/notebook/results_nb.ipynb
### 2. Running SCIP with fra_heur.py on the test bed: <b>diving_analysis.py</b>
We run SCIP with fra_heur.py on the obtained test bed with 128 instances with up to 5 diving rounds and 30 minutes maximum run time.
### 3. Running SCIP with plain settings: <b>SCIP_plain.py</b>
We run SCIP without fra_heur.py for 32 instances where it found best solutions until the solution is better than the one found with fra_heur.py.
