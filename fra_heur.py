
import gc, weakref, pytest
from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_PARAMSETTING, SCIP_HEURTIMING
from pyscipopt.scip import is_memory_freed
#from util import is_optimized_mode

#
# heuristic
#

class RoundingHeur(Heur):

   def heurinitsol(self):
      print(">>>> call heurinitsol()")

   def heurexitsol(self):
      print(">>>> call heurexitsol()")

   # execution method of the heuristic
   def heurexec(self, heurtiming, nodeinfeasible):

      print(">>>> Call feasible rounding heuristisc at a node with depth %d" % (self.model.getDepth()))
      
      # test: get Value of current LP-solution
      variables = self.model.getVars()
      assert isinstance(variables, object)
      for v in variables:
         print('Variable:',v.name)
         print('> Typ: ',str(v.vtype()))
         print('> Wert: ',str(v.getLPSol()))
         print('-----------')

      # linear rows of current LP-solution in one List
      # methods from 'Row'-class can be applied to them
      linear_rows_data = self.model.getLPRowsData()
      print(len(linear_rows_data))
      print('>>> Execution of Rows')
      # iterate through all rows
      for lrow in linear_rows_data:
         print('Linear Row:')
         columns = lrow.getCols()
         for col in columns:

         print(str(lrow))
         print('-----------')
      print('>>> Done with Rows')
      # create solution
      sol = self.model.createSol()
      #for v in vars:
      #   self.model.setSolVal(sol, v, 0.0)
      accepted = self.model.trySol(sol)
      print(">>>> accepted solution? %s" % ("yes" if accepted == 1 else "no"))

      if accepted:
         return {"result": SCIP_RESULT.FOUNDSOL}
      else:
         return {"result": SCIP_RESULT.DIDNOTFIND}


def test_heur():

   # create model
   m = Model()

   # create and add heuristic to SCIP
   heuristic = RoundingHeur()
   m.includeHeur(heuristic, "PyHeur", "feasible rounding heuristic", "Y", timingmask=SCIP_HEURTIMING.AFTERLPNODE, freq=0)

   # read problem from file
   m.readProblem('/home/stefan/Downloads/10teams.mps')

   # optimize problem
   m.optimize()

   # free model explicitly
   del m

if __name__ == "__main__":
    test_heur()
