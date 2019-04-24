from util.env import OptimizerWrapper
import experiment.optimizers as optimizers
import experiment.problems as problems
import hashlib
import multiprocessing

NO_OF_PARALLEL_EXECUTIONS = multiprocessing.cpu_count()-1 #don't make this larger than the number of cores
REPEAT_INCOMPLETE = True #when re-run, recalculate evaluations (k) where there are incomplete results.

#Experimental Settings, taken from the paper "Global optimization of Lipschitz functions" 2017 C.Malherbe et. al
DEBUG_SETTINGS_OVERWRITE = False

#number of repeated evaluations for each combination of an optimization algorithm with an objective function
K = 100

#how many function evaluations each optimization algorithm is allowed to make
MAX_EVALUATIONS = 1000

#target values, we measure how many evaluations are required to reach the different target values (fraction of the normalized maximal achievable value)
TARGETS = [0.9,0.95,0.99]

DATABASE_PATH = "results.json"

ENABLE_SYNTHETIC_PROBLEMS = True
ENABLE_DATASET_REGRESSION_PROBLEMS = True
ENABLE_TEST_PROBLEMS = False

#Optimization functions
LIST_OF_OPTIMIZERS = OptimizerWrapper.__subclasses__()

#Objective functions (actually the whole problem, function plus input domain range)
LIST_OF_PROBLEMS = list()

if (DEBUG_SETTINGS_OVERWRITE):
    K = 10
    MAX_EVALUATIONS = 100
    DATABASE_PATH = "debug.json"
    ENABLE_SYNTHETIC_PROBLEMS = True
    ENABLE_DATASET_REGRESSION_PROBLEMS = False
    ENABLE_TEST_PROBLEMS = False
    #LIST_OF_PROBLEMS.append(problems.Yacht)

if (ENABLE_SYNTHETIC_PROBLEMS):
    LIST_OF_PROBLEMS.extend(list(problems.SYNTHETIC_ALL.values()))
    
if (ENABLE_DATASET_REGRESSION_PROBLEMS):
    LIST_OF_PROBLEMS.extend(list(problems.REGRESSION.values()))

if (ENABLE_TEST_PROBLEMS):
    LIST_OF_PROBLEMS.extend(list(problems.TEST.values()))
    
    

#Calculate a hash from this configuration file (as a simple check to ensure the evaluation is done with the same configuration)
sha3 = hashlib.sha3_256()
with open("experiment/config.py", 'rb') as f:
    while True:
        data = f.read(65536)
        if not data:
            break
        sha3.update(data)       
EVALUATION_HASH = sha3.hexdigest()