#python basic imports

#3rd party imports (from packages, the environment)
import numpy as np

#Custom imports
from util.env import OptimizerWrapper
from xarmedbandits import Partitioner
from xarmedbandits import TreeNode
from xarmedbandits import HOO
from bayes_opt import BayesianOptimization as BayesianOpt
from gradopt import GradSearch
import dlib
import inspect
import random

#from pyLIPO.lipo_modified import adalipoContinous
#import sklearn.gaussian_process as gp
#from keras import optimizers as kerasopt
ADALIPODLIB_ENABLED = True
PRS_ENABLED = True #pure random search
BO_ENABLED = False #bayesian optimization (slow!)
HOO_ENABLED = True
GRADOPT_ENABLED = True 
SMAC_ENABLED = False #if true, tries to enable smac and integrates it's wrapper code into the benchmark.
try:
    if (SMAC_ENABLED):
        from smac.facade.func_facade import fmin_smac #only on linux
        SMAC_ENABLED = True
except ModuleNotFoundError as error:
    SMAC_ENABLED = False

            
#Another implementation of the AdaLipo Algorithm that also applies a local quadratic search at the end.  
#From Davis E. King 2017 http://dlib.net/dlib/global_optimization/find_max_global_abstract.h.html#find_max_global
if ( ADALIPODLIB_ENABLED ):
    class AdaLipoTR(OptimizerWrapper):
            def apply(self, problem, iterations):
                X = problem.getDomain()
                minValues = np.min(X,axis=1).tolist()
                maxValues = np.max(X,axis=1).tolist()
                
                def fWrapper2(*args):
                    return problem.evaluate(np.array(args))
                    
                spec_F = dlib.function_spec(minValues, maxValues)
                opt = dlib.global_function_search(spec_F)
                opt.set_seed(random.randint(0,10000000))
                opt.set_solver_epsilon(0)
                for i in range(iterations):
                    next = opt.get_next_x()
                    next.set(fWrapper2(*next.x))

                return problem
                
if ( ADALIPODLIB_ENABLED ):
    class AdaLipo(OptimizerWrapper):
            def apply(self, problem, iterations):
                X = problem.getDomain()
                minValues = np.min(X,axis=1).tolist()
                maxValues = np.max(X,axis=1).tolist()
                
                def fWrapper2(*args):
                    return problem.evaluate(np.array(args))
                    
                spec_F = dlib.function_spec(minValues, maxValues)
                opt = dlib.global_function_search(spec_F)
                opt.set_seed(random.randint(0,10000000))
                opt.set_solver_epsilon(100000)#NO TR SETTING
                for i in range(iterations):
                    next = opt.get_next_x()
                    next.set(fWrapper2(*next.x))

                return problem

#This optimizer is the same as pure random search, but not used for the actual comparison. 
#Since we need to scale the target value according to the mean value of the function, we cannot use the evalutions of the other classifiers cause they don't sample uniform from f(x).
#We therefore use this optimizer to sample 1.000.000 f(x) values to approximate the mean. However, for max(f(x)) we use all the evaluations because there, uniformly drawn x don't matter.
class MonteCarloSampling1M(OptimizerWrapper):
    def apply(self, problem, iterations):  
        iterations = 10*iterations #np.power(10,6) should do 1.000.000 evaluations in the final setting. With K=100 and iterations=1000 we need to multiply with 10 to get 1.000.000 random function evaluations.
        X = problem.getDomain()
        minValues = np.min(X,axis=1)
        maxValues = np.max(X,axis=1)
        for i in range(iterations):
            problem.evaluate(np.random.uniform( low=minValues, high=maxValues))
        return problem

if ( PRS_ENABLED ):
    class PureRandomSearch(OptimizerWrapper):
        def apply(self, problem, iterations):  
            X = problem.getDomain()
            minValues = np.min(X,axis=1)
            maxValues = np.max(X,axis=1)
            for i in range(iterations):
                problem.evaluate(np.random.uniform( low=minValues, high=maxValues))
            return problem
if ( BO_ENABLED ):    
    class BayesianOptimization(OptimizerWrapper):
        def apply(self, problem, iterations):
            #wrap the target domain into a dictionary as required by bayesianOpt implementation:
            domain = problem.getDomain()
            d = np.shape(domain)[0]
            minValues = np.min(domain,axis=1)
            maxValues = np.max(domain,axis=1)
            
            pbounds = dict()
            for i in range(d):
                pbounds[str(i)] = (minValues[i],maxValues[i])
                
            def fWrapper(**args ):
                x = np.array( [ x[1] for x in args.items()] )
                return problem.evaluate(x)  
                
            optimizer = BayesianOpt( f=fWrapper, pbounds = pbounds , random_state=None , verbose=0)
            optimizer.maximize( init_points = 2, n_iter = iterations -2 )
            
            return problem

if (HOO_ENABLED ):            
    class HierarchicalOptimisticOptimization(OptimizerWrapper):
        def apply(self, problem, iterations):  
            X = problem.getDomain()
            minValues = np.min(X,axis=1)
            maxValues = np.max(X,axis=1)
            partitioner = Partitioner.Partitioner(min_values=minValues, max_values=maxValues)
            x_armed_bandit = HOO.HOO(v1=60, ro=0.5, covering_generator_function=partitioner.halve_one_by_one)
            x_armed_bandit.set_time_horizon(max_plays=iterations+1)
            x_armed_bandit.set_environment(environment_function=problem.evaluate)
            x_armed_bandit.run_hoo()
            return problem

'''
#not working right now
    class GraduatedHierarchicalOptimisticOptimization(OptimizerWrapper):
        def apply(self, problem, iterations):  
            X = problem.getDomain()
            minValues = np.min(X,axis=1)
            maxValues = np.max(X,axis=1)
            partitioner = Partitioner.Partitioner(min_values=minValues, max_values=maxValues)
            x_armed_bandit = GradHOO.GradHOO(d=np.shape(X)[0], n=iterations, delta= 0.0001, covering_generator_function=partitioner.halve_one_by_one)
            x_armed_bandit.set_environment(environment_function=problem.evaluate)
            x_armed_bandit.run_hoo()
            return problem
'''

if ( True == SMAC_ENABLED ):
    class SMAC(OptimizerWrapper):
        def apply(self, problem, iterations): 
        
            def fWrapper( x ):
                return -problem.evaluate(x)  
                
            X = problem.getDomain()
            x0 = np.mean(X,axis=1)#center point as initial starting point
            
            fmin_smac(func=fWrapper,  # function
                x0=x0,    # default configuration
                bounds=problem.getDomain(),  # limits
                maxfun=iterations,   # maximum number of evaluations
                rng=3)
            return problem
                  
if ( GRADOPT_ENABLED ):       
    class GradOpt(OptimizerWrapper):
        def apply(self, problem, iterations):  
            X = problem.getDomain()
            minValues = np.min(X,axis=1)
            maxValues = np.max(X,axis=1)
            #n=iteration-1 since somehow gradopt uses one more iteration than intended.
            opt = GradSearch.GradSearch(d=np.shape(X)[0], n=iterations-1, delta=5, covering=Partitioner.Covering(lowers=minValues, uppers=maxValues),environment_function=problem.evaluate)
            opt.run_search()
            return problem
'''   
#not working right now
class SGD(OptimizerWrapper):
    def apply(self, problem, iterations):
        opt = kerasopt.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        
        opt.get_updates(loss, params):
'''