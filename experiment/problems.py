#python basic imports
import math
#3rd party imports (from packages, the environment)
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import csv
import os
#Custom imports
from util.env import ProblemWrapper

#used for caching datasets downloaded from somewhere else
DATASET_CACHE_HOME = os.path.normpath('.tmp/')


class WeightedLinearMLProblemTemplate():
    def __init__(self):
        super(WeightedLinearMLProblemTemplate, self).__init__()
        
    def _function(self, x):
        thetaParam = np.power(10, x[0])
        lambdaParam = np.power(10, x[1])
        x = x[2:]
        (Features, targets) = self._prepareDataset()
        kf = KFold(n_splits = 10, shuffle = True)
        scoreList = list()
        for train_index, test_index in kf.split(Features):
            regressor = KernelRidge(alpha=lambdaParam, kernel='rbf', gamma=thetaParam)
            regressor.fit(X=Features[train_index], y=targets[train_index], sample_weight=x[train_index])
            scoreList.append(regressor.score(X=Features[test_index],y=targets[test_index]))
        return np.mean(scoreList)

    
    def _domain(self):
        (Features, targets) = self._prepareDataset()
        dimension = np.shape(targets)[0]
        hpranges = [[-2.,4.],[-5.,5.]]
        weights = [[0,1] for i in range(dimension)]
        array = np.array(hpranges+weights)
        #array = np.array(hpranges)
        return array
        
    def cleanData( self, X , y ):
        mask = np.isnan( X )
        #print("Missing values: "+str(np.sum(mask)))
        mask = np.invert( np.any( mask, axis = 1 ) )
        X = X[mask,:]
        y = y[mask]
        
        mask = np.isinf( X )
        #print("Infinite values: "+str(np.sum(mask)))
        mask = np.invert( np.any( mask, axis = 1 ) )
        X = X[mask,:]
        y = y[mask]
        return (X,y)
        
class AutoMPGHD(WeightedLinearMLProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(AutoMPGHD, self).__init__()
        dataset = fetch_openml(data_id=196, data_home=DATASET_CACHE_HOME)
        self.dataset = self.cleanData(X=dataset.data,y=dataset.target)

    def _prepareDataset(self):
        return self.dataset
        
class BreastCancerHD(WeightedLinearMLProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(BreastCancerHD, self).__init__()
        dataset = fetch_openml(data_id=13, data_home=DATASET_CACHE_HOME)
        (X,y) = self.cleanData(X=dataset.data,y=dataset.target)
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.dataset = (X,y)
        
    def _prepareDataset(self):
        return self.dataset
        
class SlumpHD(WeightedLinearMLProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(SlumpHD, self).__init__()
        dataset = fetch_openml(data_id=41490, data_home=DATASET_CACHE_HOME)
        self.dataset = self.cleanData(X=dataset.data,y=dataset.target)

    def _prepareDataset(self):
        return self.dataset
    
class YachtHD(WeightedLinearMLProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(YachtHD, self).__init__()
        dataset = list()
        with open(os.path.normpath('./data/yacht/yacht_hydrodynamics.data')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                #skip empty rows
                if len(row) == 0:
                    continue
                try:
                    dataset.append([float(val) for val in row if not(val=='')])
                except Exception as exceptionInstance:
                    print(exceptionInstance)
                
        dataset = np.array(dataset)
        #separate features and target values
        self.dataset = self.cleanData(X=dataset[:,0:-1],y=dataset[:,-1])
        
    def _prepareDataset(self):
        return self.dataset
        
class HousingHD(WeightedLinearMLProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(HousingHD, self).__init__()
        dataset = fetch_openml(data_id=531, data_home=DATASET_CACHE_HOME)
        self.dataset = self.cleanData(X=dataset.data,y=dataset.target)

    def _prepareDataset(self):
        return self.dataset

#Objective functions (actually the whole problem, function plus input domain range)
class RBFRRProblemTemplate():
    def __init__(self):
        super(RBFRRProblemTemplate, self).__init__()
    
    def _domain(self):
        return np.array([
            [-2,4],
            [-5,5]
        ])
    
    def _function(self, x):
        #alpha = regularization parameter C
        #gamma = variance, bandwidth
        thetaParam = np.power(10, x[0])
        lambdaParam = np.power(10, x[1])
        kernelRidgeRegression = KernelRidge(alpha=lambdaParam, kernel='rbf', gamma=thetaParam)
        (X, y) = self._prepareDataset()
        return np.mean( cross_val_score(kernelRidgeRegression, X, y, cv=10, n_jobs = 1, pre_dispatch = 1) )
    
    #returns: X,y - X:clean feature values, y:label values. No NaN or inf contained!
    def _prepareDataset(self):
        return NotImplemented
        
    def cleanData( self, X , y ):
        mask = np.isnan( X )
        #print("Missing values: "+str(np.sum(mask)))
        mask = np.invert( np.any( mask, axis = 1 ) )
        X = X[mask,:]
        y = y[mask]
        
        mask = np.isinf( X )
        #print("Infinite values: "+str(np.sum(mask)))
        mask = np.invert( np.any( mask, axis = 1 ) )
        X = X[mask,:]
        y = y[mask]
        return (X,y)
    
    
class AutoMPG(RBFRRProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(AutoMPG, self).__init__()
        dataset = fetch_openml(data_id=196, data_home=DATASET_CACHE_HOME)
        self.dataset = self.cleanData(X=dataset.data,y=dataset.target)

    def _prepareDataset(self):
        return self.dataset

class BreastCancer(RBFRRProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(BreastCancer, self).__init__()
        dataset = fetch_openml(data_id=13, data_home=DATASET_CACHE_HOME)
        (X,y) = self.cleanData(X=dataset.data,y=dataset.target)
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.dataset = (X,y)
        
    def _prepareDataset(self):
        return self.dataset

    
class Slump(RBFRRProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(Slump, self).__init__()
        dataset = fetch_openml(data_id=41490, data_home=DATASET_CACHE_HOME)
        self.dataset = self.cleanData(X=dataset.data,y=dataset.target)

    def _prepareDataset(self):
        return self.dataset
    
class Yacht(RBFRRProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(Yacht, self).__init__()
        dataset = list()
        with open(os.path.normpath('./data/yacht/yacht_hydrodynamics.data')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                #skip empty rows
                if len(row) == 0:
                    continue
                try:
                    dataset.append([float(val) for val in row if not(val=='')])
                except Exception as exceptionInstance:
                    print(exceptionInstance)
                
        dataset = np.array(dataset)
        #separate features and target values
        self.dataset = self.cleanData(X=dataset[:,0:-1],y=dataset[:,-1])
        
    def _prepareDataset(self):
        return self.dataset
        
class Housing(RBFRRProblemTemplate, ProblemWrapper):
    def __init__(self):
        super(Housing, self).__init__()
        dataset = fetch_openml(data_id=531, data_home=DATASET_CACHE_HOME)
        self.dataset = self.cleanData(X=dataset.data,y=dataset.target)

    def _prepareDataset(self):
        return self.dataset

class HolderTable(ProblemWrapper):
    #no of local maxima = 36
    def _domain(self):
        return np.array([
            [-10,10],
            [-10,10]
        ])

    def _function(self, x):
        return abs(math.sin(x[0]))*abs(math.cos(x[1]))*math.exp(abs(1-math.sqrt(x[0]*x[0]+x[1]*x[1])/math.pi))

class Rosenbrock(ProblemWrapper):
    def _domain(self):
        return np.array([
            [-2.048,2.048],
            [-2.048,2.048],
            [-2.048,2.048]
        ])

    def _function(self, x):
        return -np.sum([np.abs(100*np.power(x[i+1]-(x[i]*x[i]),2) + np.power(x[i],2)) for i in range(2) ])

class Sphere(ProblemWrapper):
    #no of local maxima = 1
    def _domain(self):
        return np.array([
            [0,1.0],
            [0,1.0],
            [0,1.0],
            [0,1.0]
        ])

    def _function(self, x):
        return -np.sqrt(np.sum([np.power(x[i] - math.pi/16,2) for i in range(4)]))
        
class SphereHighDim(ProblemWrapper):
    #no of local maxima = 1
    def _domain(self):
        return np.array([
            [0,1.0],
            [0,1.0],
            [0,1.0],
            [0,1.0],
            [0,1.0],
            [0,1.0],
            [0,1.0],
            [0,1.0],
            [0,1.0],
            [0,1.0]
        ])

    def _function(self, x):
        return -np.sqrt(np.sum([np.power(x[i] - math.pi/16,2) for i in range(10)]))

class LinearSlope(ProblemWrapper):
    #no of local maxima = 1
    def _domain(self):
        return np.array([
            [-5,5],
            [-5,5],
            [-5,5],
            [-5,5]
        ])

    def _function(self, x):
        return np.sum([np.power(10,(i-1)/4.)*(x[i]-5) for i in range(4)])

class DebN1(ProblemWrapper):
    #no of local maxima = 36
    def _domain(self):
        return np.array([
            [-5,5],
            [-5,5],
            [-5,5],
            [-5,5],
            [-5,5]
        ])

    def _function(self, x):
        return 0.2 * np.sum( [ np.power( math.sin( 5 * math.pi * x[i] ), 6 ) for i in range( 5 ) ] )
        
class BraninHoo(ProblemWrapper):
    #no of local maxima = 3
    def _domain(self):
        return np.array([
            [-5,10],
            [0,15],
        ])

    def _function(self, x):
        return 10 * (1-(1/(8*math.pi))) * math.cos(x[0]) + 10 + np.power(x[1] - 5.1*x[0]*x[0] / (4*math.pi*math.pi)+5*x[0]/math.pi - 6,2)

class Himmelblau(ProblemWrapper):
    #no of local maxima = 4
    def _domain(self):
        return np.array([
            [-5,5],
            [-5,5],
        ])

    def _function(self, x):
        return - np.power( (np.power(x[0],2) + x[1] - 11), 2) - np.power( (x[0] + np.power(x[1],2) - 7), 2)

class Styblinski(ProblemWrapper):
    #no of local maxima = 4
    def _domain(self):
        return np.array([
            [-5,5],
            [-5,5],
        ])

    def _function(self, x):
        return 8*np.power(x[1],2) - 0.5*np.power(x[1],4) - 2.5*x[1] + 8*np.power(x[0],2) - 0.5*np.power(x[0],4) - 2.5*x[0]

class LevyN13(ProblemWrapper):
    #no of local maxima > 100
    def _domain(self):
        return np.array([
            [-10,10],
            [-10,10],
        ])

    def _function(self, x):
        return -np.power( (x[0]-1), 2) * ( 1 + np.power( math.sin(3*math.pi*x[1]), 2))
        
class MishraN2(ProblemWrapper):
    #no of local maxima -
    def _domain(self):
        return np.array([
            [0,1],
            [0,1],
            [0,1],
            [0,1],
            [0,1],
            [0,1]
        ])

    def _function(self, x):
        return - np.power( ( 6 - np.sum( [ 0.5*(x[i] + x[i+1]) for i in range( 5 ) ] )), 5 - np.sum( [ 0.5*(x[i] + x[i+1]) for i in range( 5 ) ] ))

class GriewankN4(ProblemWrapper):
    #no of local maxima > 100
    def _domain(self):
        return np.array([
            [-300,600],
            [-300,600],
            [-300,600],
            [-300,600]
        ])

    def _function(self, x):
        a = -1
        b = - np.sum( [ x[i]*x[i]/4000 for i in range(4)] )
        c = np.prod( [ math.cos(x[i]/math.sqrt(i+1)) for i in range(4)] )
        return a+b+c

class TestProblem(ProblemWrapper):
    #single dimension function test.
    def _domain(self):
        return np.array([
            [-5,5]
        ])

    def _function(self, x):
        return 1.0 - np.dot(x.T,x)

#func2 from test.py
class TestProblem2(ProblemWrapper):
    #multi dimension functiont test.
    def _domain(self):
        return np.array([
            [-10.,10.],
            [-10,10.]
        ])

    def _function(self, x):
        return 1 -(x[0] - 2)*(x[0] - 2)*(x[1] - 1)*(x[1] - 1)

#Kernel Ridge Regression Problems
REGRESSION = {"housing":Housing, "yacht":Yacht, "slump":Slump, "breastcancer":BreastCancer, "autompg":AutoMPG}
REGRESSION_HD = {"housinghd":HousingHD, "yachthd":YachtHD, "slumphd":SlumpHD, "breastcancerhd":BreastCancerHD, "autompghd":AutoMPGHD}

#Synthetic benchmark as in Global optimization of Lipschitz functions        
SYNTHETIC_ADALIPO = {"holdertable":HolderTable, "rosenbrock":Rosenbrock, "sphere":Sphere, "linearslope":LinearSlope, "debn1":DebN1}
#Synthetic benchmark set as in 2016 A Ranking Approach to Global Optimization Cedriv Malherbe
SYNTHETIC_RGO = {"branin-hoo":BraninHoo, "himmelblau":Himmelblau, "styblinski":Styblinski, "holdertable":HolderTable, "levyn13":LevyN13, "rosenbrock":Rosenbrock, "mishran2":MishraN2, "linearslope":LinearSlope, "debn1":DebN1, "griewankn4":GriewankN4}
SYNTHETIC_ALL = {"holdertable":HolderTable, "rosenbrock":Rosenbrock, "sphere":Sphere, "linearslope":LinearSlope, "debn1":DebN1, "spherehighdim":SphereHighDim, "branin-hoo":BraninHoo, "himmelblau":Himmelblau, "styblinski":Styblinski, "levyn13":LevyN13, "mishran2":MishraN2, "griewankn4":GriewankN4 }


#A simple function test testset
TEST = {"test1":TestProblem, "test2":TestProblem2}

ALL = {str(p.__name__):p for p in ProblemWrapper.__subclasses__()}