# -*- coding: utf-8 -*-
"""
Created on 28.03.2019
@author: christian.geissler@gt-arc.com
"""
import numpy as np
import logging
#Experimental abstract layer
class ProblemWrapper:
    def __init__(self):
        super(ProblemWrapper, self).__init__()
        self.scores = list()
        self.params = list()
    
    #Needs to be implemented
    def _function(self, x):
        return NotImplemented
    
    #Needs to be implemented
    #return format: np.array([[x0Min,x0Max],[x1Min,x1Max],...]])
    def _domain(self):
        return NotImplemented
        
    def evaluate(self, x):
        score = self._function(x)
        self.params.append(x)
        self.scores.append(score)
        #print("score: "+str(score)+" @ round "+str(len(self.scores)))
        return score
    
    def getDomain(self):
        return self._domain()

    def getScores(self):
        return self.scores
                           
    def getParams(self):
        return self.params
        
    def getResultAsDictionary(self):
        if len(self.scores) > 0:
            bestIndex = np.argmax(self.scores)
            bestScore = self.scores[bestIndex]
            bestParams = self.params[bestIndex]
            meanScore = np.mean(self.scores)
            noOfEvaluations = len(self.scores)
            return {"bestscore":bestScore,"bestparams":bestParams.tolist(),"meanScore":meanScore,"noOfEvaluations":noOfEvaluations, "scores":self.scores}
        else:
            assert False,"error: tried to get result without running any evaluation"
            
    def __str__(self):
        return str(self.getResultAsDictionary())
            
class OptimizerWrapper:
    def __init__(self):
        super(OptimizerWrapper, self).__init__()
    
    def apply(self, problem, iterations):
        return NotImplemented