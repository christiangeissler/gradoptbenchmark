# -*- coding: utf-8 -*-

import numpy as np


class GradSearch(object):
    """
    The hierarchical optimistic optimization algorithm.
    """

    def __init__(self,d, n, delta, covering,environment_function):
        self.space = covering
        self.environment_function = environment_function
        self.x =(1.0*self.space.upper + self.space.lower) / 2.0
        self.f_x=self.environment_function(self.x)
        self.x_opt =self.x
        self.f_x_opt=self.f_x
        self.r =(1.0*self.space.upper - self.space.lower) / 2.0
        self.delta=np.max(self.r)
        self.n = n
        self.d = d
        self.eta=np.zeros(shape=self.x.shape)
               
    
    def setspace(self, h , i):
        self.space= self.covering_generator_function(h, i)
    
    def gradientSearch(self):
        v= np.random.normal(size=self.x.shape)
        tilde_f_x= self.evaluate(self.x+self.delta*v)
        g= self.d/self.delta*(tilde_f_x-self.f_x)*v
        self.eta = self.eta+g*g
        self.step(g)
        self.project()
        self.f_x=self.environment_function(self.x)
        if self.f_x_opt<self.f_x:
            self.f_x_opt= self.f_x
            self.x_opt= self.x
        
       
    def evaluate(self, x):
        y=np.zeros(shape = x.shape)
        for d in range(x.shape[0]):
            if x[d]>self.space.upper[d]:
                y[d]= self.space.upper[d]
            elif x[d]<self.space.lower[d]:
                y[d]= self.space.lower[d]
            else:
                y[d]= x[d]
        return self.environment_function(y)
                
    
        
    def step(self,g):
        for d in range(self.eta.shape[0]):
            if np.abs(self.eta[d])>1e-6:
                self.x[d]= self.x[d]+np.sqrt(self.r[d])/np.sqrt(self.eta[d])*g[d]


    def project(self):
        for d in range(self.x.shape[0]):
            if self.x[d]>self.space.upper[d]:
                self.x[d]= self.space.upper[d]
            elif self.x[d]<self.space.lower[d]:
                self.x[d]= self.space.lower[d]
                
    
    def run_search(self):
        """
        run the agent for "n" rounds.
        """
        t = 0
        #note: n-1 because we make two calls to the evaluation function in gradientSearch()
        while (t*2)< self.n-1:   
            self.gradientSearch()    
            t = t + 1
            if t % 100 == 0:
                self.delta=self.delta/2
                self.f_x= self.f_x_opt
                self.x = self.x_opt
                #print("reward {0}".format(self.f_x_opt))
                #print("action {0}".format(self.x_opt))

        return self.f_x_opt
