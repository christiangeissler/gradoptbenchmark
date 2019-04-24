# -*- coding: utf-8 -*-
"""
Created on 28.03.2019
@author: christian.geissler@gt-arc.com
"""

'''
    A small, simple dirty dictionary database class, that can be used to simply store, save and load data to/from .json.
    The data is stored in a tree of dictionaries that are automatically created when referenced by a "store" method call.
    You can use any amount of arguments to encode the path. The last argument is assumed to be the value to be saved.

       # Usage: Create a database instance with
       db = Database()
       
       # store a value:
       db.store("a", "hierarchical", "path", 5)
       db.store("a", "nother", "path", 2)
       
       # save to file:
       db.saveToJson("myDbFile.json")
       
       # load from json:
       db2 = Database()
       db2.loadFromJson("myDbFile.json")
       
       # check, if a certain value exist:
       t = db2.exists("a", "hierarchical", "path")
       print(t) #should output "true"
       
       #retrieve values:
       v = db2.get("a", "hierarchical", "path")
       print(v) #should output "2"
       
       #retrieve the whole dictionary at "a" "hierarchical":
       d = db2.get("a", "hierarchical")  
'''


#system imports
import logging
import json #for tmp data storage

class Database:
    def __init__(self, *args, **kw):
        self.core = dict()

    '''
        Store a value in the database
        Input
            a couple of strings, followed by a value to be stored in the db.
            example: store("firstLayerId", "secondLayerId", 15)
    ''' 
    def store(self, *args):
        subStore = self.core
        for arg in args[:-2]:
            if not (arg in subStore):
                subStore[arg] = dict()
            subStore = subStore[arg]
        subStore[args[-2]] = args[-1]
        
    '''
        Check, if a certain layer or value exists
        Input
            a couple of strings.
        Output
            boolean
    '''
    def exists(self, *args):
        subStore = self.core
        for arg in args:
            if not (arg in subStore):
                return False
            subStore = subStore[arg]
        return True
    
    '''
        Retrieve a value from the database
        Input
            a couple of strings.
            example: get("firstLayerId", "secondLayerId")
        Output:
            Value or None, if that value doesn't exist
    ''' 
    def get(self, *args):
        subStore = self.core
        for arg in args:
            if not (arg in subStore):
                return None
            subStore = subStore[arg]
        return subStore
    
    '''
        load the metafeatures to a json file. You can load them with "loadFromJson".
        Input
            file[String] - path to the location + filename. Example: 'tmp/metastore.json'
    ''' 
    def saveToJson(self, file):
        with open(file, 'w') as outfile:
            json.dump(self.core, outfile)
            
    '''
        load the metafeatures from a json file. Warning: Overrides the current saved metafeatures of this instance.
        Input
            file[String] - path to the location + filename. Example: 'tmp/metastore.json'
    ''' 
    def loadFromJson(self, file):
        try:
            with open(file, 'r') as infile:
                self.core = json.load(infile)
        except FileNotFoundError:
            logging.warning('could not find json file to load from: '+str(file))
            pass 