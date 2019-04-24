# -*- coding: utf-8 -*-
"""
Created on 28.03.2019
@author: christian.geissler@gt-arc.com
"""
#system
import logging

#3rd party
import numpy as np

#custom

def setupLogging(logfile="debug.log"):
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    
    #loggingFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    loggingFormatter = logging.Formatter('%(message)s')
    
    loggingFileHandler = logging.FileHandler(logfile, mode='w')
    loggingFileHandler.setLevel(logging.DEBUG)
    loggingFileHandler.setFormatter(loggingFormatter)
    logging.getLogger('').addHandler(loggingFileHandler)
    
    loggingStreamHandler = logging.StreamHandler()
    loggingStreamHandler.setLevel(logging.DEBUG)
    loggingStreamHandler.setFormatter(loggingFormatter)
    logging.getLogger('').addHandler(loggingStreamHandler)
    
    #replace regular print with logger info function, because some sklearn implementation (I'm not looking into a specific direction - RandomizedSearchCV) use print for logging...
    #print = logger.info
    
    logger.info('Start Logging')
    return logger
    
def shutdownLogging():
    logger = logging.getLogger('')
    logger.info("Shutdown logger, bye bye!")
    #create a copy, so that we can savely manipulate the original list:
    copyOfList = list(logger.handlers)
    for handlerInstance in copyOfList:
        logger.removeHandler(handlerInstance)
    logger.handlers = []
    del logger
    logging.shutdown()