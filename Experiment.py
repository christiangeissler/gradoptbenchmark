#python basic imports
import math
import multiprocessing
import time
import os#to rename files
#3rd party imports (from packages, the environment)
import numpy as np

#custom (local) imports
import experiment.config as config
from util.database import Database
from util.logging import setupLogging, shutdownLogging
#from IPython.display import clear_output
from util.worker import worker

MKL_ENABLED = False
try:
    import mkl #to limit multiprocessing (to get better control)
    MKL_ENABLED = True
except ModuleNotFoundError as error:
    MKL_ENABLED = False

#Experimental Setup
if __name__ == '__main__':
    logger = setupLogging()
    db = Database()
    db.loadFromJson(config.DATABASE_PATH)
    if (MKL_ENABLED): mkl.set_num_threads(1)#disable mkl multiprocessing to get more control over the number of cores being used.
    taskList = []
    logger.info("Collect tasks that needs to be calculated")
    for optimizer in config.LIST_OF_OPTIMIZERS:
        for problem in config.LIST_OF_PROBLEMS:
            optimizerId = str(optimizer.__name__)
            problemId = str(problem.__name__)
            for k in range(config.K):
                if not db.exists(optimizerId,problemId,str(k)):
                    #optimizer().apply(problem(), config.MAX_EVALUATIONS)
                    taskList.append({'optimizer':optimizer, 'problem':problem, 'maxeval':config.MAX_EVALUATIONS, 'k':k})
                elif ( config.REPEAT_INCOMPLETE and ( np.shape( ( db.get(optimizerId,problemId,str(k)) )['scores'] )[0] < config.MAX_EVALUATIONS ) ):
                    logger.info("Repeat evaluation of "+str(optimizerId)+" on "+str(problemId)+" k="+str(k)+" because the previous did not complete 100%")
                    logger.info("Last results: "+str(np.shape( ( db.get(optimizerId,problemId,str(k)) )['scores'] )[0]))
                    #if there is already an evaluation, but it contains less values than configured (meaning it stopped somewhere because of some numerical issues)
                    #we add it again and hope that the issues will not appear this time (this is mostly for AdaLipo)
                    taskList.append({'optimizer':optimizer, 'problem':problem, 'maxeval':config.MAX_EVALUATIONS, 'k':k})
                    
    logger.info("Start benchmark")
    pool = multiprocessing.Pool(processes=config.NO_OF_PARALLEL_EXECUTIONS)
    counter = 0
    total = len(taskList)
    lastSave = 0
    timeUntilNextSave = 0 #in seconds
    for result in pool.imap_unordered(worker, taskList, chunksize=1 ):
        counter += 1
        #clear_output(wait=True)
        optimizerId = str(result['optimizer'].__name__)
        problemId = str(result['problem'].__name__)
        k = result['k']
        
        if ( 'result' in result ):
            p = result['result']

            logger.info("Finished evaluation "+str(counter)+"/"+str(total)+" ( "+str(100*counter/total)+"%)")
            logger.info(optimizerId+" on "+problemId + " k="+str(k)+"(0-"+str(config.K-1)+")")
            #logger.debug(p.getResultAsDictionary())

            db.store(optimizerId,problemId,str(k),p.getResultAsDictionary())
            if lastSave + timeUntilNextSave < ( time.time() ): #in seconds
                lastSave = time.time()
                db.saveToJson(config.DATABASE_PATH+"tmp")
                os.replace(config.DATABASE_PATH+"tmp",config.DATABASE_PATH) #to prevent corrupt files.
                timeUntilNextSave = max( 60, ( time.time() - lastSave ) * 100) #the time spend on saving should grow if saving takes more time, but should be at least 60 seconds.
                lastSave = time.time()
        else:
            logger.info("Failed evaluation "+str(counter)+"/"+str(total)+" ( "+str(100*counter/total)+"%)")
            logger.info(optimizerId+" on "+problemId + " k="+str(k+1)+"/"+str(config.K))
            exception = result['exception']
            logger.error(str(type(exception).__name__)+":"+str(exception))
            logger.error(exception)
            logger.error("error occured, skipping to the next evaluation...")
    pool.close()  
    db.saveToJson(config.DATABASE_PATH)    
    logger.info("Evaluation finished")