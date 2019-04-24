#system
import logging

def worker(x):
    try:
        optimizer = x['optimizer']
        problem = x['problem']
        maxeval = x['maxeval']
        x['result'] = optimizer().apply( problem(), maxeval )
    except Exception as exceptionInstance:
        x['exception'] = exceptionInstance
    return x