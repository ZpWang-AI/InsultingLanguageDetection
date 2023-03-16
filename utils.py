import datetime
import time
import os
import sys
import threading

from typing import *
from pathlib import Path as path


# sys.path.append(os.getcwd())
sys.setrecursionlimit(10**8)


# a timer to print running time of the target function
# can be used as @clock or @clock()
def clock(func=None, start_info='', end_info='', sym='---'):
    if func:
        def new_func(*args, **kwargs):
            if start_info:
                print('  '+start_info)
            print(f'{sym} {func.__name__} starts')
            start_time = time.time()
            res = func(*args, **kwargs)
            running_time = time.time()-start_time
            if running_time > 60:
                running_time = datetime.timedelta(seconds=int(running_time))
            else:
                running_time = '%.2f s' % running_time
            print(f'{sym} {func.__name__} ends, running time: {running_time}')
            if end_info:
                print('  '+end_info)
            return res
        return new_func
    else:
        return lambda func: clock(func, start_info, end_info, sym)
        

# run function asynchronously
# can be used as @async_run or @async_run()
def async_run(func=None):
    if func:
        def new_func(*args, **kwargs):
            new_thread = threading.Thread(target=func, args=args, kwargs=kwargs)
            new_thread.start()
        return new_func
    else:
        return async_run
    
    
def get_all_files(root_fold:Union[path, str], recursion:bool=False):
    root_fold = path(root_fold)
    son_files = []
    for nxt in os.listdir(root_fold):
        nxt = root_fold/nxt
        if nxt.is_dir():
            if recursion:
                son_files.extend(get_all_files(nxt, True))
        else:
            son_files.append(nxt)
    return son_files


def try_remove(root_fold:Union[path, str], recursion:bool=False):
    root_fold = path(root_fold)
    if not root_fold.exists():
        return
    if recursion:
        for nxt in get_all_files(root_fold):
            try_remove(nxt, True)
    os.remove(root_fold)
    
    
if __name__ == '__main__':
    # for d in get_all_files('.'):
    for d in clock(get_all_files)(os.getcwd(), True)[:5]:  # type: ignore        
        print(d)
