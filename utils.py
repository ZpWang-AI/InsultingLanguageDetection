import logging
import datetime
import time
import os
import sys
import threading 
import torch
import pynvml

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


def get_cur_time(time_zone_hours=8):
    time_zone = datetime.timezone(offset=datetime.timedelta(hours=time_zone_hours))
    cur_time = datetime.datetime.now(time_zone)
    return cur_time.strftime('%Y-%m-%d_%H:%M:%S')


class MyLogger:
    def __init__(self, fold='', file='', info='', just_print=False, log_with_time=True) -> None:
        self.logger = logging.getLogger(str(datetime.datetime.now()))
        self.just_print = just_print
        self.log_with_time = log_with_time
    
        self.logger.setLevel(logging.DEBUG)
        
        if self.log_with_time:
            formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S')
        else:
            formatter = logging.Formatter('%(message)s')
            
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if not self.just_print:
            cur_time = get_cur_time().replace(':', '_')
            if not fold:
                fold = f'saved_res/{cur_time}_{info}'
            if not file:
                file = f'{cur_time}_{info}.out'
            self.log_file = path(fold)/path(file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.info(f'LOG FILE >> {self.log_file}')
            
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, *args, sep=' '):
        self.logger.info(sep.join(map(str, args)))
    
    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()
        
    def reset(self):
        self.n = 0
        self.val = 0
        self.sum = 0
        self.average = 0
    
    def add(self, val, n=1):
        self.n += n
        self.val = val
        self.sum += val
        self.average = self.sum / self.n

    def __add__(self, val):
        self.add(val)
        return self
        

class ManageGPUs:
    @staticmethod
    def query_gpu_memory(cuda_id=0, show=True, to_mb=True):
        def norm_mem(mem):
            if to_mb:
                return f'{mem/(1024**2):.0f}MB'
            unit_lst = ['B', 'KB', 'MB', 'GB', 'TB']
            for unit in unit_lst:
                if mem < 1024:
                    return f'{mem:.2f}{unit}'
                mem /= 1024
            return f'{mem:.2f}TB'
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if show:
            print(
                f'cuda: {cuda_id}, '
                f'free: {norm_mem(info.free)}, '
                f'used: {norm_mem(info.used)}, '
                f'total: {norm_mem(info.total)}'
            )
        return info.free, info.used, info.total

    @staticmethod
    def _get_all_cuda_id():
        return list(range(torch.cuda.device_count()))
        
    @staticmethod
    def _get_most_free_gpu(device_range=None):
        if not device_range:
            device_range = ManageGPUs._get_all_cuda_id()
        max_free = -1
        free_id = -1
        for cuda_id in device_range:
            cur_free = ManageGPUs.query_gpu_memory(cuda_id, show=False)[0]
            if cur_free > max_free:
                max_free = cur_free
                free_id = cuda_id
        return max_free, free_id
    
    @staticmethod
    def get_free_gpu(
        target_mem_mb=8000, 
        force=False, 
        wait=True, 
        wait_gap=5, 
        show_waiting=False,
        device_range=None, 
    ):
        if not device_range:
            device_range = ManageGPUs._get_all_cuda_id()

        if force:
            return ManageGPUs._get_most_free_gpu(device_range=device_range)[1]
        if wait:
            while 1:
                device_id = ManageGPUs.get_free_gpu(
                    target_mem_mb=target_mem_mb,
                    force=False,
                    wait=False,
                    device_range=device_range,
                )
                if device_id != -1:
                    return device_id
                if show_waiting:
                    print('waiting cuda ...')
                time.sleep(wait_gap)
        
        target_mem_mb *= 1024**2
        for cuda_id in device_range:
            if ManageGPUs.query_gpu_memory(cuda_id=cuda_id, show=False)[0] > target_mem_mb:
                return cuda_id
        return -1
        
    @staticmethod
    def _occupy_one_gpu(cuda_id, target_mem_mb=8000):
        '''
        < release by following >
        gpustat -cpu
        kill -9 <num>
        '''
        device = torch.device(f'cuda:{cuda_id}')
        used_mem = ManageGPUs.query_gpu_memory(cuda_id=cuda_id, show=False)[1]
        used_mem_mb = used_mem/(1024**2)
        one_gb = torch.zeros(224*1024**2)  # about 951mb
        gb_cnt = int((target_mem_mb-used_mem_mb)/1000)
        if gb_cnt < 0:
            return
        lst = [one_gb.detach().to(device) for _ in range(gb_cnt+1)]
        while 1:
            time.sleep(2**31)
            
    @staticmethod
    def wait_and_occupy_free_gpu(
        target_mem_mb=8000,
        wait_gap=5,
        show_waiting=False,
        device_range=None, 
    ):
        if not device_range:
            device_range = ManageGPUs._get_all_cuda_id()
        cuda_id = ManageGPUs.get_free_gpu(
            target_mem_mb=target_mem_mb,
            force=False,
            wait=True,
            wait_gap=wait_gap,
            show_waiting=show_waiting,
            device_range=device_range,
        )
        ManageGPUs._occupy_one_gpu(
            cuda_id=cuda_id,
            target_mem_mb=target_mem_mb,
        )
        
    
if __name__ == '__main__':
    # for d in get_all_files('.'):
    # for d in clock(get_all_files)(os.getcwd(), True)[:5]:  # type: ignore        
    #     print(d)
    # print(get_cur_time().replace(':', '-'))
    
    # test_logger = MyLogger(log_with_time=False, just_print=False)
    # for root, dirs, files in os.walk('./'):
    #     test_logger.info(root, dirs, files)
    
    # ManageGPUs._occupy_one_gpu(6)
    # free_cuda_id = ManageGPUs.get_free_gpu(wait=True, force=False)
    # print(free_cuda_id)
    # ManageGPUs.query_gpu_memory(free_cuda_id)
        
    # a = AverageMeter()
    # a += 10
    # print(a.val)
    pass
