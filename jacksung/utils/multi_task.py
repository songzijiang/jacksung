import concurrent.futures
import threading
import multiprocessing
import time
from tqdm import tqdm

type_thread = 'type_thread'
type_process = 'type_process'


class ThreadingLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.owner = None

    def acquire(self, blocking=True):
        acquired = self.lock.acquire(blocking)
        if acquired:
            self.owner = threading.current_thread().name
        return acquired

    def release(self):
        self.lock.release()
        self.owner = None

    def get_owner(self):
        return self.owner


def init_process(pl):
    """进程池初始化函数，只设置进程锁"""
    global p_lock
    p_lock = pl


def init_thread(tl):
    """线程池初始化函数，只设置线程锁"""
    global t_lock
    t_lock = tl


class MultiTasks:
    def __init__(self, threads=10, pool=type_thread, desc="Mult. Pro."):
        self.threads = threads
        self.task_list = {}
        self.features = {}
        self.results = {}
        self.submitted = []
        self.pool_type = pool

        if pool == type_thread:
            # 线程池不需要在初始化时传递锁
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.threads)
        elif pool == type_process:
            # 进程池需要特殊处理锁
            self.process_mutex = multiprocessing.Manager().Lock()
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.threads,
                initializer=init_process,
                initargs=(self.process_mutex,)
            )
        self.desc = desc
        self.progress_bar = None

    def add_task(self, k, function, args):
        self.task_list[k] = (function, args)

    def execute_task_nowait(self, save=False, print_log=False):
        if print_log:
            print('Now running tasks:', rf'{len(self.submitted)}/{len(self.task_list)}')
        for k, f_and_a in self.task_list.items():
            if k not in self.submitted:
                r = self.executor.submit(f_and_a[0], *f_and_a[1])
                if save:
                    self.features[k] = r
                self.submitted.append(k)

    def wrap_fun(self, fun, args):
        """包装函数，处理进度条更新"""
        result = fun(*args)
        if self.progress_bar:
            # 使用适当的方式更新进度条
            if self.pool_type == type_thread:
                # 线程环境下直接更新
                self.progress_bar.update(1)
            else:
                # 进程环境下需要特殊处理，这里简化处理
                raise Exception('进程环境下需要特殊处理，这里简化处理.')
        return result

    def execute_task(self, print_percent=True, desc=None):
        if print_percent:
            self.progress_bar = tqdm(total=len(self.task_list), desc=desc if desc else self.desc)

        # 根据池类型选择不同的执行策略
        if self.pool_type == type_thread:
            # 线程池直接执行
            for k, f_and_a in self.task_list.items():
                self.features[k] = self.executor.submit(self.wrap_fun, f_and_a[0], f_and_a[1])
        else:
            # 进程池需要避免传递不可序列化对象
            for k, f_and_a in self.task_list.items():
                # 直接提交函数，不包装进度条更新
                self.features[k] = self.executor.submit(f_and_a[0], *f_and_a[1])

        # 收集结果并更新进度条
        for i, (k, feature) in enumerate(self.features.items()):
            self.results[k] = feature.result()
            if self.progress_bar and self.pool_type == type_process:
                # 进程环境下在主线程更新进度条
                self.progress_bar.update(1)

        if self.progress_bar:
            self.progress_bar.close()

        return self.results


def worker(i):
    # print(i)
    time.sleep(0.1)
    return i * 2


if __name__ == '__main__':
    mt = MultiTasks(10, pool=type_thread)
    for i in range(100):
        mt.add_task(i, worker, [i])
    mt.execute_task()
