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


def init(tl, pl):
    global t_lock
    global p_lock
    t_lock = tl
    p_lock = pl


class MultiTasks:
    def __init__(self, threads=10, pool=type_thread, desc="Mult. Pro."):
        if pool == type_thread:
            self.pool = concurrent.futures.ThreadPoolExecutor
        elif pool == type_process:
            self.pool = concurrent.futures.ProcessPoolExecutor
        self.threads = threads
        self.task_list = {}
        self.features = {}
        self.results = {}
        self.thread_mutex = threading.Lock()
        self.process_mutex = multiprocessing.Lock()
        self.executor = self.pool(max_workers=self.threads, initializer=init,
                                  initargs=(self.thread_mutex, self.process_mutex))
        self.desc = desc
        self.progress_bar = None

    def add_task(self, k, function, args):
        self.task_list[k] = (function, args)

    def execute_task_nowait(self, save=False):
        for k, f_and_a in self.task_list.items():
            r = self.executor.submit(f_and_a[0], *f_and_a[1])
            if save:
                self.features[k] = r

    def wrap_fun(self, fun, args):
        result = fun(*args)
        if self.progress_bar:
            self.progress_bar.update(1)
        return result

    def execute_task(self, print_percent=True, desc=None):
        with self.pool(max_workers=self.threads, initializer=init,
                       initargs=(self.thread_mutex, self.process_mutex)) as executor:
            if print_percent:
                self.progress_bar = tqdm(total=len(self.task_list.items()), desc=desc if desc else self.desc)
            else:
                self.progress_bar = None
            for k, f_and_a in self.task_list.items():
                self.features[k] = executor.submit(self.wrap_fun, *(f_and_a[0], f_and_a[1]))
            for k, feature in self.features.items():
                self.results[k] = feature.result()
        return self.results


def worker(i):
    print(i)
    time.sleep(1)


if __name__ == '__main__':
    mt = MultiTasks(10)
    for i in range(100):
        mt.add_task(i, worker, [i])
    mt.execute_task(print_percent=False)
