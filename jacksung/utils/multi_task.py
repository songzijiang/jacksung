import concurrent.futures
import threading
import multiprocessing
from tqdm import tqdm

type_thread = concurrent.futures.ThreadPoolExecutor
type_process = concurrent.futures.ProcessPoolExecutor


def init(tl, pl):
    global t_lock
    global p_lock
    t_lock = tl
    p_lock = pl


class MultiTasks:
    def __init__(self, threads=10, pool=type_thread, desc="Multi Progress"):
        self.pool = pool
        self.threads = threads
        self.task_list = {}
        self.features = {}
        self.results = {}
        self.thread_mutex = threading.Lock
        self.process_mutex = multiprocessing.Lock
        self.executor = self.pool(max_workers=self.threads, initializer=init,
                                  initargs=(self.thread_mutex, self.process_mutex))
        self.desc = desc

    def add_task(self, k, function, args):
        self.task_list[k] = (function, args)

    def execute_task_nowait(self, save=False):
        for k, f_and_a in self.task_list.items():
            r = self.executor.submit(f_and_a[0], *f_and_a[1])
            if save:
                self.features[k] = r

    def execute_task(self, print_percent=True, desc=None):
        with self.pool(max_workers=self.threads, initializer=init,
                       initargs=(self.thread_mutex, self.process_mutex)) as executor:
            for k, f_and_a in self.task_list.items():
                self.features[k] = executor.submit(f_and_a[0], *f_and_a[1])
            if print_percent:
                progress_bar = tqdm(total=len(self.features), desc=desc if desc else self.desc)
            for k, feature in self.features.items():
                self.results[k] = feature.result()
                if print_percent:
                    progress_bar.update(1)
        return self.results
