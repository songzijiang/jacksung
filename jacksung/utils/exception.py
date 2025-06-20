import time
from jacksung.utils.log import oprint as print


class NoFileException(Exception):
    def __init__(self, file_name):
        self.file_name = file_name
        super().__init__(f'No such file: {file_name}')


class NanNPException(Exception):
    def __init__(self, file_name):
        self.file_name = file_name
        super().__init__(f'Nan value in np data: {file_name}')


def wait_fun(fun, args, catch_exception=Exception, sleep_time=0.5, wait_time=5, open_log=True):
    try:
        return fun(*args)
    except catch_exception as e:
        if open_log:
            print(f'Task {args} failed, retry in {sleep_time}s, remain waiting time: {wait_time}s')
        if wait_time <= 0:
            raise e
        else:
            time.sleep(sleep_time)
            return wait_fun(fun, args, catch_exception, sleep_time=sleep_time, wait_time=wait_time - sleep_time)
