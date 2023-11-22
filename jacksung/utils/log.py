import requests
from urllib.parse import quote
import _thread
import time
import threading

threadLock = threading.Lock()


def oprint(*args):
    log = '[' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ']\t' + ' '.join([str(x) for x in args])
    print(log)


def thread_send_log(url, content, name):
    threadLock.acquire()
    content = quote(content, 'utf-8')
    name = quote(name, 'utf-8')
    url = url + '&name=' + name + '&content=' + content
    # print('sendLog:' + url)
    try:
        # print("----------------sendLog...----------------")
        requests.get(url, timeout=5)
        # print('\nsendLog finish', r.status_code, r.content)
        # print('sendLog finish')
    except Exception as e:
        print('\nsendLog network error!')
    finally:
        # print("----------------sendLog...----------------")
        threadLock.release()


class LogClass:
    def __init__(self, on=False, url=None):
        self.on = on
        self.url = url

    def send_log(self, content, name):
        if self.on:
            try:
                _thread.start_new_thread(thread_send_log, (self.url, content, name))
            except Exception as e:
                print("Cloud Log Error")
