import requests
from urllib.parse import quote
import _thread
import time
import threading
import sys

threadLock = threading.Lock()


def format_log(*args):
    return '[' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ']\t' + ' '.join([str(x) for x in args])


def oprint(*args):
    log = format_log(*args)
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


class StdLog(object):
    def __init__(self, filename='default.log', common_path='warning_log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
        self.common_log = None
        self.common_path = common_path

    def write(self, message):
        message = str(message)
        if message.count('[TemporaryTag]') == 0:
            if message.count('[Common]') != 0 or message.count('[Warning]') != 0 \
                    or message.count('[Error]') != 0 or message.count('[OnlyFile]') != 0:
                if self.common_log is None:
                    self.common_log = open(self.common_path, 'a')
                self.common_log.write(message.replace('[Common]', '').replace('[OnlyFile]', ''))
                self.common_log.flush()
            else:
                self.log.write(message)
                self.log.flush()
        else:
            message = message.replace('[TemporaryTag]', '')
        if message.count('[OnlyFile]') == 0:
            self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        if self.common_log is not None:
            self.common_log.flush()


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
