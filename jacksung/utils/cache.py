import threading
import time


class Cache:
    def __init__(self, cache_len=120):
        self.cache_L = threading.Lock()
        self.cache = {}
        self.cache_list = []
        self.cache_len = cache_len

    # 判断key是否在缓存中，在则返回key对应的值，否则返回False, 同时锁住该key的所有查询，直到被放入数据
    def get_key_in_cache(self, key):
        try:
            while True:
                self.cache_L.acquire()
                if key in self.cache_list:
                    if self.cache[key].is_ok:
                        result = self.cache[key].value
                        self.cache_L.release()
                        return result
                else:
                    self.cache_L.release()
                    break
                self.cache_L.release()
                time.sleep(0.5)
            self.cache_L.acquire()
            self.__add_key(key)
            self.cache_L.release()
            return False
        except Exception as e:
            self.cache_L.release()
            raise e

    def add_key(self, key, value):
        try:
            self.cache_L.acquire()
            if key not in self.cache_list:
                self.__add_key(key)
            self.cache[key].set_value(value)
            self.cache_L.release()
        except Exception as e:
            self.cache_L.release()
            raise e

    def __add_key(self, key):
        self.cache_list.append(key)
        self.cache[key] = self.CacheClass(key)
        if len(self.cache_list) > self.cache_len:
            del_key = self.cache_list.pop(0)
            del self.cache[del_key]

    class CacheClass:
        def __init__(self, key):
            self.is_ok = False
            self.key = key
            self.value = None

        def set_value(self, value):
            self.value = value
            self.is_ok = True
