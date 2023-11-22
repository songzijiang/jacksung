import os
import time
import pymysql
from datetime import datetime, timedelta
from tqdm import tqdm

import threading
import multiprocessing
import traceback, sys

import configparser


def convert_str(s):
    return "'" + s.replace("'", '"') + "'"


def convert_num(n):
    return str(n)


class BaseDB:
    def __init__(self, ini_path='db.ini'):
        self.conn = None
        self.db = None
        self.passwd = None
        self.user = None
        self.host = None
        self.autocommit = True
        self.ini_path = ini_path
        pymysql.install_as_MySQLdb()
        self.lock_t = threading.Lock()

    def read_config(self):
        config = configparser.ConfigParser()
        config.read(self.ini_path)
        # Create the connection object
        self.host = config['database']['host']
        self.user = config['database']['user']
        self.passwd = config['database']['password']
        self.db = config['database']['database']

    def reconnect(self):
        if not self.conn:
            self.read_config()
        # 打开数据库连接
        connection = pymysql.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db,
                                     autocommit=self.autocommit)
        tqdm.write('Reconnected!')
        return connection

    def execute(self, sql):
        self.lock_t.acquire()
        cursor = None
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(sql)
        except Exception as e:
            self.conn = self.reconnect()
            cursor = self.conn.cursor()
            result = cursor.execute(sql)
        finally:
            if cursor:
                cursor.close()
        self.lock_t.release()
        return result, cursor
