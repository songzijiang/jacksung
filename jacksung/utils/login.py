import time
import os
import requests
import subprocess
# 从selenium导入webdriver
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import platform
from selenium.common.exceptions import NoSuchElementException
import traceback, sys
import argparse


class ecnu_login:
    def __init__(self, driver_path=None, tmp_path=None):
        options = webdriver.ChromeOptions()

        options.add_argument("--no-sandbox")
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--user-agent=Mozilla/5.0 HAHA')
        options.add_experimental_option("detach", True)
        options.add_argument('--headless')  # 浏览器隐式启动
        self.url = "https://login.ecnu.edu.cn/"
        self.driver = None
        if tmp_path:
            options.add_argument("crash-dumps-dir=" + tmp_path)
        self.options = options
        self.driver_path = driver_path

    def get_drive(self):
        print('driver is going to init!')
        self.driver = webdriver.Chrome(service=Service(self.driver_path) if self.driver_path else None,
                                       options=self.options)
        print('driver is inited!')
        self.driver.get(self.url)
        time.sleep(0.5)
        self.driver.refresh()
        time.sleep(2)

    #     todo:准备打包
    def refresh_drive(self):
        self.driver.get(self.url)
        self.driver.refresh()
        time.sleep(3)

    def close_driver(self):
        self.driver.close()
        self.driver.quit()

    def login_check(self, username, password):
        print('checking net......')
        if self.print_ip():
            return True
        else:
            return self.login(username, password)

    def print_ip(self):
        try:
            time.sleep(4)
            ipv4 = self.driver.find_element(By.ID, 'ipv4')
            if ipv4:
                print('当前IP：', ipv4.text)
                return True
        except Exception as e:
            return False

    def login(self, username, password):
        if self.print_ip():
            print('目前已登录！')
            return True
        driver = self.driver
        try:
            driver.refresh()
            time.sleep(2)
            username_ele = driver.find_element(By.ID, "username")
            password_ele = driver.find_element(By.ID, "password")
            username_ele.send_keys(username)
            password_ele.send_keys(password)

            driver.find_element(By.ID, 'login-account').click()
            time.sleep(2)
            try:
                ipv4 = driver.find_element(By.ID, 'ipv4')
                print('登录成功')
                print('当前IP：', ipv4.text)
                return True
            except NoSuchElementException as e:
                print(driver.find_element(By.XPATH,
                                          "/html/body/div[2]/div[@class='component dialog confirm active']/div[@class='content']/div[@class='section']")
                      .text)
                return False
        except Exception as e:
            print('登录失败！')
            traceback.print_exc()
            return False

    def logout(self):
        driver = self.driver
        try:
            driver.find_element(By.ID, 'logout').click()
            time.sleep(3)
            driver.find_element(By.CLASS_NAME, 'btn-confirm').click()
            print('登出成功')
            return True
        except Exception as e:
            print('登出失败！')
            traceback.print_exc()
            return False


def main():
    if platform.system().lower() == 'windows':
        driver_path = os.path.expanduser("~/chrome/chromedriver.exe")
    else:
        driver_path = os.path.expanduser("~/chrome/chromedriver")

    login = ecnu_login(driver_path=driver_path, tmp_path=os.path.expanduser("~/chrome/tmp"))
    login.get_drive()
    parser = argparse.ArgumentParser(
        prog='login',  # 程序名
        description='login or logout',  # 描述
        epilog='Copyright(r), 2023'  # 说明信息
    )
    parser.add_argument('-t', default='login_check')
    parser.add_argument('-u', default='')
    parser.add_argument('-p', default='')
    args = parser.parse_args()
    if args.t == 'login_check':
        login.login_check(args.u, args.p)
    elif args.t == 'login':
        login.login(args.u, args.p)
    elif args.t == 'logout':
        login.logout()