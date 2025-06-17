import os
import platform
# 从selenium导入webdriver
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from jacksung.utils.log import oprint as print
from selenium.webdriver.common.by import By


def make_driver(url, is_headless=False, tmp_path=None, download_dir=None, options=webdriver.ChromeOptions()):
    if tmp_path:
        options.add_argument("crash-dumps-dir=" + tmp_path)
    options.add_argument("--no-sandbox")
    # options.add_argument("--auto-open-devtools-for-tabs")
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--disable-web-security")  # 禁用Web安全
    options.add_argument("--allow-running-insecure-content")  # 允许不安全的内容
    options.add_argument('--user-agent=Mozilla/5.0')
    options.add_argument('--ignore-ssl-errors=yes')
    options.add_argument('--allow-insecure-localhost')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument("--lang=zh-CN")  # 将语言设置为简体中文，英文为en-US
    options.add_experimental_option("detach", True)
    if download_dir:
        options.add_experimental_option("prefs", {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "safebrowsing.enabled": True,  # 允许“不安全”文件自动下载
            "safebrowsing.disable_download_protection": True  # 禁用“可能有害文件”的拦截
            # "download.directory_upgrade": True,
        })
    options.set_capability('pageLoadStrategy', 'none')
    options.set_capability("unhandledPromptBehavior", "accept")
    if is_headless:
        options.add_argument('--headless')  # 浏览器隐式启动
    # driver_path = os.path.expanduser("~/chrome/chromedriver.exe")
    print('driver is going to init!')
    if platform.system().lower() == 'windows':
        driver_path = None
        # driver_path = os.path.expanduser("~/chrome/chromedriver.exe")
    else:
        driver_path = os.path.expanduser("~/chrome/chromedriver")
    driver = webdriver.Chrome(service=Service(driver_path) if driver_path else None, options=options)
    # driver.maximize_window()
    driver.implicitly_wait(10)
    driver.set_page_load_timeout(10)
    print(f'请求地址：{url}')
    driver.get(url)
    return driver
