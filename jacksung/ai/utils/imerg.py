from jacksung.utils.data_convert import nc2np, np2tif
import numpy as np
import netCDF4 as nc
from einops import rearrange
import os
import shutil
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import platform


class Downloader:
    def __init__(self, download_file_path, save_path=None):
        self.download_file_path = download_file_path
        if save_path is not None:
            self.save_path = save_path
        else:
            if platform.system().lower() == 'windows':
                self.save_path = 'D:\\imerg'
            else:
                self.save_path = '/mnt/data1/szj/imerg'

    def make_driver(self, url, is_headless=False, tmp_path=None, download_dir=None):
        options = webdriver.ChromeOptions()
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
                # "download.prompt_for_download": False,
                # "download.directory_upgrade": True,
                # "safebrowsing.enabled": True
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
        print('driver is inited!')
        print(f'请求地址：{url}')
        driver.get(url)
        return driver

    # 进度条模块
    def progressbar(self, url, path):
        if not os.path.exists(path):  # 看是否有该文件夹，没有则创建文件夹
            os.mkdir(path)
        start = time.time()  # 下载开始时间
        response = requests.get(url, stream=True)
        # https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2023/001/3B-HHR.MS.MRG.3IMERG.20230101-S000000-E002959.0000.V07B.HDF5
        name = url. \
            replace(
            'https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2023/001/3B-HHR.MS.MRG.3IMERG.',
            '').replace('.0000.V07B.HDF5', '.HDF5')
        size = 0  # 初始化已下载大小
        chunk_size = 1024  # 每次下载的数据大小
        content_size = int(response.headers['content-length'])  # 下载文件总大小
        if response.status_code == 200:  # 判断是否响应成功
            print('Start download,[File size]:{size:.2f} MB'.format(
                size=content_size / chunk_size / 1024))  # 开始下载，显示下载文件大小
            filepath = rf'{path}\{name}'  # 设置图片name，注：必须加上扩展名
            with open(filepath, 'wb') as file:  # 显示进度条
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    size += len(data)
                    print('\r' + '[下载进度]:%s%.2f%%' % (
                        '>' * int(size * 50 / content_size), float(size / content_size * 100)), end=' ')
        end = time.time()  # 下载结束时间
        print('Download completed!,times: %.2f秒' % (end - start))  # 输出下载用时时间

    def simulate(self, url, driver, path):
        start = time.time()  # 下载开始时间
        driver.get(url)
        while not os.path.exists(path):
            time.sleep(1)
        # 3B-HHR-E.MS.MRG.3IMERG.20230101-S033000-E035959.0210.V07B.HDF5
        names = path.split('/')[-1].split('.')
        date = names[4].split('-')[0]
        move_path = path.replace('.'.join(names[:4]), f'{date}{os.path.sep}{".".join(names[:4])}')
        if not os.path.exists(os.path.dirname(move_path)):
            os.makedirs(os.path.dirname(move_path))
        shutil.move(path, move_path)
        end = time.time()  # 下载结束时间
        print('Download completed!,times: %.2f秒' % (end - start))  # 输出下载用时时间

    def start_download(self):
        download_file_path = self.download_file_path
        print(f'开始下载:{download_file_path}')
        f = open(download_file_path, 'r')
        save_f = open('downloaded.txt', 'r')

        downloaded_list = save_f.readlines()
        downloaded_list = [u.replace('\n', '') for u in downloaded_list if u.count('.pdf') == 0]
        save_f.close()
        with open('downloaded.txt', 'a') as save_f_w:
            driver = self.make_driver('https://urs.earthdata.nasa.gov', download_dir=self.save_path, is_headless=True)
            time.sleep(10)
            username = driver.find_element(By.ID, 'username')
            passwd = driver.find_element(By.ID, 'password')
            username.send_keys('chesser')
            passwd.send_keys('az_ePR4,5L.gq/B')
            driver.find_element(By.NAME, 'commit').click()
            time.sleep(5)
            for line in f.readlines():
                url = line.strip()
                if line.replace('\n', '') not in downloaded_list:
                    print(url)
                    file_path = self.save_path + os.sep + url.split('/')[-1]
                    try:
                        self.simulate(url, driver, file_path)
                        save_f_w.write(url + '\n')
                    except Exception as e:
                        print(f'{url} download failed')
                else:
                    print(f'{url} already downloaded')
            driver.close()


def getNPfromHDF(hdf_path, lock=None, save_file=True):
    if lock:
        lock.acquire()
    ds = nc.Dataset(hdf_path)
    if lock:
        lock.release()
    np_data = np.array(ds.groups['Grid']['precipitation'][:]).astype(np.float32)
    np_data = np_data[np.newaxis, :]
    ds.close()
    np_data = rearrange(np_data[0][0], 'w h->h w')[::-1, :]
    np_data[np_data < 0] = 0
    if save_file:
        np2tif(np_data, save_path='np2tif_dir', left=-180, top=90, x_res=0.1, y_res=0.1, out_name='IMERG',
               dtype='float32')
    return np_data
