# Jacksung's utils
Python version is required above 3.9.

Recommend version: python 3.11.

Create env by conda: 
```conda create -n jacksung python=3.11```

## Installation
```pip install jacksung```
## Login ecnu
1 [download chromedriver](https://googlechromelabs.github.io/chrome-for-testing/)
    windows:chromedriver.exe
    linux:chromedriver

p.s. chrome should be installed first, 
[see how to install Chrome](https://www.google.com/chrome/)

2 make the directory 'chrome' in the home path

3 put the driver file, and make a new directory 'tmp' in the 'chrome' directory. The structure of directory:
```
--home
--|--chrome
--|--|--chromedriver.exe
--|--|--tmp
```

4 run cmd
```ecnu_login -u 账号 -p 密码```

Or using python code:
```
from jacksung.utils.login import ecnu_login

login = ecnu_login(driver_path='chromedriver_path', tmp_path='tmp_path')
login.get_drive()
# Check net status. If have not logged in, will log in.
login.login_check('username', 'password')
login.login('username', 'password')
login.logout()
```

## Log
time format log
```
from jacksung.utils.log import oprint as print

print('this is a log')
```

```
# the URL should accept parameters name and content, e.g. https://www.example.com?log&api-key=123&name=logname&content=logcontent
log_class = LogClass(url='https://www.example.com?log&api-key=123')
log_class.send_log('35.8', 'PSNR')
log_class.send_log('35.8', 'PSNR')
time.sleep(10)
```

## Multi threadings
```
from jacksung.utils.multi_task import MultiTasks
import time

def worker(idx):
    print(idx)
    time.sleep(2)
    return idx

mt = MultiTasks(3)
for idx in range(10):
    mt.add_task(idx, worker, (idx))
results = mt.execute_task()
```
## Connecting to mysql
```
from jacksung.utils.base_db import BaseDB, convert_str
class DB:
    def __init__(self, ini_path='../db.ini'):
        self.bd = BaseDB(ini_path)
    # List an example of MySQL table, change the sql code to your own. 
    def insert_record(self, year, month, day):
        sql = rf"INSERT INTO `data_record` (`year`,`month`,`day`) VALUES ({year},{month},{day});"
        self.bd.execute(sql)

    def select_record(self, year, month, day):
        sql = rf"select count(1) from data_record where year={year} and month={month} and day={day};"
        result, cursor = self.bd.execute(sql)
        return cursor.fetchone()[0]
```
db.ini is the form  as follows:
```
[database]
host = 127.0.0.1
user = root
password = root
database = XXXX
```
## Show Nvdia information
```watch_gpu```
or set the command line to the bash file
```alias watch-gpu='watch -n 1 -d watch_gpu'```

## Time calculating
### RemainTime
```
from jacksung.utils.time import RemainTime
import time

epochs=100
rt = RemainTime(epochs)
for i in range(epochs):
    rt.update()
    time.sleep(2)
```
### StopWatch:
pinch(): return the seconds from the last stopwatch reset.

reset(): return the seconds from the last stopwatch reset and reset the stopwatch.
```
from jacksung.utils.time import Stopwatch
import time

sw = Stopwatch()
time.sleep(1)
print(sw.pinch())
time.sleep(1)
print(sw.reset())
time.sleep(1)
print(sw.pinch())
```


## Convert utils
convert .nc to numpy, convert numpy to tif (with or without geocoordinate)

```
from jacksung.utils.data_convert import nc2np, np2tif
import numpy as np

nc_t = nc2np(r'C:\Users\ECNU\Desktop\upper.nc')
# nptype data and the path aims to the .npy file are allowed in the np2tif.
# without geocoordinate
np2tif(nc_t, 'constant_masks/upper')
# with geocoordinate
np2tif('constant_masks/land_mask.npy', save_path='constant_masks', out_name='land_mask', left=0, top=90, x_res=0.25,
       y_res=0.25, dtype=np.float32)
```

## Note
#### Commit new dependence
Please refer to how to upload a dependence
```
python setup.py sdist bdist_wheel
twine upload dist/*
```
The repo is built by jacksung, contact me by jacksung1995@gmail.com
