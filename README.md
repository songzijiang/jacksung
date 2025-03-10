# Jacksung's utils
Python version is required above 3.9.

Recommend version: python 3.11.

Create env by conda: 
```conda create -n jacksung python=3.11```

## Installation
```pip install jacksung```
## Login ecnu
1 [download chromedriver](https://googlechromelabs.github.io/chrome-for-testing/) on windows is chromedriver.exe and on linux:chromedriver
    
*p.s. chrome should be installed first, [see how to install Chrome](https://www.google.com/chrome/)*

2 make the directory 'chrome' in the home path

3 put the driver file, and make a new directory 'tmp' in the 'chrome' directory (i.e., ~/chrome/chromedriver.exe and ~/chrome/tmp). The structure of directory:
```
--Home directory
--|--chrome
--|--|--chromedriver.exe
--|--|--tmp
```

4 How to run?
- run cmd
```ecnu_login -u 账号 -p 密码```
- Or
set the username and password in ~/.ecnu_login
```
u: 账户
p: 密码
```
and then run cmd ```ecnu_login```
- Or using python code:
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
### time format log
```
from jacksung.utils.log import oprint as print

print('this is a log')
```
### Send log message to server
```
# the URL should accept parameters name and content, e.g. https://www.example.com?log&api-key=123&name=logname&content=logcontent
log_class = LogClass(url='https://www.example.com?log&api-key=123')
log_class.send_log('35.8', 'PSNR')
log_class.send_log('35.8', 'PSNR')
time.sleep(10)
```
### record the terminal log to file

```
from jacksung.utils.log import StdLog
import sys

# please put the following code in the '__main__' function
if __name__ == '__main__':
    sys.stdout = StdLog(filename='log.txt', common_path='warning.txt')
    print(f'[TemporaryTag]Only in terminal', end='[TemporaryTag]\n')
    print(f'[Warning]In warning.txt and terminal', end='[Warning]\n')
    print(f'[Error]In warning.txt and terminal', end='[Error]\n')
    print(f'[Common]Common in warning.txt and terminal', end='[Common]\n')
    print(f'[OnlyFile]OnlyFile in warning.txt and terminal', end='[OnlyFile]\n')
    print(f'In log.txt and terminal')
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
## Fast numpy
```
import jacksung.utils.fastnumpy as fnp

# fast than numpy.load('xx.npy')
fnp.load('xx.npy')
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
    
nc_t, dim = nc2np(r'C:\Users\ECNU\Desktop\upper.nc')
# nptype data and the path aims to the .npy file are allowed in the np2tif.
# dim_value examples:
# 3 dimension example
# dim_value=[{'value': ['WIN', 'TMP', 'PRS', 'PRE']}]
# 4 dimension examples
# dim_value=[{'value': ['WIN', 'TMP']},{'value': ['PRSS', 'HEIGHT']}]
# without geocoordinate
np2tif(nc_t, 'constant_masks/upper', dim_value=dim)
# with geocoordinate
np2tif('constant_masks/land_mask.npy', save_path='constant_masks', out_name='land_mask', left=0, top=90, x_res=0.25,
       y_res=0.25, dtype=np.float32, dim_value=dim)
```
## AI tools
### latex auto polish
auto polish latex using LLM.
```
from jacksung.ai.latex_tool import polish
# e.g.
# if your main.tex located in '/mnt/paper1/main.tex'
# main_dir_path is '/mnt/paper1' and tex_file is 'main.tex'
# If your paper is in Chinese or other language, use cn_prompt=True
# If you want to use custom prompt, you can use prompt with '{text}'.
# You can define the skip or rewrite part using skip_part_list and rewrite_list. If you don`t know how to set, the default setting is recommended.
polish(main_dir_path='your latex root directory', tex_file='your main tex path consider from main_dir_path', server_url='The full LLM server url with v1',
       token='Your token here',
       )
```
- After running, three .tex file: "old.tex","new.tex","diff.tex" in the parnent directory will generated. The file change track PDF will compiled by diff.tex.
- If there are errors in compiling diff.tex, you need to revise the produced new.tex to fix the bug first.
- The LLM is recommend at least Deepseek-R1-70b and bigger, smaller model will produce more bugs lead to errors.
## Note
#### Commit new dependence
Please refer to how to upload a dependence
```
python setup.py sdist bdist_wheel
twine upload dist/*
```
The repo is built by jacksung, contact me by jacksung1995@gmail.com
