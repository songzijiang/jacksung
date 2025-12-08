import shutil

from setuptools import setup, find_packages
from tinydb import TinyDB, Query
import subprocess

shutil.rmtree('build', ignore_errors=True)
shutil.rmtree('dist', ignore_errors=True)
shutil.rmtree('jacksung.egg-info', ignore_errors=True)
db = TinyDB('loacaldb.json')
infos = db.all()
if len(infos) == 0:
    version = '0.0.0.0'
    db.insert({'version': version})
else:
    version = db.all()[0]['version']
version = '.'.join(version.split('.')[:-1]) + '.' + str(int(version.split('.')[-1]) + 1)
db.update({'version': version})
db.close()
setup(
    name='jacksung',
    version=version,
    author='Zijiang Song',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tqdm',
        'requests',
        'pymysql',
        'pytz',
        'selenium',
        'termcolor',
        'einops',
        'rasterio',
        'netCDF4',
        'pyyaml',
        'opencv-python',
        'Pillow',
        'openai',
        'satpy',
        'pyresample'
    ],
    entry_points={
        'console_scripts': [
            'ecnu_login = jacksung.utils.login:main',
            'watch_gpu = jacksung.utils.nvidia:main'
        ]
    }
)
try:
    subprocess.run(["git", "commit", "-am", rf"Update package {version}"], check=True)
except subprocess.CalledProcessError as e:
    print("Git 命令执行失败:", e)
