from setuptools import setup, find_packages

setup(
    name='jacksung',
    version='0.0.2.145',
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
        'openai'
    ],
    entry_points={
        'console_scripts': [
            'ecnu_login = jacksung.utils.login:main',
            'watch_gpu = jacksung.utils.nvidia:main'
        ]
    }
)
