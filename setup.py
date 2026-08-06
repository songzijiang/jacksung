import json
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

# ---------------------------------------------------------------------------
# 版本管理：仅在本地 git 仓库中自增版本、清理构建产物并自动提交。
# 这样 pip 从 sdist 安装时不会误改版本号，也不会在临时目录里执行 git 提交。
# 用标准库 json 替代 tinydb，避免在 pip 隔离构建环境中因缺 tinydb 而 ImportError。
# ---------------------------------------------------------------------------
DB_FILE = 'loacaldb.json'
IS_REPO = os.path.isdir('.git')


def _load_version():
    """读取当前版本号，兼容旧 TinyDB 存储格式 {"_default": {"1": {"version": "..."}}}。"""
    version = '0.0.0.0'
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        if isinstance(data, dict):
            default = data.get('_default')
            if isinstance(default, dict):
                for doc in default.values():
                    if isinstance(doc, dict) and 'version' in doc:
                        version = doc['version']
                        break
            else:
                version = data.get('version', version)
    return version


def _bump_version(current):
    """把 0.0.4.86 递增为 0.0.4.87；末段不是数字时追加 .1。"""
    parts = str(current).split('.')
    if parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]) + 1)
    else:
        parts.append('1')
    return '.'.join(parts)


version = _load_version()

if IS_REPO:
    # 仅在本地发版时清理构建产物并递增版本号
    for stale in ('build', 'dist', 'jacksung.egg-info'):
        shutil.rmtree(stale, ignore_errors=True)
    version = _bump_version(version)
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump({'version': version}, f)

setup(
    name='jacksung',
    version=version,
    author='Zijiang Song',
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    package_data={'jacksung': ['libs/*']},
    # 精简后的真实运行时依赖（宽松下限版本）。
    # 完整的环境快照仍保留在 requirements.txt 中供开发环境使用。
    install_requires=[
        # --- 基础科学计算 ---
        'numpy>=1.26.4',
        'scipy>=1.11.3',
        'matplotlib>=3.10.7',
        'einops>=0.7.0',
        'tqdm>=4.66.1',
        # --- 深度学习 ---
        'torch>=2.1.0',  # 如需 torchvision/torchaudio，请按官方源单独安装
        'torchmetrics>=1.2.1',
        'pytorch-msssim>=1.0.0',
        # --- 遥感 / 气象 ---
        'netCDF4>=1.6.4',
        'rasterio>=1.3.9',
        'GDAL>=3.6.2',
        'cartopy>=0.22.0',
        'xarray>=2023.10.1',
        'pyresample>=1.34.2',
        'satpy>=0.59.0',
        # --- 图像 / 机器学习 ---
        'opencv-python>=4.9.0.80',
        'Pillow>=10.4.0',
        'scikit-learn>=1.3.2',
        # --- 网络 / 数据获取 ---
        'requests>=2.31.0',
        'selenium>=4.35.0',
        'openai>=1.64.0',
        'PyMySQL>=1.1.0',
        # --- 其他 ---
        'PyYAML>=6.0.1',
        'pytz>=2023.3.post1',
        'termcolor>=2.3.0',
    ],
    entry_points={
        'console_scripts': [
            'ecnu_login = jacksung.utils.login:main',
            'watch_gpu = jacksung.utils.nvidia:main'
        ]
    },
)

if IS_REPO:
    try:
        subprocess.run(["git", "commit", "-am", f"Update package {version}"], check=True)
    except subprocess.CalledProcessError as e:
        print("Git 命令执行失败:", e)
