# 屏蔽无害的第三方库警告（必须在 import cv2 之前设置环境变量）
import os
import warnings

# cv2 5 读取 GeoTIFF 时对未知地理标签的 WARN 日志直出 stderr，不走日志级别通道，
# 只能在 import cv2 前用环境变量屏蔽
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')

try:
    from rasterio.errors import NotGeoreferencedWarning
    # 写无地理参考的临时图（如 fy.py 的中间 tif）时的提示，无碍
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
except Exception:
    pass

from jacksung import *
