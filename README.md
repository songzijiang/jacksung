# Jacksung

`jacksung` is a personal Python utility package for research and engineering workflows. It includes helpers for ECNU network login, logging, multithreading, MySQL access, NumPy/GeoTIFF/NetCDF conversion, image processing, NVIDIA GPU monitoring, LaTeX polishing with LLMs, and several AI/meteorological data utilities.

Python 3.9 or later is required. Python 3.11 is recommended.

## Installation

Install from PyPI:

```bash
pip install jacksung
```

Create a clean conda environment first if needed:

```bash
conda create -n jacksung python=3.11
conda activate jacksung
pip install jacksung
```

For local development:

```bash
pip install -r requirements.txt
```

## Package Layout

```text
jacksung/
  utils/        General utilities: login, log, database, time, image, conversion, GPU monitor
  ai/           AI helpers: metrics, LaTeX polishing, GeoNet/GeoAttX, satellite data utilities
```

Main command line tools:

- `ecnu_login`: log in, check, or log out of the ECNU campus network.
- `watch_gpu`: print NVIDIA GPU status using `nvidia-smi`.

## ECNU Login

The login tool uses Selenium and ChromeDriver. Install Chrome first, then download the matching ChromeDriver from [Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/).

Expected driver layout:

```text
Home directory
`-- chrome
    |-- chromedriver.exe   # Windows
    |-- chromedriver       # Linux/macOS
    `-- tmp
```

On Windows the default path is `~/chrome/chromedriver.exe`; on other systems it is `~/chrome/chromedriver`.

Run with credentials:

```bash
ecnu_login -u account -p password
```

Or store credentials in `~/.ecnu_login`:

```yaml
u: account
p: password
```

Then run:

```bash
ecnu_login
```

Supported actions:

```bash
ecnu_login -t login_check
ecnu_login -t login
ecnu_login -t logout
```

Python usage:

```python
from jacksung.utils.login import ecnu_login

login = ecnu_login(driver_path="chromedriver_path", tmp_path="tmp_path")
login.get_drive()
login.login_check("username", "password")
login.login("username", "password")
login.logout()
login.close_driver()
```

## Logging

Print messages with timestamps:

```python
from jacksung.utils.log import oprint as print

print("this is a log")
```

Send log messages to a server. The URL should accept `name` and `content` parameters, for example `https://www.example.com?log&api-key=123&name=logname&content=logcontent`.

```python
from jacksung.utils.log import LogClass

log_class = LogClass(on=True, url="https://www.example.com?log&api-key=123")
log_class.send_log("35.8", "PSNR")
```

Record terminal output to files:

```python
import sys
from jacksung.utils.log import StdLog

if __name__ == "__main__":
    sys.stdout = StdLog(filename="log.txt", common_path="warning.txt")
    print("[TemporaryTag]Only in terminal", end="[TemporaryTag]\n")
    print("[Warning]In warning.txt and terminal", end="[Warning]\n")
    print("[Error]In warning.txt and terminal", end="[Error]\n")
    print("[Common]Common in warning.txt and terminal", end="[Common]\n")
    print("[OnlyFile]OnlyFile in warning.txt and terminal", end="[OnlyFile]\n")
    print("In log.txt and terminal")
```

## Multithreading

```python
import time
from jacksung.utils.multi_task import MultiTasks

def worker(idx):
    print(idx)
    time.sleep(2)
    return idx

mt = MultiTasks(threads=3)
for idx in range(10):
    mt.add_task(idx, worker, [idx])

results = mt.execute_task()
```

`MultiTasks` uses a thread pool by default. Process mode is also available through `pool=type_process` from `jacksung.utils.multi_task`.

## Fast NumPy

```python
import jacksung.utils.fastnumpy as fnp

data = fnp.load("data.npy")
fnp.save("copy.npy", data)
```

## MySQL

`BaseDB` reads connection settings from an ini file and executes SQL through PyMySQL.

```python
from jacksung.utils.base_db import BaseDB

class DB:
    def __init__(self, ini_path="db.ini"):
        self.bd = BaseDB(ini_path)

    def insert_record(self, year, month, day):
        sql = f"INSERT INTO `data_record` (`year`, `month`, `day`) VALUES ({year}, {month}, {day});"
        self.bd.execute(sql)

    def select_record(self, year, month, day):
        sql = f"SELECT COUNT(1) FROM data_record WHERE year={year} AND month={month} AND day={day};"
        result, cursor = self.bd.execute(sql)
        return cursor.fetchone()[0]
```

Example `db.ini`:

```ini
[database]
host = 127.0.0.1
user = root
password = root
database = XXXX
```

## NVIDIA GPU Monitor

Print GPU information:

```bash
watch_gpu
```

On Linux, you can wrap it with `watch`:

```bash
alias watch-gpu="watch -n 1 -d watch_gpu"
```

## Time Utilities

Estimate remaining time:

```python
import time
from jacksung.utils.time import RemainTime

epochs = 100
rt = RemainTime(epochs)

for _ in range(epochs):
    rt.update()
    time.sleep(2)
```

Use a stopwatch:

```python
import time
from jacksung.utils.time import Stopwatch

sw = Stopwatch()
time.sleep(1)
print(sw.pinch())  # elapsed time since last reset
time.sleep(1)
print(sw.reset())  # elapsed time and reset
```

## Data Conversion

Convert NetCDF to NumPy and NumPy arrays to TIFF/GeoTIFF.

```python
import numpy as np
from jacksung.utils.data_convert import nc2np, np2tif

nc_data, dim = nc2np(r"C:\Users\ECNU\Desktop\upper.nc")

# Without geocoordinates
np2tif(nc_data, "constant_masks/upper", dim_value=dim)

# With geocoordinates
np2tif(
    "constant_masks/land_mask.npy",
    save_path="constant_masks",
    out_name="land_mask",
    left=0,
    top=90,
    x_res=0.25,
    y_res=0.25,
    dtype=np.float32,
)
```

`dim_value` can be used to name output files generated from multi-dimensional arrays:

```python
dim_value = [{"value": ["WIN", "TMP"]}, {"value": ["PRS", "HEIGHT"]}]
```

Other useful helpers include `Coordinate`, `nc2tif`, `get_transform_from_lonlat_matrices`, `haversine_distance`, and `fill_nan_with_window_mean`.

## Image Utilities

`jacksung.utils.image` includes helpers for drawing text and borders, building color maps, cropping PNGs, concatenating images, creating GIFs, zooming image regions, and drawing boundaries.

```python
import cv2
from jacksung.utils.image import concatenate_images, create_gif

img1 = cv2.imread("a.png")
img2 = cv2.imread("b.png")
merged = concatenate_images([img1, img2], direction="h")
cv2.imwrite("merged.png", merged)

create_gif("frames_dir", "demo.gif", duration=500)
```

## AI Tools

### LaTeX Auto Polish

`jacksung.ai.latex_tool.polish` can polish a LaTeX manuscript through an OpenAI-compatible LLM server.

```python
from jacksung.ai.latex_tool import polish

polish(
    main_dir_path="your latex root directory",
    tex_file="main.tex",
    server_url="The full LLM server url with /v1",
    token="Your token here",
)
```

Notes:

- If the paper is Chinese or needs a Chinese prompt, set `cn_prompt=True`.
- To use a custom prompt, pass `prompt` containing `{text}`.
- The tool generates `old.tex`, `new.tex`, and `diff.tex` in the parent directory.
- The change-tracking PDF is compiled from `diff.tex`.
- If `diff.tex` fails to compile, fix the generated `new.tex` first.
- A strong model is recommended; small models may introduce LaTeX syntax errors.

### Metrics and Meteorological Utilities

The `jacksung.ai` package also contains:

- `jacksung.ai.metrics`: precipitation metrics, RMSE, PSNR, SSIM, AUROC, and bootstrap uncertainty.
- `jacksung.ai.GeoNet`: GeoNet model definitions.
- `jacksung.ai.GeoAttX`: prediction helpers for GeoAttX-related workflows.
- `jacksung.ai.utils`: satellite and precipitation data utilities for FY, FY-3G, GOES, GSMaP, IMERG, CMORPH, SEVIRI/Meteosat, normalization, and PyTorch training helpers.

These modules depend on scientific and geospatial packages such as NumPy, rasterio, netCDF4, satpy, pyresample, OpenCV, Pillow, and PyTorch.

## Hash and Miscellaneous Utilities

```python
from jacksung.utils.hash import calculate_file_hash, hash_string
from jacksung.utils.number import round2str

print(hash_string("hello"))
print(calculate_file_hash("README.md"))
print(round2str(3.14159, digits=2))
```

## Development and Release （for developers）

Build and upload a release:

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

Be aware that `setup.py` currently increments the local version stored in `loacaldb.json`, removes build artifacts, and attempts to commit the version update with Git.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).

## Contact

Maintained by Zijiang Song. Contact: <jacksung1995@gmail.com>.
