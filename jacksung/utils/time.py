import time
from datetime import datetime
import pytz
from jacksung.utils.log import oprint
import calendar


def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]


def cal_time(fun_str):
    # 记录第一个时间戳
    start_time = time.time()
    eval(fun_str)
    # 记录第二个时间戳
    end_time = time.time()
    # 计算耗时
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time, "seconds")


def cur_timestamp_str():
    now = datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)
    microsecond = str(now.microsecond // 10000).zfill(2)
    content = "{}-{}{}-{}{}-{}{}".format(year, month, day, hour, minute, second, microsecond)
    return content


class RemainTime:
    def __init__(self, epoch):
        self.start_time = time.time()
        self.epoch = epoch
        self.now_epoch = 0

    def update(self, log_temp='Spe {:.0f}s, Rem Epoch:{}, Fin in {}', print_log=True, update_step=1):
        self.now_epoch += update_step
        epoch_time = time.time() - self.start_time
        epoch_remaining = self.epoch - self.now_epoch
        time_remaining = epoch_time * epoch_remaining
        pytz.timezone('Asia/Shanghai')  # 东八区
        t = datetime.fromtimestamp(int(time.time()) + time_remaining,
                                   pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        log = log_temp.format(epoch_time, epoch_remaining, t)
        if print_log:
            oprint(log)
        self.start_time = time.time()
        return epoch_remaining, t


class Stopwatch:
    def __init__(self):
        self.start_time = time.time()

    def reset(self, format_type='{:>5.1f}s'):
        timer_end = time.time()
        duration = timer_end - self.start_time
        self.start_time = timer_end
        return format_type.format(duration)

    def pinch(self, format_type='{:>5.1f}s'):
        return format_type.format(time.time() - self.start_time)


def getHumanSize(in_size):
    unit = 'B'
    if in_size >= 1024:
        in_size /= 1024
        unit = 'K'
    if in_size >= 1024:
        in_size /= 1024
        unit = 'M'
    if in_size >= 1024:
        in_size /= 1024
        unit = 'G'
    return f'{round(in_size, 2)} {unit}'


if __name__ == '__main__':
    print(getHumanSize(1023))
