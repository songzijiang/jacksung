import time
from datetime import datetime
import pytz
from jacksung.utils.log import oprint


class RemainTime:
    def __init__(self, epoch):
        self.start_time = time.time()
        self.epoch = epoch
        self.now_epoch = 0

    def update(self, log_temp='Rem Epochs:{}, Fin in {}', print_log=True, update_step=1):
        self.now_epoch += update_step
        epoch_time = time.time() - self.start_time
        epoch_remaining = self.epoch - self.now_epoch
        time_remaining = epoch_time * epoch_remaining
        pytz.timezone('Asia/Shanghai')  # 东八区
        t = datetime.fromtimestamp(int(time.time()) + time_remaining,
                                   pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        log = log_temp.format(epoch_remaining, t)
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
