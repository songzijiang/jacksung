import sys
import os
import re
import subprocess
import select
import argparse
from termcolor import colored

MEMORY_FREE_RATIO = 0.05
MEMORY_MODERATE_RATIO = 0.9
GPU_FREE_RATIO = 0.05
GPU_MODERATE_RATIO = 0.75


def colorize(_lines):
    for i in range(len(_lines)):
        line = _lines[i]
        m = re.match(r"\| (?:N/A|..%)\s+[0-9]{2,3}C.*\s([0-9]+)MiB\s+\/\s+([0-9]+)MiB.*\s([0-9]+)%", line)
        if m is not None:
            used_mem = int(m.group(1))
            total_mem = int(m.group(2))
            gpu_util = int(m.group(3)) / 100.0
            mem_util = used_mem / float(total_mem)

            is_low = is_moderate = is_high = False
            is_high = gpu_util >= GPU_MODERATE_RATIO or mem_util >= MEMORY_MODERATE_RATIO
            if not is_high:
                is_moderate = gpu_util >= GPU_FREE_RATIO or mem_util >= MEMORY_FREE_RATIO

            if not is_high and not is_moderate:
                is_free = True

            c = 'red' if is_high else ('yellow' if is_moderate else 'green')
            _lines[i] = colored(_lines[i], c)
            _lines[i - 1] = colored(_lines[i - 1], c)

    return _lines


def main():
    parser = argparse.ArgumentParser(
        prog='nvidia-watch',  # 程序名
        description='watch gpu',  # 描述
        epilog='Copyright(r), 2023'  # 说明信息
    )
    parser.add_argument('-l', '--command-length', default=20, const=100, type=int, nargs='?')
    parser.add_argument('-c', '--color', action='store_true')
    args = parser.parse_args()

    # parse the command length argument
    command_length = args.command_length
    color = args.color

    processes = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    lines = processes.stdout.decode().split("\n")[:-1]
    lines_to_print = []
    # Copy the utilization upper part verbatim
    pid_idx = 0
    for i in range(len(lines)):
        if not lines[i].startswith("| Processes:"):
            if lines[i].count('MIG M.') > 0 or lines[i].count('N/A') > 0:
                continue
            lines_to_print.append(lines[i].rstrip())
        else:
            pid_idx = i + 3
            break

    if color:
        lines_to_print = colorize(lines_to_print)

    for line in lines_to_print:
        print(line)

    # Parse the PIDs from the lower part
    gpu_num = []
    pid = []
    gpu_mem = []
    user = []
    cpu = []
    mem = []
    time = []
    command = []
    for i in range(pid_idx, len(lines)):
        if lines[i].startswith("+--") or lines[i].startswith('|==') or "Not Supported" in lines[i]:
            continue
        no_running_process = "No running processes found"
        if no_running_process in lines[i]:
            print("|  " + no_running_process + " " * (83 - len(no_running_process)) + "  |")
            print(lines[-1])
            sys.exit()
        line = lines[i]
        line = re.split(r'\s+', line)
        gpu_num.append(line[1])
        pid.append(line[4])
        gpu_mem.append(line[7])
        user.append("")
        cpu.append("")
        mem.append("")
        time.append("")
        command.append("")

    # Query the PIDs using ps
    ps_format = "pid,user,%cpu,%mem,etime,command"
    processes = subprocess.run(["ps", "-o", ps_format, "-p", ",".join(pid)], stdout=subprocess.PIPE)

    # Parse ps output
    for line in processes.stdout.decode().split("\n"):
        if line.strip().startswith('PID') or len(line) == 0:
            continue
        parts = re.split(r'\s+', line.strip(), 5)
        idx = pid.index(parts[0])
        user[idx] = parts[1]
        cpu[idx] = parts[2]
        mem[idx] = parts[3]
        time[idx] = parts[4] if not "-" in parts[4] else parts[4].split("-")[0] + " days"
        command[idx] = parts[5][0:100]

    format = ("| %5s %10s %8s %10s %8s %8s %8s %-" + str(command_length) + "." + str(command_length) + "s  |")

    print(format % (
        "GPU", "PID", "USER", "GPU MEM", "%CPU", "%MEM", "TIME", "COMMAND"
    ))

    for i in range(len(pid)):
        # continue
        print(format % (
            gpu_num[i],
            pid[i],
            user[i],
            gpu_mem[i],
            cpu[i],
            mem[i],
            time[i],
            command[i]
        ))

    print(lines[-1])
