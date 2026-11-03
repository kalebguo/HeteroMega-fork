#!/usr/bin/env python3
"""Sample sensors at 10 Hz until killed: time, GPU busy percent, active sclk/fclk/mclk in MHz, socket power in W, busy CPU cores, average shader clock in Hz."""
import glob
import os
import re
import sys
import time

DEVICE = '/sys/class/drm/card0/device/'


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ''


def active_mhz(name):
    for line in read(DEVICE + name).splitlines():
        if line.rstrip().endswith('*'):
            found = re.search(r'(\d+)\s*Mhz', line, re.I)
            if found:
                return found.group(1)
    return ''


def cpu_times():
    fields = [int(v) for v in read('/proc/stat').splitlines()[0].split()[1:]]
    return sum(fields) - fields[3] - fields[4], sum(fields)


freq_paths = glob.glob(DEVICE + 'hwmon/hwmon*/freq1_input')
power_paths = glob.glob(DEVICE + 'hwmon/hwmon*/power1_input') + glob.glob(DEVICE + 'hwmon/hwmon*/power1_average')
with open(sys.argv[1], 'w', buffering=1) as out:
    out.write('t,busy,sclk,fclk,mclk,power,cores,freq\n')
    cores = os.cpu_count()
    last = cpu_times()
    while True:
        busy = read(DEVICE + 'gpu_busy_percent').strip()
        power = read(power_paths[0]).strip() if power_paths else ''
        now = cpu_times()
        used = cores * (now[0] - last[0]) / max(now[1] - last[1], 1)
        last = now
        out.write(f"{time.time():.2f},{busy},{active_mhz('pp_dpm_sclk')},{active_mhz('pp_dpm_fclk')},{active_mhz('pp_dpm_mclk')},"
                  f"{int(power) / 1e6 if power else ''},{used:.1f},{read(freq_paths[0]).strip() if freq_paths else ''}\n")
        time.sleep(0.1)
