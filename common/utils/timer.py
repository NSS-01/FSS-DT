"""
计时工具⏲
时间输出为绿色
单位为秒
"""

import time
from tqdm import tqdm


def get_time(func, *args):
    """
    获取函数单次运行时间
    返回值与函数返回值相同
    :param func: 函数名
    :param args: 函数参数
    :return: 与函数返回值相同
    """
    start = time.time()
    res = func(*args)
    end = time.time()
    print(f"\033[32mTime consuming of {func.__name__}: {end - start}\033[0m")
    return res


def get_avg_time(func, *args, times=10):
    """
    获取函数平均运行时间
    :param func: 函数名
    :param args: 函数参数
    :param times: 运行次数
    :return: None
    """
    start = time.time()
    for _ in range(times):
        func(*args)
    end = time.time()
    print(f"\033[32mAverage time consuming of {func.__name__}: {(end - start) / times}\033[0m")


def get_time_by_tqdm(func, *args, times=10):
    """
    获取函数平均运行时间
    通过tqdm显示进度
    :param func: 函数名
    :param args: 函数参数
    :param times: 运行次数
    :return: None
    """
    start = time.time()
    for _ in tqdm(range(times)):
        func(*args)
    end = time.time()
    print(f"\033[32mAverage time consuming of {func.__name__}: {(end - start) / times}\033[0m")
