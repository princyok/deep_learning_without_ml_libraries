"""
@author: Prince Okoli
"""
from numpy import __version__ as np_version
def is_iterable(arg):
    """
    Checks if an object or data is iterable, and returns True if it is.
    """
    try:
        (i for i in arg)
        ans=True
    except TypeError:
        ans=False
    return ans
    

def sigfig(number, num_sigfig=5):
    """
    Rounds a number to the specified significant figure.
    """
    dtype=type(number)
    fmt_str="{:."+str(num_sigfig)+"g}"
    result=float(fmt_str.format(number))
    return dtype(result)

def check_numpy_ver():
    if(int(np_version[0])!=1):
        message="Caution: This module was created with numpy major version 1, "+\
        "but the current major version is "+str(np_version[0])
        print(message)