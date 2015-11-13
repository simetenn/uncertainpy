import multiprocess as mp

from xvfbwrapper import Xvfb

def f(x):
    with Xvfb() as xvfb:
        pass

print mp.current_process().name

pool = mp.Pool(4)
for i in range(500):

    #with Xvfb() as xvfb:
    pool.map(f, range(1, 9))
