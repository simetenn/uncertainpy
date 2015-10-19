import multiprocess as mp

def f(x):
    print mp.current_process().name.split("-")[-1]
    return x**2

pool = mp.Pool(4)
result = pool.map(f, range(1, 5))
print mp.current_process()
print mp.current_process().name.split("-")[-1]
