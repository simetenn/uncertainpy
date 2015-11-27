import os
import time
import psutil
import multiprocessing as mp

class Memory:
    """
    Class for reporting memory usage
    """
    def __init__(self, delta_poll=10):
        self.delta_poll = delta_poll

        self.t_start = time.time()
        self._proc_status = '/proc/%d/status' % os.getpid()

        self._scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
                       'KB': 1024.0, 'MB': 1024.0*1024.0}

        self.filename = "memory.dat"

        self.exit = mp.Event()


    def __del__(self):
        #self.f.close()
        try:
            self.total_f.close()
        except:
            pass

    def _VmB(self, VmKey):
        """
        Private.
        """
        # get pseudo file  /proc/<pid>/status
        try:
            t = open(self._proc_status)
            v = t.read()
            t.close()
        except:
            print "Failing?"
            return 0.0  # non-Linux?

        # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = v.index(VmKey)
        v = v[i:].split(None, 3)  # whitespace
        if len(v) < 3:
            return 0.0  # invalid format?
        # convert Vm value to bytes
        return float(v[1])*self._scale[v[2]]


    def memory(self, since=0.0):
        '''Return memory usage in bytes.
        '''
        return self._VmB('VmSize:') - since


    def resident(self, since=0.0):
        '''Return resident memory usage in bytes.
        '''
        return self._VmB('VmRSS:') - since


    def stacksize(self, since=0.0):
        '''Return stack size in bytes.
        '''
        return self._VmB('VmStk:') - since


    def percent(self):
        return self.memory()/psutil.virtual_memory().total*100

    def saveCurrentProcess(self, info=""):
        memory_GB = "%.2f" % (self.memory()/1024**3)
        t = "%.2f" % (time.time()-self.t_start)
        p = "%.2f" % (self.percent())
        if info:
            info = " : " + info
        self.f.write(t + " " + p + " " + memory_GB + info + "\n")

    def saveTotal(self, info=""):
        memory_GB = "%.2f" % (self.totalUsed()/1024**3)
        t = "%.2f" % (time.time()-self.t_start)
        p = "%.2f" % (self.totalPercent())
        if info:
            info = " : " + info
        self.total_f.write(t + " " + p + " " + memory_GB + info + "\n")

    def __str__(self):
        return "%.2f GB" % (self.memory()/1024.**3)


    def totalMemory(self):
        return psutil.virtual_memory().total


    def availableMemory(self):
        return psutil.virtual_memory().available

    def totalPercent(self):
        return psutil.virtual_memory().percent


    def totalUsed(self):
        return psutil.virtual_memory().used

    def log(self):
        self.total_f = open(self.filename, "w")
        self.t_start = time.time()
        while not self.exit.is_set():
            self.saveTotal()
            time.sleep(self.delta_poll)

        self.total_f.close()


    def start(self):
        self.p = mp.Process(target=self.log)
        self.p.start()


    def end(self):
        # Is this necesary?
        self.exit.set()
        time.sleep(self.delta_poll + 1)
        self.p.terminate()

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from prettyPlot import prettyPlot

    total_data = np.loadtxt("memory.dat", unpack=True)
    total_time = total_data[0]
    total_percentage = total_data[1]
    total_memory = total_data[2]

    prettyPlot(total_time, total_percentage, "Memory usage, percentage", "time, s",
               "percentage", 1)
    plt.savefig("memory_percentage.png", bbox_inches="tight")

    prettyPlot(total_time, total_memory, "Memory usage, GB", "time, s", "memory, GB",
               1)
    plt.savefig("memory_total.png", bbox_inches="tight")
