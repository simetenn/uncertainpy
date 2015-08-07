import os, time, psutil

class Memory:
    def __init__(self):
        self.t_start = time.time()
        self._proc_status = '/proc/%d/status' % os.getpid()

        self._scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
                  'KB': 1024.0, 'MB': 1024.0*1024.0}
        self.filename = "memory.dat"
        self.total_filename= "memory_total.dat"
        self.f = open(self.filename, "w")
        self.total_f = open(self.total_filename, "w")

    def __del__(self):
        self.f.close()
        self.total_f.close()

        
    def _VmB(self, VmKey):
        '''Private.
        '''
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
        return float(v[1]) * self._scale[v[2]]


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
        mem = psutil.virtual_memory()
        return self.memory() /psutil.virtual_memory().total*100
  
        
    def save(self, info = ""):
        memory_GB = "%.2f" % (self.memory()/1024**3)
        f = open(self.filename, "a")
        t = "%.2f" % (time.time()-self.t_start)
        p = "%.2f" % (self.percent())
        if info:
           info = " : " + info 
        self.f.write(t + " " + p + " " + memory_GB + info + "\n")
        
    def saveAll(self, info = ""):
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

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from prettyPlot import prettyPlot
    
    data = np.loadtxt("memory.dat", unpack=True)
    time = data[0]
    percentage = data[1]
    memory = data[2]

    total_data = np.loadtxt("memory_total.dat", unpack=True)
    total_time = data[0]
    total_percentage = data[1]
    total_memory = data[2]
    

    prettyPlot(time, percentage, "Memory usage, percentage", "time, s", "percentage")
    prettyPlot(total_time, total_percentage, "Memory usage, percentage", "time, s", "percentage", 1, new_figure = False)
    plt.legend(["self", "total"])
    plt.savefig("memory_percentage.png", bbox_inches="tight")

    prettyPlot(time, memory, "Memory usage, GB", "time, s", "memory, GB")
    prettyPlot(total_time, total_memory, "Memory usage, GB", "time, s", "memory, GB", 1, new_figure = False)
    plt.legend(["self", "total"])
    plt.savefig("memory_GB.png", bbox_inches="tight")
    
    #prettyPlot(total_time, total_memory, "Total memory usage, percentage", "time, s", "percentage")
    #plt.savefig("memory_total_percentage.png", bbox_inches="tight")

    #prettyPlot(total_time, total_memory, "Memory usage, GB", "time, s", "memory, GB")
    #plt.savefig("memory_total_GB.png", bbox_inches="tight")
