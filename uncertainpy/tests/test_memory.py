from memory import Memory
import time

log = Memory()
print "starting log"
log.start()
print "log is started"
time.sleep(120)
log.end()
print "logging finished"
