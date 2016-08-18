import argparse
import subprocess
import sys


parser = argparse.ArgumentParser(description="Run a Python file until it fails")
parser.add_argument("file")

args = parser.parse_args()


count = 1
result = 0
while result == 0:
    print "\rRunning: {}".format(count),
    sys.stdout.flush()

    program = subprocess.Popen(["python", args.file],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    ut, err = program.communicate()
    result = program.returncode

    count += 1


subprocess.call(["play", "-q", "-v", "0.2", "crash.wav"])


print
print "-----------------------------------------------------------------------"
print "{} failed after {} repeat(s)".format(args.file, count)
print "-----------------------------------------------------------------------"
print ut
print "-----------------------------------------------------------------------"
print err
print "-----------------------------------------------------------------------"
