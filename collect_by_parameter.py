import os
import re
import shutil
import glob

path = "figures/"
outputpath = "figures/parameter_sorted"
file_extension = ".png"


def saveByParameters():
    print "Copying files..."

    if os.path.isdir(outputpath):
        shutil.rmtree(outputpath)

    for folder in os.listdir(path):
        if re.search("._\d", folder):
            distribution, interval = folder.split("_")
            tmp_path = os.path.join(path, folder)
            for f in glob.glob(tmp_path + "/*" + file_extension):
                parameter, value = f.split(file_extension)[0].split("/")[-1].split("_")

                outputdir = os.path.join(outputpath, distribution, parameter)
                if not os.path.isdir(outputdir):
                     os.makedirs(outputdir)

                shutil.copy(f, outputdir + "/" + interval + "_" + value + file_extension)


def createGIF():
    print "Creating GIFs..."

    os.chdir(outputpath)
    for distribution in os.listdir("."):
        os.chdir(distribution)
        for value in os.listdir("."):
            os.chdir(value)
            values = []
            for f in glob.glob("*" + file_extension):
                interval, value = f.split(file_extension)[0].split("/")[-1].split("_")
                values.append(value)

            for gif_value in set(values):
                cmd = "convert -set delay 100 *_%s%s %s.gif" % ( gif_value, file_extension, gif_value)
                os.system(cmd)

            os.chdir("..")
        os.chdir("..")
    os.chdir("../..")

if __name__ == '__main__':
        saveByParameters()
        createGIF()
