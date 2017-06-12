import subprocess
from uncertainpy import Model
import yaml
import os
import sys
import h5py

from shutil import copyfile

class LgnSimulator(Model):
    def __init__(self,
                 config_file_base=None,
                 config_file=None,
                 output_file=None):

        Model.__init__(self,
                       labels=["spatial points", "response"])


        self.config_file_base = config_file_base
        self.config_file = config_file
        self.output_file = output_file


    def set_parameters(self, **parameters):
        copyfile(self.config_file_base, self.config_file)

        print "config file: ", self.config_file
        with open(self.config_file, 'r') as stream:
            config_data = yaml.load(stream)

        print parameters

        config_data["OutputManager"]["outputFilename"] = unicode(self.output_file)
        if "w_rc" in parameters:
            config_data["relay"]["Krc"]["w"] = unicode(parameters["w_rc"])
        if "w_ri" in parameters:
            config_data["relay"]["Kri"]["w"] = unicode(parameters["w_ri"])
        if "w_ic" in parameters:
            config_data["interneuron"]["Kic"]["w"] = unicode(parameters["w_ic"])

        if "a_ri" in parameters:
            config_data["relay"]["Kri"]["spatial"]["a"] = unicode(parameters["a_ri"])
        if "a_rc" in parameters:
            config_data["relay"]["Krc"]["spatial"]["a"] = unicode(parameters["a_rc"])
        if "a_ic" in parameters:
            config_data["interneuron"]["Kic"]["spatial"]["a"] = unicode(parameters["a_ic"])


        with open(self.config_file, 'w') as stream:
            yaml.safe_dump(config_data, stream, default_flow_style=False)


    def run(self, **parameters):
        self.set_parameters(**parameters)

        build_path = "/home/simen/src/lgn-simulator-build"
        project_path = "/home/simen/src/lgn-simulator"
        app_name = "spatialSummation"
        sys.path.append("/home/simen/src/lgn-simulator/tools")

        print "Building in:\n", build_path


        #build and run----------------------------------------------------------------------------------
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        subprocess.call(["qmake", project_path], cwd=build_path)
        subprocess.call(["make", "-j", "8"], cwd=build_path)

        app_path = os.path.join(build_path, "apps", app_name)
        lib_path = os.path.join(build_path, "lib")

        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = lib_path

        run_argument = ["./lgnSimulator_spatialSummation",
                        self.config_file,
                        os.path.dirname(self.config_file)]
        print " ".join(run_argument)
        proc = subprocess.call(run_argument, cwd=app_path, env=env)

        print "Results saved to this directory:\n", os.path.dirname(self.config_file) + "/*"
        #os.remove(config_file)

        #Reading data---------------------------------------------------------------------
        import analysis.Simulation as sim

        f = h5py.File(self.output_file, "r")
        exp = sim.Simulation(self.config_file, f)
        Ns = exp.integrator.Ns
        Nt = exp.integrator.Nt
        s_points = exp.integrator.s_points[Ns/2:]
        irf = exp.relay.irf()[0, Ns/2, Ns/2:]

        t = s_points
        U = irf

        return t, U



if __name__ == "__main__":
    config_file_base = "/home/simen/src/lgn-simulator/apps/spatialSummation/spatialSummation.yaml"
    config_file = "/home/simen/src/lgn-simulator/apps/spatialSummation/_spatialSummation.yaml"
    output_file = "/home/simen/src/lgn-simulator/apps/spatialSummation/spatialSummation.h5"

    parameters = {"w_rc": 1.0, "w_ic": 1.0}

    lgn = LgnSimulator(config_file=config_file,
                       config_file_base=config_file_base,
                       output_file=output_file)

    lgn.run(**parameters)
