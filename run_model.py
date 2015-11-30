import argparse

def main():
    parser = argparse.ArgumentParser(description="Run a model simulation")
    parser.add_argument("--model_name")
    parser.add_argument("--CPU", type=int)

    args, parameter_args = parser.parse_known_args()

    module = __import__(args.model_name)
    model = getattr(module, args.model_name)

    simulation = model()
    simulation.load()

    tmp_parameters = parameter_args[len(simulation.cmd()) - 4:]

    if len(tmp_parameters) % 2 != 0:
        print "ERROR: Number of parameters does not match number"
        print "         of parametervalues sent to simulation.py"
        sys.exit(1)

    parameters = {}
    i = 0
    while i < len(tmp_parameters):
        parameters[tmp_parameters[i]] = float(tmp_parameters[i+1])
        i += 2

    simulation.setParameters(parameters)
    simulation.run()
    simulation.save(args.CPU)


if __name__ == "__main__":
    main()
