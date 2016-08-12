import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run a model simulation")
    parser.add_argument("--model_name")
    parser.add_argument("--file_dir")
    parser.add_argument("--file_name")
    parser.add_argument("--save_path")
    parser.add_argument("--CPU", type=int)
    parser.add_argument("--parameters", nargs="*")
    # parser.add_argument("--kwargs", nargs="*")


    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    print sys.modules.keys()


    sys.path.insert(0, args.file_dir)
    module = __import__(args.file_name.split(".")[0])
    model = getattr(module, args.model_name)

    simulation = model()
    simulation.load()

    parameters = {}
    i = 0
    while i < len(args.parameters):
        parameters[args.parameters[i]] = float(args.parameters[i+1])
        i += 2


    if len(args.parameters) % 2 != 0:
        print "ERROR: Number of parameters does not match number"
        print "       of parametervalues sent to simulation"
        sys.exit(1)

    simulation.setParameterValues(parameters)
    simulation.run()
    simulation.save(CPU=args.CPU, save_path=args.save_path)


if __name__ == "__main__":
    main()
