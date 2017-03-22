import argparse
import sys
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Run a model simulation")
    parser.add_argument("--model_name")
    parser.add_argument("--file_dir")
    parser.add_argument("--file_name")
    parser.add_argument("--save_path")
    parser.add_argument("--CPU", type=int)
    parser.add_argument("--parameters", nargs="*")
    parser.add_argument("--model_kwargs", nargs="*")

    args = parser.parse_args()

    sys.path.insert(0, args.file_dir)
    module = __import__(args.file_name.split(".")[0])
    model = getattr(module, args.model_name)

    model_kwargs = dict(zip(args.model_kwargs[::2], args.model_kwargs[1::2]))
    simulation = model(**model_kwargs)

    if args.parameters is None:
        args.parameters = []


    if len(args.parameters) % 2 != 0:
        msg = "Number of parameters does not match number of parametervalues sent to simulation"
        raise ValueError(msg)


    parameters = {}
    i = 0
    while i < len(args.parameters):
        parameters[args.parameters[i]] = float(args.parameters[i + 1])
        i += 2


    result = simulation.run(parameters)

    if (not isinstance(result, tuple) or not isinstance(result, list)) and len(result) != 2:
        raise RuntimeError("model.run() must return t and U (return t, U | return None, U)")

    t, U = result

    if U is None:
        raise ValueError("U has not been calculated")

    if t is None:
        t = np.nan

    np.save(os.path.join(args.save_path, ".tmp_U_%d" % args.CPU), U)
    np.save(os.path.join(args.save_path, ".tmp_t_%d" % args.CPU), t)


if __name__ == "__main__":
    main()
