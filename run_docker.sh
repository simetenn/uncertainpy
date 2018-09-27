#!/bin/bash
set -eo pipefail

if [ $# -eq 0 ]
    then
        echo "Python 2 or 3 must be specified"
        exit 1
fi

docker build -t python$1 -f .docker/Dockerfile_python$1 .
docker build -t uncertainpy$1 -f .docker/Dockerfile_uncertainpy$1 .
docker run -i -v $(pwd):/home/docker/uncertainpy -t uncertainpy$1 bash