#!/bin/bash
set -eo pipefail

docker build -t python -f .docker/Dockerfile_python .
docker build -t uncertainpy -f .docker/Dockerfile_uncertainpy .
docker run -i -v $(pwd):/home/docker/uncertainpy -t uncertainpy bash