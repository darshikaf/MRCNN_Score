#!/usr/bin/env bash
#
# Commands:
# - build:     builds all the required images
# - run:       run instance by replacing all running containers.
# - run-bash:  run a Bash session in the instance by replacing all running containers.
#
######################################################################

if [ $1 = "run" ] || [ $1 = "run-bash" ]; then
   export IID=${2? container ID not supplied}
fi

function build() {
    make local
}

function run-bash() {
    docker run --rm -it \
        --name mrcnn \
        -v $(pwd):/mrcnn \
        -w /mrcnn \
    ${IID} bash
    echo "Started scoring bash session."
}

function for_manifest() {
    fun_name=$1
    $fun_name
}

for_manifest ${1:-status}