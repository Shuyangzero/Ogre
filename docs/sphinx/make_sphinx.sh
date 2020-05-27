#!/bin/bash

# cd ../../
# python setup_simple.py install
# cd -

build="build_sphinx/"
source="source/"
make_target="latex"
sphinx-build -b $make_target $source $build
make $make_target
