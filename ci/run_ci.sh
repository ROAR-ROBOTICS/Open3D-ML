#!/usr/bin/env bash
#
#
# 1. clone Open3D-ML repo (done by the CI system, this is just for testing locally)
#git clone git@github.com:intel-isl/Open3D-ML.git
PATH_TO_OPEN3D_ML=$(pwd)
echo $PATH_TO_OPEN3D_ML
cd ..
#
# 2. clone and build Open3D
# For now we have to clone the o3dml_integration branch of open3d
git clone --recursive --branch o3dml_integration  https://github.com/intel-isl/Open3D.git
#
# install deps + tf + torch
# 
# build open3d only with relevant modules enabled to minimize build time
#       -DBUNDLE_OPEN3D_ML=ON \
#       -DOPEN3D_ML_ROOT=path_to_open3d_ml_repo_root \
#       -DBUILD_TENSORFLOW_OPS=ON \
#       -DBUILD_PYTORCH_OPS=ON \
#       -DBUILD_GUI=OFF \
#       -DBUILD_RPC_INTERFACE=OFF \
#       -DBUILD_UNIT_TESTS=OFF \
#       -DBUILD_BENCHMARKS=OFF \
#       -DBUILD_EXAMPLES=OFF \
# 
./Open3D/util/install_deps_ubuntu.sh assume-yes
mkdir Open3D/build
pushd Open3D/build
cmake -DBUNDLE_OPEN3D_ML=ON \
      -DOPEN3D_ML_ROOT=$PATH_TO_OPEN3D_ML \
      -DBUILD_TENSORFLOW_OPS=ON \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_GUI=OFF \
      -DBUILD_RPC_INTERFACE=OFF \
      -DBUILD_UNIT_TESTS=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_EXAMPLES=OFF \
      ..

# 3. install wheel
make -j install-pip-package

#
# 4. run examples/tests in the Open3D-ML repo outside of the repo directory to 
#    make sure that the installed package works.
#
popd
mkdir test_workdir
pushd test_workdir
mv $PATH_TO_OPEN3D_ML/examples .
python examples/ttest_integration.py
python examples/test_semantic_seg.py
popd