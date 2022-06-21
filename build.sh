mkdir -p build

src_dir=.
build_dir=build

cmake -S ${src_dir} -B ${build_dir} \
      -DVTK_DIR=/home/jieyang/dev/VTK-9.1.0/install/lib/cmake/vtk-9.1 \
      -DCMAKE_PREFIX_PATH=/home/jieyang/dev/VTK-9.1.0/install \
      -DBOOST_ROOT=/home/jieyang/dev/boost \
      -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build ${build_dir}

