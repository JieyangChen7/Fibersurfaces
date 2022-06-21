ml cuda/11.5.2
ml cmake/3.23.1
ml gcc/7.5.0

mkdir -p build

src_dir=.
build_dir=build

cmake -S ${src_dir} -B ${build_dir} \
      -DVTK_DIR=/ccs/home/jieyang/VTK/install/lib/cmake/vtk-9.2 \
      -DCMAKE_PREFIX_PATH=/ccs/home/jieyang/VTK/install \
      -DCMAKE_CUDA_ARCHITECTURES=70

cmake --build ${build_dir}

