project(ufibersurfaces LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 2.8.10)

set(CMAKE_CXX_FLAGS "-g -Ofast -fopenmp")
set(CMAKE_CXX_STANDARD 14)

#set(VTK_DIR "/home/ssane/Downloads/VTK-install/lib/cmake/vtk-8.2")
find_package(VTK REQUIRED)
if(VTK_FOUND)
 add_definitions("-DUSE_VTK")
  message("Found VTK")
  include_directories(${VTK_INCLUDE_DIRS})
  link_libraries(${VTK_LIBRARIES})
#include(${VTK_USE_FILE})
endif()

find_package(Boost)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  add_executable(uniform_fs uniform.cxx)
  add_executable(gaussian_fs gaussian.cxx)

#target_link_libraries(ufibersurfaces PRIVATE ${VTK_LIBRARIES})
else()
message("Boost not found")
endif()

add_executable(uniform_gpu uniform.cu)