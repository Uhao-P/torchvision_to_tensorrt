#find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -fPIC)

file(GLOB_RECURSE OPENBAYES_CUDA_UTIL_HEADS *.h *.hpp *.cuh)
file(GLOB OPENBAYES_CUDA_UTIL_SOURCES *.cpp *.cu)

source_group("Include" FILES ${OPENBAYES_CUDA_UTIL_HEADS})
source_group("Source" FILES ${OPENBAYES_CUDA_UTIL_SOURCES})

cuda_add_library(openbayes_cuda_util STATIC ${OPENBAYES_CUDA_UTIL_HEADS} ${OPENBAYES_CUDA_UTIL_SOURCES})
target_link_libraries(openbayes_cuda_util ${CUDA_LIBRARY})
# target_link_libraries(openbayes_cuda_util ${CUDA_npp_LIBRARY})

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
set_property(TARGET openbayes_cuda_util PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
endif()
