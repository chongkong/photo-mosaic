cmake_minimum_required(VERSION 2.8.11)
project(photo_mosaic)

set(CMAKE_C_FLAGS "-std=c99 -O3 -Wall -funroll-loops -mavx")

include_directories(extlibs)
add_subdirectory(extlibs)
set(COMMON_LIBS qdbmp log)

include_directories(src)
set(COMMON_SOURCES
    src/main.c
    src/util.c
    src/util.h
    src/photomosaic.h)

# OpenMP implementation
add_executable(omp
    ${COMMON_SOURCES}
    src/openmp/photomosaic.c
    ${EXTLIB_FILES})
set_target_properties(omp PROPERTIES COMPILE_FLAGS "-fopenmp")
target_link_libraries(omp ${COMMON_LIBS} -fopenmp)

# OpenCL implementation
add_executable(opencl
    ${COMMON_SOURCES}
    src/opencl/photomosaic.c
    src/opencl/common.h
    src/opencl/common.c
    src/opencl/clwrapper.h
    src/opencl/clwrapper.c
    ${EXTLIB_FILES})
target_link_libraries(opencl ${COMMON_LIBS} -lOpenCL)
target_compile_definitions(opencl PUBLIC _MC_OPENCL=1 NUM_GPUS=1)

# OpenCL implementation
add_executable(opencl2
    ${COMMON_SOURCES}
    src/opencl/photomosaic.c
    src/opencl/common.h
    src/opencl/common.c
    src/opencl/clwrapper.h
    src/opencl/clwrapper.c
    ${EXTLIB_FILES})
target_link_libraries(opencl2 ${COMMON_LIBS} -lOpenCL)
target_compile_definitions(opencl2 PUBLIC _MC_OPENCL=1 NUM_GPUS=4)

find_package(MPI REQUIRED)
add_executable(mpi
    ${COMMON_SOURCES}
    src/mpi/photomosaic.c
    src/opencl/common.h
    src/opencl/common.c
    src/opencl/clwrapper.h
    src/opencl/clwrapper.c
    ${EXTLIB_FILES})
target_include_directories(mpi PUBLIC ${MPI_C_INCLUDE_PATH})
target_link_libraries(mpi ${COMMON_LIBS} ${MPI_C_LIBRARIES} -lOpenCL)
target_compile_definitions(mpi PUBLIC _MC_OPENCL=1 _MC_MPI=1 NUM_GPUS=4)

add_executable(snucl
    ${COMMON_SOURCES}
    src/opencl/photomosaic.c
    src/opencl/common.h
    src/opencl/common.c
    src/opencl/clwrapper.h
    src/opencl/clwrapper.c
    ${EXTLIB_FILES})
link_directories($ENV{SNUCLROOT}/lib)
target_include_directories(snucl PUBLIC ${MPI_C_INCLUDE_PATH} $ENV{SNUCLROOT}/inc)
target_link_libraries(snucl ${COMMON_LIBS} ${MPI_C_LIBRARIES} -L$ENV{SNUCLROOT}/lib -lsnucl_cluster)
target_compile_definitions(snucl PUBLIC _MC_OPENCL=1 _MC_SNUCL=1 NUM_GPUS=16)
