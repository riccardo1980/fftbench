cmake_minimum_required(VERSION 2.8)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set(SRCS main.cpp)

if (HAVE_FFTW)
  list(APPEND SRCS FFTW/C2C_fft_driver.cpp FFTW/Z2Z_fft_driver.cpp
    FFTW/R2C_fft_driver.cpp FFTW/D2Z_fft_driver.cpp
    FFTW/fftwbench.cpp)
  list(APPEND CXX_FLAGS -IFFTW)
  add_definitions(-DENABLEFFTW)
endif ()

if (HAVE_MKL)
  list(APPEND SRCS MKL/*.cpp)
  list(APPEND CXX_FLAGS -IMKL)
endif ()

if (HAVE_CUDA)
  list(APPEND SRCS CUFFT/*.cpp)
  list(APPEND CXX_FLAGS -ICUFFT)
endif ()

add_executable( fftime ${SRCS} )

set_target_properties(fftime PROPERTIES
  COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
list(APPEND CMAKE_CXX_FLAGS ${CXX_FLAGS})

set_target_properties(fftime PROPERTIES
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")

  #target_link_libraries(fftime ${TBB_LIBRARIES})

