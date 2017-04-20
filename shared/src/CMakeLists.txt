cmake_minimum_required(VERSION 2.8)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set(SRCS main.cpp parseUtils.cpp)
include_directories( ${CMAKE_SOURCE_DIR}/src )

find_package(OpenMP REQUIRED)

find_package(LibFFTW3)
if (LibFFTW3_FOUND)
  list(APPEND SRCS FFTW/C2C_fft_driver.cpp FFTW/Z2Z_fft_driver.cpp
    FFTW/R2C_fft_driver.cpp FFTW/D2Z_fft_driver.cpp
    FFTW/fftwbench.cpp)
  include_directories( ${LibFFTW3_INCLUDE_DIRS} )
  include_directories( ${CMAKE_SOURCE_DIR}/src/FFTW )
  list(APPEND LIBS ${LibFFTW3_LIBRARIES} )
  list(APPEND LIBS ${LibFFTW3_PARALLEL_LIBRARIES} )
  list(APPEND LIBS ${LibFFTW3F_LIBRARIES} )
  list(APPEND LIBS ${LibFFTW3F_PARALLEL_LIBRARIES} )
  add_definitions(-DENABLEFFTW)
endif ()

add_executable( fftime ${SRCS} )

set_target_properties(fftime PROPERTIES
  COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")

set_target_properties(fftime PROPERTIES
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")

target_link_libraries(fftime ${LIBS} )

