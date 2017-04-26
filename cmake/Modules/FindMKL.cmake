# Users may modify the behavior of this module with the following environment
# variables:
#
# * MKLROOT 
# * MKL_INCLUDE_PATH 
# * MKL_LIBRARY_PATH 
#
# Try to find MKL library 
# Once done, this will define
# LibMKL_FOUND
# LibMKL_INCLUDE_DIRS
# LibMKL_LIBRARIES
# LibMKL_PARALLEL_LIBRARIES

find_path( LibMKL_INCLUDE_DIR mkl.h 
  HINTS  
  $ENV{MKLROOT}/include
  $ENV{MKL_INCLUDE_PATH}
  /usr/include /usr/local/include
  DOC "MKL headers"
  )

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(__path_suffixes lib lib/ia32)
else()
  set(__path_suffixes lib lib/intel64)
endif()

find_library(LibMKL_LIBRARY
