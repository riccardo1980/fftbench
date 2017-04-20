# Users may modify the behavior of this module with the following environment
# variables:
#
# * FFTW3ROOT 
# * FFTW3_INCLUDE_PATH 
# * FFTW3_LIBRARY_PATH 
#
# Try to find FFTW3 library 
# Once done, this will define
# LibFFTW3_FOUND
# LibFFTW3_INCLUDE_DIRS
# LibFFTW3_LIBRARIES
# LibFFTW3_PARALLEL_LIBRARIES

include(FindPackageHandleStandardArgs) 

find_path( LibFFTW3_INCLUDE_DIR fftw3.h 
  HINTS  
  $ENV{FFTW3ROOT}/include
  $ENV{FFTW3_INCLUDE_PATH}
  /usr/include /usr/local/include
  DOC "FFTW3 headers"
  )

find_library( LibFFTW3_LIBRARY
  NAMES fftw3
  HINTS 
  $ENV{FFTW3ROOT}/lib
  $ENV{FFTW3_LIBRARY_PATH}
  /usr/lib /usr/local/lib
  )

find_library( LibFFTW3_PARALLEL_LIBRARY
  NAMES fftw3_threads
  HINTS 
  $ENV{FFTW3ROOT}/lib
  $ENV{FFTW3_LIBRARY_PATH}
  /usr/lib /usr/local/lib
  )    

find_library( LibFFTW3F_LIBRARY
  NAMES fftw3f
  HINTS 
  $ENV{FFTW3ROOT}/lib
  $ENV{FFTW3_LIBRARY_PATH}
  /usr/lib /usr/local/lib
  )

find_library( LibFFTW3F_PARALLEL_LIBRARY
  NAMES fftw3f_threads
  HINTS 
  $ENV{FFTW3ROOT}/lib
  $ENV{FFTW3_LIBRARY_PATH}
  /usr/lib /usr/local/lib
  )    


find_package_handle_standard_args(LibFFTW3  DEFAULT_MSG 
  LibFFTW3_INCLUDE_DIR LibFFTW3_LIBRARY LibFFTW3_PARALLEL_LIBRARY
  LibFFTW3F_LIBRARY LibFFTW3F_PARALLEL_LIBRARY)


mark_as_advanced( LibFFTW3_INCLUDE_DIR )
mark_as_advanced( LibFFTW3_LIBRARY )
mark_as_advanced( LibFFTW3_PARALLEL_LIBRARY )
mark_as_advanced( LibFFTW3F_LIBRARY )
mark_as_advanced( LibFFTW3F_PARALLEL_LIBRARY )
set( LibFFTW3_INCLUDE_DIRS ${LibFFTW3_INCLUDE_DIR} )
set( LibFFTW3_LIBRARIES ${LibFFTW3_LIBRARY} )
set( LibFFTW3_PARALLEL_LIBRARIES ${LibFFTW3_PARALLEL_LIBRARY} )
set( LibFFTW3F_LIBRARIES ${LibFFTW3F_LIBRARY} )
set( LibFFTW3F_PARALLEL_LIBRARIES ${LibFFTW3F_PARALLEL_LIBRARY} )

