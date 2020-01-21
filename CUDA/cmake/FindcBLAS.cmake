
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================
#
# Defines the following variables:
#   CBLAS_FOUND          Boolean holding whether or not the Netlib BLAS library was found
#   CBLAS_INCLUDE_DIRS   The Netlib BLAS include directory
#   CBLAS_LIBRARIES      The Netlib BLAS library
#
# In case BLAS is not installed in the default directory, set the CBLAS_ROOT variable to point to
# the root of BLAS, such that 'cblas.h' can be found in $CBLAS_ROOT/include. This can either be
# done using an environmental variable (e.g. export CBLAS_ROOT=/path/to/BLAS) or using a CMake
# variable (e.g. cmake -DCBLAS_ROOT=/path/to/BLAS ..).
#
# ==================================================================================================

# Sets the possible install locations
set(CBLAS_HINTS
  ${CBLAS_ROOT}
  $ENV{CBLAS_ROOT}
)
set(CBLAS_PATHS
  /usr
  /usr/local
  /usr/local/opt
  /System/Library/Frameworks
)

# Finds the include directories
find_path(CBLAS_INCLUDE_DIRS
  NAMES cblas.h
  HINTS ${CBLAS_HINTS}
  PATH_SUFFIXES
    include inc include/x86_64 include/x64
    openblas/include include/blis blis/include blis/include/blis
    Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers
  PATHS ${CBLAS_PATHS}
  DOC "Netlib BLAS include header cblas.h"
)
mark_as_advanced(CBLAS_INCLUDE_DIRS)

# Finds the library
find_library(CBLAS_LIBRARIES
  NAMES cblas blas blis openblas accelerate
  HINTS ${CBLAS_HINTS}
  PATH_SUFFIXES
    lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
    openblas/lib blis/lib lib/atlas-base
  PATHS ${CBLAS_PATHS}
  DOC "Netlib BLAS library"
)
mark_as_advanced(CBLAS_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT CBLAS_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'cblas.h', install a CPU Netlib BLAS or set CBLAS_ROOT")
endif()
if(NOT CBLAS_LIBRARIES)
    message(STATUS "Could NOT find a CPU Netlib BLAS library, install it or set CBLAS_ROOT")
endif()

# Determines whether or not BLAS was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_INCLUDE_DIRS CBLAS_LIBRARIES)

# ==================================================================================================