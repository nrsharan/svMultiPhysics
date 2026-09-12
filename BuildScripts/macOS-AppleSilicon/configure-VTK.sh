#!/bin/bash
# VTK 9.7.0 for svMultiPhysics on macOS / Apple silicon: static libraries with
# only the reading/writing part of VTK (no rendering, imaging, views, web, Qt,
# MPI or Python), installed to ~/opt/vtk (2026-09). Reconstructed from the
# CMakeCache.txt of that build; it follows the VTK instructions in the
# top-level README.md.
#
# Usage:
#   curl -LO https://www.vtk.org/files/release/9.7/VTK-9.7.0.tar.gz && tar xf VTK-9.7.0.tar.gz
#   mkdir build && cd build && VTK_SOURCE_DIR=../VTK-9.7.0 <this script> && make -j8 install

VTK_SOURCE_DIR=${VTK_SOURCE_DIR:-$HOME/dev/vtk/VTK-9.7.0}
INSTALL_DIR=${VTK_INSTALL_DIR:-$HOME/opt/vtk}

cmake -G "Unix Makefiles" \
      -DBUILD_SHARED_LIBS:BOOL=OFF \
      -DCMAKE_BUILD_TYPE:STRING=RELEASE \
      -DVTK_BUILD_EXAMPLES=OFF \
      -DVTK_BUILD_TESTING=OFF \
      -DVTK_USE_SYSTEM_EXPAT:BOOL=ON \
      -DVTK_USE_SYSTEM_ZLIB:BOOL=ON \
      -DVTK_WRAP_PYTHON=OFF \
      -DVTK_WRAP_JAVA=OFF \
      -DVTK_GROUP_ENABLE_Imaging=NO \
      -DVTK_GROUP_ENABLE_Views=NO \
      -DVTK_GROUP_ENABLE_Web=NO \
      -DVTK_GROUP_ENABLE_Qt=DONT_WANT \
      -DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
      -DVTK_GROUP_ENABLE_MPI=DONT_WANT \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      $VTK_SOURCE_DIR
