# Helios standard CMakeLists.txt file version 1.9
cmake_minimum_required(VERSION 3.15)
project(helios)

# Keep legacy third-party CMakeLists working with newer CMake versions.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum CMake policy version" FORCE)

#-------- USER INPUTS ---------#

#provide the path to Helios base directory, either as an absolut path or a path relative to the location of this file
set( BASE_DIRECTORY "../.." )

#define the name of the executable to be created
set( EXECUTABLE_NAME "executable" )

#provide name of source file(s) (separate multiple file names with semicolon)
set( SOURCE_FILES "main.cpp" )

#specify which plug-ins to use (separate plug-in names with semicolon)
set( PLUGINS "radiation;plantarchitecture;energybalance;visualizer;voxelintersection;canopygenerator;solarposition;photosynthesis;stomatalconductance;planthydraulics" )

#-------- DO NOT MODIFY ---------#
include( "${BASE_DIRECTORY}/core/CMake_project.txt" )
