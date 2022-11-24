cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(streg LANGUAGES CXX)

find_package( OpenCV REQUIRED )

set(LIBS "${OpenCV_LIBRARIES}" "stdc++fs")
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "./lib" )

file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.hpp )

add_library( streg_lib ${LIB_SOURCES} ${LIB_HEADERS} )

set_property(TARGET streg_lib PROPERTY CXX_STANDARD 14)

file( GLOB APP_SOURCES apps/*.cpp )

foreach( testsourcefile ${APP_SOURCES} )
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    get_filename_component(bn ${testname} NAME)

    add_executable( ${bn} ${testsourcefile})
    target_link_libraries( ${bn}  ${LIBS} streg_lib)
    set_property(TARGET ${bn} PROPERTY CXX_STANDARD 14)
endforeach( testsourcefile ${APP_SOURCES} )

