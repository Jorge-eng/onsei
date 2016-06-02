cmake_minimum_required (VERSION 2.8) 

PROJECT (TinyTensor)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10")

set(CMAKE_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(SndFile REQUIRED)

add_definitions(-DUSE_CPP_DEBUG_LOGGING)
add_definitions(-DNO_EQUALIZATION)

include_directories (${SNDFILE_INCLUDE_DIRS} "gtest/include/" "gtest" "./")

message(STATUS "${SNDFILE_INCLUDE_DIRS}")
message(STATUS "${SNDFILE_LIBRARIES}")

add_library(tensor STATIC
tinytensor_math.c
tinytensor_net.c
tinytensor_tensor.c
tinytensor_conv_layer.c
tinytensor_maxpool_layer.c
tinytensor_fullyconnected_layer.c
tinytensor_lstm_layer.c
tinytensor_features.c
)

add_library(hellomath STATIC 
hellomath/fft.c 
hellomath/hellomath.c)

add_library(gtest STATIC gtest/gtest-all.cc)
add_executable(unit-test 
unit-test/main.cpp 
unit-test/testconv.cpp
unit-test/testfull.cpp
unit-test/testlstm.cpp
unit-test/testmaxpool.cpp
unit-test/testnet.cpp
unit-test/testfeatures.cpp
)

target_link_libraries(unit-test gtest tensor hellomath)


add_executable(tinytensor
tinytensor.cpp)


target_link_libraries(tinytensor tensor hellomath ${SNDFILE_LIBRARIES})

