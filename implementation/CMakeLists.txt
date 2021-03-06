cmake_minimum_required (VERSION 2.8) 

PROJECT (TinyTensor)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10")

set(CMAKE_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(SndFile REQUIRED)
find_package(PortAudio REQUIRED)
find_package(RapidJSON REQUIRED)

add_definitions(-DUSE_CPP_DEBUG_LOGGING)
add_definitions(-DNO_EQUALIZATION)

include_directories (${SNDFILE_INCLUDE_DIRS} ${PORTAUDIO_INCLUDE_DIRS} ${RAPIDJSON_INCLUDE_DIRS} "gtest/include/" "gtest" "./")
message(STATUS "${SNDFILE_INCLUDE_DIRS}")
message(STATUS "${SNDFILE_LIBRARIES}")

add_library(tensor STATIC
tinytensor_math.c
tinytensor_net.c
tinytensor_tensor.c
tinytensor_conv_layer.c
tinytensor_fullyconnected_layer.c
tinytensor_lstm_layer.c
tinytensor_features.c
)
#############
add_library(hellomath STATIC 
hellomath/fft.c 
hellomath/hellomath.c
hellomath/sha1.c)
#############
add_library(gtest STATIC gtest/gtest-all.cc)
target_link_libraries(gtest pthread)

############
add_executable(unit-test 
unit-test/main.cpp 
unit-test/testconv.cpp
unit-test/testrecurrentnet.cpp
unit-test/testrecurrentnet2.cpp
unit-test/testfeatures.cpp
unit-test/testmath.cpp
)

target_link_libraries(unit-test gtest tensor hellomath)

#copy resources over
add_custom_command(
TARGET unit-test POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_SOURCE_DIR}/unit-test/runtime_data
$<TARGET_FILE_DIR:unit-test>
)

###########
add_executable(tinyfeats
tinyfeats.cpp)

target_link_libraries(tinyfeats tensor hellomath ${SNDFILE_LIBRARIES})

###########
add_executable(vadrunner
vadrunner.cpp)

target_link_libraries(vadrunner tensor hellomath ${SNDFILE_LIBRARIES})

###########
add_executable(liverunner 
liverunner.cpp)

target_link_libraries(liverunner tensor hellomath ${PORTAUDIO_LIBRARIES})

#########
add_executable(wavrunner
wavrunner.cpp)

target_link_libraries(wavrunner tensor hellomath ${SNDFILE_LIBRARIES})
target_compile_features(wavrunner PRIVATE cxx_range_for)

#############
add_executable(serverfeatsrunner
serverfeatsrunner.cpp
base64.cpp)

target_link_libraries(serverfeatsrunner tensor hellomath ${RAPIDJSON_LIBRARIES}) 

