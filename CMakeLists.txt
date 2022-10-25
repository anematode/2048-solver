cmake_minimum_required(VERSION 3.10)
find_package(Catch2 3 REQUIRED)
project(morbius)

set(CMAKE_CXX_STANDARD 17)  # if constexpr used
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O2")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CURRENT_SOURCE_DIR src)

if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
	# guessing Apple, which refuses -march=native for some reason
	message(WARNING "Guessing Apple M1 processor. If this is an ARM chip, change this to the appropriate flag.")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(SOURCES src/shuffle.cc src/shuffle.h src/move_lut.cc src/move_lut.h src/position.cc src/position.h)

add_executable(main src/main.cc ${SOURCES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCATCH_CONFIG_ENABLE_BENCHMARKING")
add_executable(test tests/test.cc tests/helper.h tests/helper.cc ${SOURCES})

target_link_libraries(test PRIVATE Catch2::Catch2WithMain)
