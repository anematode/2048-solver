cmake_minimum_required(VERSION 3.10)
project(morbius)

set(CMAKE_CXX_STANDARD 17)  # if constexpr used
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
	# guessing Apple, which refuses -march=native for some reason
	message(WARNING "Guessing Apple M1 processor. If this is an ARM chip, change this to the appropriate flag.")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

add_executable(test src/test.cc)
