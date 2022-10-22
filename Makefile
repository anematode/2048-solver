
FLAGS=-O2 -std=c++20 -g -Wall -Wno-format
ARCH=$(shell uname -p)

ifeq ($(ARCH), arm)
	FLAGS:=$(FLAGS) -mcpu=apple-m1
else
	FLAGS:=$(FLAGS) -march=native
endif

test: test.cc 2048.h dbg.cc 2048.cc
	g++ test.cc 2048.cc -o bin/test $(FLAGS)

main: main.cc 2048.h dbg.cc 2048.cc
	g++ main.cc 2048.cc -o bin/main $(FLAGS)

.PHONY clean:
	rm bin/*
