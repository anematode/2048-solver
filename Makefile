
FLAGS=-O2 -std=c++17 -g -Wall
ARCH=$(shell uname -p)

ifeq ($(ARCH), arm)
	FLAGS:=$(FLAGS) -mcpu=apple-m1
else
	FLAGS:=$(FLAGS) -march=native
endif

test: test.cc 2048.h dbg.cc
	g++ test.cc -o bin/test $(FLAGS)

main: main.cc 2048.h dbg.cc
	g++ main.cc -o bin/main $(FLAGS)

.PHONY clean:
	rm bin/*
