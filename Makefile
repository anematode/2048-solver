FLAGS=-O2 -std=c++20 -g -Wall
ARCH=$(shell uname -p)

.PHONY clean:
	rm bin/*

ifeq ($(ARCH), arm)
	# macOS compiler rejects march=native, for some reason....
	FLAGS:=$(FLAGS) -mcpu=apple-m1
else
	FLAGS:=$(FLAGS) -march=native
endif

main: main.cc
	g++ test.cc -o bin/test $(FLAGS)

