CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all test serve bench clean
all: build/libserpent.so

build/libserpent.so: src/engine.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fPIC -shared $< -o $@

build/test_reach: tests/test_reach.cpp src/engine.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_reach.cpp -o $@

test: all build/test_reach
	./build/test_reach
	python3 -m unittest discover -s tests -v

serve: all
	python3 -m snake.server

bench: all
	python3 -m tools.benchmark --games 12 --budget-ms 15

clean:
	rm -f build/libserpent.so build/test_reach
