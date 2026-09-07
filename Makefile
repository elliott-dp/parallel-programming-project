# gemm2d — build
MPICC   ?= mpicc
OPT     ?= -O3 -march=native -funroll-loops
WARN    ?= -Wall -Wextra
OMP     ?= -fopenmp
CFLAGS  += -std=c11 $(OPT) $(WARN) $(OMP) -Iinclude
LDFLAGS += $(OMP)
LDLIBS  += -lm

SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)
BIN := gemm2d

all: $(BIN)

$(BIN): $(OBJ)
	$(MPICC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c include/gemm2d.h
	$(MPICC) $(CFLAGS) -c -o $@ $<

test: $(BIN)
	./tests/run_tests.sh

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean test
