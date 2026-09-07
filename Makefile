# gemm2d — build
MPICC   ?= mpicc
# -ffp-contract=fast lets the compiler fuse a*b+c into a single FMA.
# GCC disables contraction in strict ISO mode (-std=c11), which costs the
# local kernel roughly 20%. This is far weaker than -ffast-math -- it does
# not reorder or assume associativity, it only skips one rounding step per
# product -- but it does change results in the last bit, so report it
# (guide S8.4) and keep it in mind when comparing residuals across builds.
OPT     ?= -O3 -march=native -funroll-loops -ffp-contract=fast
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
