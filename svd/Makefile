CC = g++

CFLAGS = -std=c++11 -pedantic -Wall -I ./include -I /usr/local/include/eigen -Ofast
# CFLAGS = -std=c++11 -fopenmp -pedantic -Wall -I ./include -I ./eigen -Ofast

LIBS = read util Model Baseline SVD Time_SVD SVDpp Time_SVDpp FactorNM

OBJS = $(addprefix out/,$(LIBS:=.o))

all: bin/run_Baseline bin/run_SVD bin/run_Time_SVD bin/run_SVDpp \
	bin/run_Time_SVDpp bin/run_FactorNM \
	bin/get_train_valid_probe bin/old_Time_SVD

out/%.o: library/%.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

out/%.o: run/%.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

bin/%: out/%.o $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f out/* bin/*

.PHONY: all clean run
.PRECIOUS: out/%.o
