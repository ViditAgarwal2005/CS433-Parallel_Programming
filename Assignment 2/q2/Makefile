CC     = gcc
# -std=gnu11: newer gcc defaults to C23, which redirects atoi/atol to
# __isoc23_strtol (glibc >= 2.38 only); gnu11 keeps the binaries portable.
CFLAGS = -std=gnu11 -O2 -Wall -Wextra -pthread

BINS = seq p1 p2 p3 p4 p5 p6

all: $(BINS) gen_ops

seq: graph.c
	$(CC) $(CFLAGS) -DIMPL=0 -o $@ $<
p1: graph.c
	$(CC) $(CFLAGS) -DIMPL=1 -o $@ $<
p2: graph.c
	$(CC) $(CFLAGS) -DIMPL=2 -o $@ $<
p3: graph.c
	$(CC) $(CFLAGS) -DIMPL=3 -o $@ $<
p4: graph.c
	$(CC) $(CFLAGS) -DIMPL=4 -o $@ $<
p5: graph.c
	$(CC) $(CFLAGS) -DIMPL=5 -o $@ $<
p6: graph.c
	$(CC) $(CFLAGS) -DIMPL=6 -o $@ $<

gen_ops: gen_ops.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS) gen_ops
