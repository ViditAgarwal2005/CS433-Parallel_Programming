CC     = gcc
# -std=gnu11: newer gcc defaults to C23, which redirects atoi/atol to
# __isoc23_strtol (glibc >= 2.38 only); gnu11 keeps the binaries portable.
CFLAGS = -std=gnu11 -O2 -Wall -Wextra -pthread

all: p1 p2 p3 p4

p1: mpsc.c
	$(CC) $(CFLAGS) -DIMPL=1 -o $@ $<
p2: mpsc.c
	$(CC) $(CFLAGS) -DIMPL=2 -o $@ $<
p3: mpsc.c
	$(CC) $(CFLAGS) -DIMPL=3 -o $@ $<
p4: mpsc.c
	$(CC) $(CFLAGS) -DIMPL=4 -o $@ $<

clean:
	rm -f p1 p2 p3 p4
