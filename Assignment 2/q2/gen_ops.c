/*
 * CS433 Assignment 2, Problem 2: operation-list generator.
 *
 * Usage: ./gen_ops <out_file> <op_count> <insert%> <delete%> [n] [m] [seed]
 *        search% = 100 - insert% - delete%
 *        n, m, seed describe the initial graph (default 10000 2000000 433)
 *
 * The first line of the file is "n m seed"; the programs build the initial
 * graph from it, so all files written with the same n, m, seed start from
 * the same graph.  Every following line is "op u v" with op 0 = insert,
 * 1 = delete, 2 = search and u != v drawn uniformly from [0, n).
 *
 * Uniform mixing: operation types are emitted in blocks of 100 that hold
 * exactly insert%/delete%/search% operations of each type, shuffled within
 * the block.  So the ratio is exact for every prefix whose length is a
 * multiple of 100 (one file can serve several operation counts) and the
 * types are mixed evenly over the whole list, unlike drawing until a type
 * runs out, which leaves the tail of the list dominated by one type.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_FIND   2

static uint64_t rng;

static uint64_t splitmix64(void)
{
	uint64_t z = (rng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

int main(int argc, char *argv[])
{
	long op_count, i, n = 10000, m = 2000000, seed = 433;
	int ins, del, block[100], j, k, t, u, v;
	FILE *fp;

	if (argc < 5 || argc > 8) {
		fprintf(stderr, "Usage: %s <out_file> <op_count> <insert%%> <delete%%> [n] [m] [seed]\n", argv[0]);
		return 1;
	}
	op_count = atol(argv[2]);
	ins = atoi(argv[3]);
	del = atoi(argv[4]);
	if (argc > 5) n = atol(argv[5]);
	if (argc > 6) m = atol(argv[6]);
	if (argc > 7) seed = atol(argv[7]);

	if (op_count < 1 || ins < 0 || del < 0 || ins + del > 100) {
		fprintf(stderr, "Need op_count >= 1 and insert%% + delete%% <= 100\n");
		return 1;
	}
	if (n < 2 || n > 40000 || m < 0 || m > n * (n - 1) / 2) {
		fprintf(stderr, "Need 2 <= n <= 40000 and 0 <= m <= n(n-1)/2 (initial graph must not be complete)\n");
		return 1;
	}

	fp = fopen(argv[1], "w");
	if (!fp) { perror(argv[1]); return 1; }
	setvbuf(fp, NULL, _IOFBF, 1 << 20);

	rng = (uint64_t)seed * 1000003ULL + (uint64_t)ins * 101ULL + (uint64_t)del;
	fprintf(fp, "%ld %ld %ld\n", n, m, seed);

	for (i = 0; i < op_count; i += 100) {
		for (j = 0; j < 100; j++)
			block[j] = (j < ins) ? OP_INSERT : (j < ins + del) ? OP_DELETE : OP_FIND;
		for (j = 99; j > 0; j--) {          /* Fisher-Yates shuffle of the block */
			k = (int)(splitmix64() % (uint64_t)(j + 1));
			t = block[j]; block[j] = block[k]; block[k] = t;
		}
		for (j = 0; j < 100 && i + j < op_count; j++) {
			u = (int)(splitmix64() % (uint64_t)n);
			do v = (int)(splitmix64() % (uint64_t)n); while (v == u);
			fprintf(fp, "%d %d %d\n", block[j], u, v);
		}
	}
	fclose(fp);
	return 0;
}
