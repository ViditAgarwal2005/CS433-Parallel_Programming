#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_FIND 2

#define EMPTY (-1)	// never-used slot
#define TOMB (-2)	// deleted slot (keeps probe chains intact)

#define DEFAULT_THREADS_PER_BLOCK 256

__device__ unsigned long long result_sum;	
__device__ unsigned long long next_op;		


__device__ int hash_insert (int *A, int n, int k)
{
	int i = k % n, p;
	for (p=0; p<n; p++) {
		int v = atomicCAS(&A[i], EMPTY, EMPTY);
		if (v == k) return 0;			
		if (v == EMPTY) {
			int old = atomicCAS(&A[i], EMPTY, k);
			if (old == EMPTY) return 1;	
			if (old == k) return 0;		
		}
		i++;
		if (i == n) i = 0;
	}
	return 0;	
}

__device__ int hash_delete (int *A, int n, int k)
{
	int i = k % n, p;
	for (p=0; p<n; p++) {
		int v = atomicCAS(&A[i], EMPTY, EMPTY);
		if (v == k) return (atomicCAS(&A[i], k, TOMB) == k);
		if (v == EMPTY) return 0;
		i++;
		if (i == n) i = 0;
	}
	return 0;
}

__device__ int hash_find (int *A, int n, int k)
{
	int i = k % n, p;
	for (p=0; p<n; p++) {
		int v = atomicCAS(&A[i], EMPTY, EMPTY);
		if (v == k) return 1;
		if (v == EMPTY) return 0;
		i++;
		if (i == n) i = 0;
	}
	return 0;
}

__device__ int do_op (int *A, int n, int op, int k)
{
	if (op == OP_INSERT) return hash_insert(A, n, k);
	else if (op == OP_DELETE) return hash_delete(A, n, k);
	return hash_find(A, n, k);
}

__device__ void reduce_and_add (unsigned long long local)
{
	__syncwarp();	
	for (int i=warpSize/2; i>0; i=i/2) local += __shfl_down_sync(0xffffffff, local, i);
	if ((threadIdx.x % warpSize) == 0) atomicAdd(&result_sum, local);
}


__global__ void p1_static_kernel (int *A, int n, int op, int *keys, int cnt)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int nthreads = gridDim.x*blockDim.x;
	unsigned long long local = 0;
	int i;

	for (i=id; i<cnt; i+=nthreads) local += do_op(A, n, op, keys[i]);
	reduce_and_add(local);
}


__global__ void p1_dynamic_kernel (int *A, int n, int op, int *keys, int cnt, int chunk)
{
	unsigned long long local = 0;

	while (1) {
		unsigned long long base = atomicAdd(&next_op, (unsigned long long)chunk);
		if (base >= (unsigned long long)cnt) break;
		unsigned long long end = base + chunk;
		if (end > (unsigned long long)cnt) end = cnt;
		for (unsigned long long j=base; j<end; j++) local += do_op(A, n, op, keys[j]);
	}
	reduce_and_add(local);
}


__device__ int group_op (int *A, int n, int op, int k, int lane, int glane, unsigned gmask)
{
	long long off, base = k % n;
	int g = __popc(gmask); 

	for (off=0; off<n; off+=g) {
		int retry;
		do {
			retry = 0;
			int pos = (int)((base + off + glane) % n);
			int v = ((off + glane) < n) ? atomicCAS(&A[pos], EMPTY, EMPTY) : TOMB;

			int first_lane = -1;
			int first_value = TOMB;
			for (int t=0; t<g; t++) {
				int source_lane = (lane - glane) + t;
				int candidate = __shfl_sync(gmask, v, source_lane);
				if (first_lane == -1 && (candidate == k || candidate == EMPTY)) {
					first_lane = source_lane;
					first_value = candidate;
				}
			}

			if (first_lane == -1) break; 

			int is_key = (first_value == k);
			if (op == OP_FIND) return is_key;
			if (op == OP_DELETE) {
				if (!is_key) return 0; 
				int old = 0;
				if (lane == first_lane) {
					int target_glane = first_lane - (lane - glane);
					int target_pos = (int)((base + off + target_glane) % n);
					old = atomicCAS(&A[target_pos], k, TOMB);
				}
				old = __shfl_sync(gmask, old, first_lane);
				return (old == k);
			}

			
			if (is_key) return 0; 
			int old = 0;
			if (lane == first_lane) {
				int target_glane = first_lane - (lane - glane);
				int target_pos = (int)((base + off + target_glane) % n);
				old = atomicCAS(&A[target_pos], EMPTY, k);
			}
			old = __shfl_sync(gmask, old, first_lane);
			if (old == EMPTY) return 1;
			if (old == k) return 0;
			
			retry = 1;
		} while (retry);
	}
	return 0;
}

__global__ void p2_kernel (int *A, int n, int op, int *keys, int cnt, int g)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int nthreads = gridDim.x*blockDim.x;
	int lane = threadIdx.x % warpSize;
	int glane = lane % g;		
	int gbase = lane - glane;	
	unsigned gmask = (g == 32) ? 0xffffffffu : (((1u << g) - 1u) << gbase);
	int gid = id/g;			
	int ngroups = nthreads/g;
	unsigned long long local = 0;
	int i;

	for (i=gid; i<cnt; i+=ngroups) {
		int r = group_op(A, n, op, keys[i], lane, glane, gmask);
		if (glane == 0) local += r;
	}

	__syncwarp();
	for (i=warpSize/2; i>0; i=i/2) local += __shfl_down_sync(0xffffffff, local, i);
	if (lane == 0) atomicAdd(&result_sum, local);
}

int cmp_int (const void *a, const void *b)
{
	int x = *(const int*)a, y = *(const int*)b;
	return (x > y) - (x < y);
}


int op_count, n, *h_op, *h_key, *d_key, *d_A;

typedef struct {
	int version;		
	int mode;		
	int chunk;		
	int group;		
	int nthreads, tpb;
} config_t;

long run_version (config_t *c, int *ok)
{
	int i, j, nblocks = c->nthreads/c->tpb;
	unsigned long long expected, zero = 0, h_result = 0;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;
	long total_us = 0;
	const char *name[] = {"insert", "delete", "search"};

	cudaMemset(d_A, 0xFF, sizeof(int)*(size_t)n);	
	cudaDeviceSynchronize();

	i = 0;
	while (i < op_count) {
		int cnt;
		long us;
		j = i;
		while (j < op_count && h_op[j] == h_op[i]) j++;
		cnt = j - i;

		cudaMemcpyToSymbol(result_sum, &zero, sizeof(zero));
		cudaMemcpyToSymbol(next_op, &zero, sizeof(zero));
		cudaDeviceSynchronize();

		gettimeofday(&tv0, &tz0);
		if (c->version == 1) {
			if (c->mode == 0) p1_static_kernel<<<nblocks, c->tpb>>>(d_A, n, h_op[i], d_key + i, cnt);
			else p1_dynamic_kernel<<<nblocks, c->tpb>>>(d_A, n, h_op[i], d_key + i, cnt, c->chunk);
		}
		else p2_kernel<<<nblocks, c->tpb>>>(d_A, n, h_op[i], d_key + i, cnt, c->group);
		cudaDeviceSynchronize();
		gettimeofday(&tv1, &tz1);

		cudaMemcpyFromSymbol(&h_result, result_sum, sizeof(h_result));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); }

		expected = (h_op[i] == OP_INSERT) ? (unsigned long long)n : (unsigned long long)cnt;
		if (h_result != expected) *ok = 0;
		us = (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec);
		printf("  Phase %s: ops=%d, successes=%llu (expected %llu), time: %ld microseconds\n", name[h_op[i]], cnt, h_result, expected, us);
		total_us += us;
		i = j;
	}
	return total_us;
}

int main (int argc, char *argv[])
{
	int i, j, op, key, num_ins = 0, ok1 = 1, ok2 = 1, *tmp;
	FILE *fp;
	config_t c1, c2;
	long t1, t2;

	if (argc < 4) {
		printf ("Usage: %s op_file num_ops num_threads [p1_mode(0=static,1=dynamic) [p1_chunk [p1_tpb [p2_group [p2_tpb [p2_num_threads]]]]]]\n", argv[0]);
		exit(1);
	}
	op_count = atoi(argv[2]);

	c1.version = 1;
	c1.nthreads = atoi(argv[3]);
	c1.mode = (argc > 4) ? atoi(argv[4]) : 0;
	c1.chunk = (argc > 5) ? atoi(argv[5]) : 16;
	c1.tpb = (argc > 6) ? atoi(argv[6]) : DEFAULT_THREADS_PER_BLOCK;
	c1.group = 0;

	c2.version = 2;
	c2.group = (argc > 7) ? atoi(argv[7]) : 32;
	c2.tpb = (argc > 8) ? atoi(argv[8]) : DEFAULT_THREADS_PER_BLOCK;
	c2.nthreads = (argc > 9) ? atoi(argv[9]) : c1.nthreads;	
	c2.mode = 0; c2.chunk = 0;

	if (c1.nthreads < c1.tpb) c1.tpb = c1.nthreads;
	if (c2.nthreads < c2.tpb) c2.tpb = c2.nthreads;
	assert((c1.tpb % 32) == 0 && (c1.nthreads % c1.tpb) == 0 && c1.chunk >= 1);
	assert((c2.tpb % 32) == 0 && (c2.nthreads % c2.tpb) == 0);
	assert(c2.group >= 1 && c2.group <= 32 && (c2.group & (c2.group-1)) == 0);

	h_op = (int*)malloc(op_count*sizeof(int));
	h_key = (int*)malloc(op_count*sizeof(int));
	assert(h_op != NULL && h_key != NULL);
	fp = fopen(argv[1], "r");
	assert(fp != NULL);
	for (i=0; i<op_count; i++) {
		if (fscanf(fp, "%d %d", &op, &key) != 2) { printf("Op file too short\n"); exit(1); }
		h_op[i] = op;
		h_key[i] = key;
		if (op == OP_INSERT) num_ins++;
	}
	fclose(fp);

	tmp = (int*)malloc(num_ins*sizeof(int));
	for (i=0, j=0; i<op_count; i++) if (h_op[i] == OP_INSERT) tmp[j++] = h_key[i];
	qsort(tmp, num_ins, sizeof(int), cmp_int);
	n = 0;
	for (i=0; i<num_ins; i++) if (i == 0 || tmp[i] != tmp[i-1]) n++;
	free(tmp);
	assert(n > 0);

	cudaMalloc((void**)&d_A, sizeof(int)*(size_t)n);
	cudaMalloc((void**)&d_key, sizeof(int)*(size_t)op_count);
	cudaMemcpy(d_key, h_key, sizeof(int)*(size_t)op_count, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	printf("Table size n=%d, ops=%d\n", n, op_count);

	printf("P1 (%s, nthreads=%d, tpb=%d, chunk=%d):\n", c1.mode ? "dynamic" : "static", c1.nthreads, c1.tpb, c1.mode ? c1.chunk : 0);
	t1 = run_version(&c1, &ok1);
	printf("P1 total: %ld microseconds, throughput: %.2f ops per second, check: %s\n", t1, ((double)op_count*1000000.0)/t1, ok1 ? "PASS" : "FAIL");

	printf("P2 (group=%d, nthreads=%d, tpb=%d):\n", c2.group, c2.nthreads, c2.tpb);
	t2 = run_version(&c2, &ok2);
	printf("P2 total: %ld microseconds, throughput: %.2f ops per second, check: %s\n", t2, ((double)op_count*1000000.0)/t2, ok2 ? "PASS" : "FAIL");

	printf("Speedup of P2 over P1: %.2fx\n", (double)t1/t2);
	printf("RESULT ops=%d p1_us=%ld p2_us=%ld p1_tput=%.2f p2_tput=%.2f\n", op_count, t1, t2,
		((double)op_count*1000000.0)/t1, ((double)op_count*1000000.0)/t2);

	cudaFree(d_A);
	cudaFree(d_key);
	return 0;
}
