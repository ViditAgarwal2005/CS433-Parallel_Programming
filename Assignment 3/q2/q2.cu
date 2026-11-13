#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 256
#define WARMUP 1

// OpenMP CPU version with static row assignment and SIMD reduction
double xtay_cpu (float *a, float *x, float *y, int m, int n, int t1)
{
	long long i, j;
	double total = 0.0, rowsum;

#pragma omp parallel for num_threads(t1) schedule(static) reduction(+:total)
	for (i=0; i<m; i++) {
		rowsum = 0.0;
#pragma omp simd reduction(+:rowsum)
		for (j=0; j<n; j++) {
			rowsum += (double)a[i*n + j]*(double)y[j];
		}
		total += (double)x[i]*rowsum;
	}
	return total;
}

// Initialization kernel (excluded from timing)
__global__ void init_kernel (float *a, long long size)
{
	long long id = threadIdx.x + (long long)blockIdx.x*blockDim.x;

	if (id < size) a[id] = (float)id/size;
}

// Warp-level tree reduction using shuffle instructions
__device__ float warp_reduce (float v)
{
	int w = (blockDim.x < warpSize) ? blockDim.x : warpSize;
	unsigned mask = 0xffffffffu >> (32 - w);
	int delta;

	for (delta=w/2; delta>0; delta>>=1) {
		v += __shfl_down_sync(mask, v, delta);
	}
	return v;
}

// Version 1: Basic kernel (contiguous chunk per thread)
__global__ void xtay_basic_kernel (float *a, float *x, float *y, float *res, int m, int n, long long chunk)
{
	long long tid = threadIdx.x + (long long)blockIdx.x*blockDim.x;
	long long e = tid*chunk;
	long long end = e + chunk;
	int i = (int)(e/n);
	int j = (int)(e%n);
	float rowsum = 0, total = 0;

	for (; e<end; e++) {
		rowsum += a[e]*y[j];
		j++;
		if (j == n) {
			total += x[i]*rowsum;
			rowsum = 0;
			j = 0;
			i++;
		}
	}
	if (j != 0) total += x[i]*rowsum;

	atomicAdd(res, total);
}

// Version 2: Memory access coalescing
__global__ void xtay_coalesced_kernel (float *a, float *x, float *y, float *res, int m, int n)
{
	int t = gridDim.x*blockDim.x;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int i_step = (t > n) ? t/n : 1;
	int j_step = (t > n) ? n : t;
	int i0 = tid/n;
	int j0 = tid%n;
	int i, j;
	float colsum, total = 0;

	for (j=j0; j<n; j+=j_step) {
		colsum = 0;
		for (i=i0; i<m; i+=i_step) {
			colsum += x[i]*a[(long long)i*n + j];
		}
		total += y[j]*colsum;
	}

	atomicAdd(res, total);
}

// Version 3: Coalescing + tree reduction
__global__ void xtay_tree_kernel (float *a, float *x, float *y, float *res, int m, int n)
{
	int t = gridDim.x*blockDim.x;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int i_step = (t > n) ? t/n : 1;
	int j_step = (t > n) ? n : t;
	int i0 = tid/n;
	int j0 = tid%n;
	int i, j;
	float colsum, total = 0;

	for (j=j0; j<n; j+=j_step) {
		colsum = 0;
		for (i=i0; i<m; i+=i_step) {
			colsum += x[i]*a[(long long)i*n + j];
		}
		total += y[j]*colsum;
	}

	total = warp_reduce(total);
	if (threadIdx.x % warpSize == 0) atomicAdd(res, total);
}

// Version 4: Coalescing + tree reduction + shared memory
__global__ void xtay_shmem_kernel (float *a, float *x, float *y, float *res, int m, int n)
{
	int t = gridDim.x*blockDim.x;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int i_step = (t > n) ? t/n : 1;
	int j_step = (t > n) ? n : t;
	int i0 = tid/n;
	int j0 = tid%n;
	int rows_per_thread = m/i_step;
	int j, k, r, tile;
	float colsum, total = 0;
	__shared__ float xs[THREADS_PER_BLOCK];

	for (j=j0; j<n; j+=j_step) {
		colsum = 0;
		for (k=0; k<rows_per_thread; k+=blockDim.x) {
			tile = rows_per_thread - k;
			if (tile > blockDim.x) tile = blockDim.x;
			if (threadIdx.x < tile) xs[threadIdx.x] = x[i0 + (k + threadIdx.x)*i_step];
			__syncthreads();
			for (r=0; r<tile; r++) {
				colsum += xs[r]*a[(long long)(i0 + (k + r)*i_step)*n + j];
			}
			__syncthreads();
		}
		total += y[j]*colsum;
	}

	total = warp_reduce(total);
	if (threadIdx.x % warpSize == 0) atomicAdd(res, total);
}

void launch (int v, int blocks, int tpb, float *a, float *x, float *y, float *res, int m, int n, long long t2)
{
	if (v == 1) xtay_basic_kernel<<<blocks, tpb>>>(a, x, y, res, m, n, ((long long)m*n)/t2);
	else if (v == 2) xtay_coalesced_kernel<<<blocks, tpb>>>(a, x, y, res, m, n);
	else if (v == 3) xtay_tree_kernel<<<blocks, tpb>>>(a, x, y, res, m, n);
	else xtay_shmem_kernel<<<blocks, tpb>>>(a, x, y, res, m, n);
}

int main (int argc, char *argv[])
{
	int v, m, n, t1, tpb, blocks, best_v = 1;
	long long t2, sizeA;
	float *a, *x, *y, *res;
	float gpu_val[5];
	long gpu_time[5], cpu_time, best_gpu_time;
	double cpu_val, err, rel_err, speedup;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;
	const char *v_names[5] = {"", "basic", "coalesced", "tree reduction", "shared memory"};

	if (argc != 5) {
		printf("Need m, n, t1 (CPU threads), and t2 (GPU threads).\nAborting...\n");
		exit(1);
	}

	m = atoi(argv[1]);
	n = atoi(argv[2]);
	t1 = atoi(argv[3]);
	t2 = atoll(argv[4]);
	sizeA = (long long)m*n;

	// Check power of two constraints
	assert(m > 0 && (m & (m - 1)) == 0);
	assert(n > 0 && (n & (n - 1)) == 0);
	assert(t1 > 0 && (t1 & (t1 - 1)) == 0);
	assert(t2 > 0 && (t2 & (t2 - 1)) == 0);

	if (t2 > sizeA) t2 = sizeA;

	omp_set_num_threads(t1);

	// Configure 1D thread blocks
	tpb = THREADS_PER_BLOCK;
	if (tpb > t2) tpb = (int)t2;
	if (tpb > n) tpb = n;
	blocks = (int)(t2/tpb);

	// Check if arrays fit in GPU memory
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	assert(sizeof(float)*(sizeA + m + n + 1) <= free_mem);

	cudaMallocManaged((void**)&a, sizeof(float)*sizeA);
	cudaMallocManaged((void**)&x, sizeof(float)*m);
	cudaMallocManaged((void**)&y, sizeof(float)*n);
	cudaMallocManaged((void**)&res, sizeof(float));

	int device = -1;
	cudaGetDevice(&device);

	// Advise and prefetch unified memory to GPU
	cudaMemAdvise(a, sizeof(float)*sizeA, cudaMemAdviseSetPreferredLocation, device);
	cudaMemAdvise(a, sizeof(float)*sizeA, cudaMemAdviseSetAccessedBy, device);
	cudaMemPrefetchAsync(a, sizeof(float)*sizeA, device, 0);

	cudaMemAdvise(x, sizeof(float)*m, cudaMemAdviseSetPreferredLocation, device);
	cudaMemAdvise(x, sizeof(float)*m, cudaMemAdviseSetAccessedBy, device);
	cudaMemPrefetchAsync(x, sizeof(float)*m, device, 0);

	cudaMemAdvise(y, sizeof(float)*n, cudaMemAdviseSetPreferredLocation, device);
	cudaMemAdvise(y, sizeof(float)*n, cudaMemAdviseSetAccessedBy, device);
	cudaMemPrefetchAsync(y, sizeof(float)*n, device, 0);

	cudaMemAdvise(res, sizeof(float), cudaMemAdviseSetPreferredLocation, device);
	cudaDeviceSynchronize();

	// Initialize arrays on GPU
	init_kernel<<<(sizeA + 1023)/1024, 1024>>>(a, sizeA);
	init_kernel<<<(m + 1023)/1024, 1024>>>(x, m);
	init_kernel<<<(n + 1023)/1024, 1024>>>(y, n);
	cudaDeviceSynchronize();

	cudaMemAdvise(a, sizeof(float)*sizeA, cudaMemAdviseSetReadMostly, device);
	cudaMemAdvise(x, sizeof(float)*m, cudaMemAdviseSetReadMostly, device);
	cudaMemAdvise(y, sizeof(float)*n, cudaMemAdviseSetReadMostly, device);
	cudaDeviceSynchronize();

	// Run and time all 4 CUDA versions
	for (v=1; v<=4; v++) {
		if (WARMUP) {
			cudaMemset(res, 0, sizeof(float));
			cudaDeviceSynchronize();
			launch(v, blocks, tpb, a, x, y, res, m, n, t2);
			cudaDeviceSynchronize();
		}
		cudaMemset(res, 0, sizeof(float));
		cudaDeviceSynchronize();

		gettimeofday(&tv0, &tz0);
		launch(v, blocks, tpb, a, x, y, res, m, n, t2);
		cudaDeviceSynchronize();
		gettimeofday(&tv1, &tz1);

		gpu_val[v] = *res;
		gpu_time[v] = (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec);
	}

	// Run and time OpenMP CPU version
	if (WARMUP) {
		cpu_val = xtay_cpu(a, x, y, m, n, t1);
	}

	gettimeofday(&tv0, &tz0);
	cpu_val = xtay_cpu(a, x, y, m, n, t1);
	gettimeofday(&tv1, &tz1);
	cpu_time = (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec);

	printf("m: %d\n", m);
	printf("n: %d\n", n);
	printf("t1 (CPU threads): %d\n", t1);
	printf("t2 (GPU threads): %lld\n", t2);
	printf("Blocks: %d\n", blocks);
	printf("Threads per block: %d\n\n", tpb);

	printf("CPU value: %f\n", cpu_val);
	printf("CPU time: %ld microseconds\n\n", cpu_time);

	best_gpu_time = gpu_time[1];
	for (v=1; v<=4; v++) {
		err = fabs((double)gpu_val[v] - cpu_val);
		rel_err = (cpu_val != 0.0) ? (err/fabs(cpu_val))*100.0 : 0.0;
		printf("Version %d (%s):\n", v, v_names[v]);
		printf("GPU value: %f\n", gpu_val[v]);
		printf("CPU value: %f\n", cpu_val);
		printf("Absolute error: %f\n", err);
		printf("Relative error: %.6f%%\n", rel_err);
		printf("Time: %ld microseconds\n\n", gpu_time[v]);
		if (gpu_time[v] < best_gpu_time) {
			best_gpu_time = gpu_time[v];
			best_v = v;
		}
	}

	speedup = (double)cpu_time/best_gpu_time;
	printf("Best GPU version: %d (%s)\n", best_v, v_names[best_v]);
	printf("Best GPU time: %ld microseconds\n", best_gpu_time);
	printf("Speedup: %.4fx\n", speedup);

	cudaFree(a);
	cudaFree(x);
	cudaFree(y);
	cudaFree(res);
	return 0;
}