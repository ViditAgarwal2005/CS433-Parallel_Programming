
# CS433: Parallel Programming

### Group Members

| Name | Roll Number |
|---|---|
| Akshay Reddy Kamatam | 230517 |
| Srijani Gadupudi | 231033 |
| Vidit Agarwal | 231148 |

---

## Assignment 1: Shared Memory Parallel Programming (OpenMP)

### Question 1: Chromatic Graph Coloring

**Description:** Find the minimum number of colors required to color the vertices and edges of an undirected graph, ensuring that adjacent vertices, incident vertex-edge pairs, and edges belonging to the same triangle receive distinct colors.

**Optimizations:**
- Replaced brute-force enumeration with backtracking and pruning, using highest-degree-first vertex ordering, 64-bit bitmasks for forbidden colors, and symmetry breaking.
- Parallelized the search by generating valid partial colorings and dynamically distributing independent search subtrees among OpenMP threads, with atomic early termination.
- Used direct coloring of the original graph and a modular arithmetic formula to color edges efficiently when the chromatic number is odd, avoiding an expensive search on the conflict graph.

### Question 2: Hourglass System of Equations

**Description:** Solve a system of linear equations \(Ax=y\), where \(A\) is a structured hourglass matrix of dimension \((2n+1)\times(2n+1)\), exploiting its symmetric sparsity pattern.

**Optimizations:**
- Exploited the hourglass structure to solve the system from the center outwards using successive 2 × 2 systems, storing only non-zero matrix elements to reduce memory usage.
- Parallelized the independent outer-row updates using OpenMP while processing the dependent inner triangular region sequentially.
- Introduced chunk-based elimination (chunk size = 16) to balance sequential work, parallelism, synchronization overhead, and cache efficiency. Reused a single OpenMP thread team across all iterations.

---

## Assignment 2: Synchronization and Concurrent Data Structures (POSIX Threads)

### Question 1: Lock-Free FIFO Semaphore and Multi-Producer Single-Consumer

**Description:** Implement a lock-free FIFO semaphore using only the `cmpxchg` instruction and evaluate it through a multi-producer single-consumer (MPSC) bounded-buffer program. Compare four implementations: P1 (lock-free semaphore), P2 (busy-waiting lock-based semaphore), P3 (Linux semaphore), and P4 (condition variables).

**Optimizations:**
- Implemented the lock-free FIFO semaphore using atomic fetch-and-add built on CompareAndSet (CAS), with separate ticket and grant counters to allow multiple waiters to proceed concurrently while preserving FIFO ordering.
- Used cache-line alignment and padding to eliminate false sharing between frequently accessed synchronization variables and buffer slots.
- Used lightweight CPU pause instructions while busy-waiting and separated synchronization from data publication to reduce unnecessary cache-coherence traffic.

### Question 2: Concurrent Directed Graph

**Description:** Implement a concurrent directed graph using an array of linked lists, supporting edge insertion, deletion, and search. Compare six synchronization strategies: P1 (global mutex), P2 (per-list mutex), P3 (per-node mutex), P4 (per-node read-write lock), P5 (per-node custom CAS lock), and P6 (optimistic locking with custom CAS locks).

**Optimizations:**
- Progressively reduced lock contention by moving from a global graph lock to per-list and per-node locking, using hand-over-hand locking to ensure safe concurrent traversal and modification.
- Implemented per-node read-write locks to allow concurrent readers and custom CAS-based spin locks with test-and-test-and-set and exponential backoff.
- Implemented optimistic locking (P6), allowing traversal with reduced locking overhead, followed by lock acquisition and revalidation before modifications.
- Used sorted linked lists, sentinel nodes, cache-line-aligned list headers, and preallocated node pools to improve traversal efficiency and reduce allocation overhead.

---

## Assignment 3: GPU Programming and CUDA Optimization

### Question 1: Concurrent Open-Addressed Hash Table

**Description:** Implement a concurrent GPU-based open-addressed hash table using CUDA, supporting insertion, deletion, and search with linear probing and atomic operations. Compare P1 (one thread per operation) and P2 (one thread group per operation).

**Optimizations:**
- Implemented atomic slot operations using `atomicCAS`, along with EMPTY and TOMBSTONE markers to ensure correctness during concurrent insertions and deletions.
- In P1, evaluated static and dynamic task assignment and optimized the dynamic chunk size to improve workload distribution.
- In P2, assigned groups of threads within a warp to individual operations, enabling cooperative probing of consecutive hash-table locations and coalesced global memory accesses.
- Used warp shuffle instructions to identify the first decisive slot and broadcast operation results without requiring shared memory or block-level synchronization.
- Optimized the thread-group size, thread block size, and number of blocks to improve GPU utilization.

### Question 2: GPU Acceleration and Optimization of \(x^T Ay\)

**Description:** Compute the scalar quadratic form \(x^T Ay\), where \(A\) is an \(m\times n\) matrix and \(x\) and \(y\) are vectors, using CUDA. Compare GPU implementations with an optimized OpenMP CPU baseline.

**Optimizations:**
- **Version 1 – Basic CUDA:** Divided the matrix into contiguous chunks, assigning each thread a chunk of elements and accumulating partial results.
- **Version 2 – Memory Coalescing:** Reorganized thread-to-data mapping so consecutive threads access consecutive matrix columns, improving global memory access coalescing.
- **Version 3 – Tree-Based Reduction:** Introduced warp-level tree reduction using `__shfl_down_sync` to combine partial sums and reduce global atomic operations by a factor of 32.
- **Version 4 – Shared Memory:** Introduced shared memory tiling to cooperatively load vector elements, reduce redundant global memory accesses, and improve memory hierarchy utilization.
- Optimized GPU thread hierarchy and configuration, and compared the best CUDA implementation against an OpenMP baseline using static scheduling and SIMD vectorization.

---

## Technologies Used

- **Languages:** C, C++, CUDA
- **Parallel Programming:** OpenMP, POSIX Threads
- **GPU Programming:** CUDA Runtime API, Warp-Level Primitives
- **Synchronization:** CAS (`cmpxchg`), Mutexes, Read-Write Locks, Semaphores, Condition Variables
- **Performance Optimization:** Cache-Line Padding, Memory Coalescing, Tree-Based Reduction, Shared Memory Tiling, SIMD Vectorization
