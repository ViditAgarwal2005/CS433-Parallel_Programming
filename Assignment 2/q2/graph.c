/*
 * CS433 Assignment 2, Problem 2
 * Concurrent directed graph stored as an array of sorted singly linked
 * adjacency lists (list #u holds every v such that (u,v) is an edge).
 * Supports insertion, deletion and search of an edge.  The synchronisation
 * scheme is selected at compile time with -DIMPL=<0..6>:
 *
 *   IMPL=0  SEQ: sequential baseline, no synchronisation (always 1 thread)
 *   IMPL=1  P1 : one POSIX mutex for the entire graph
 *   IMPL=2  P2 : one POSIX mutex per adjacency list
 *   IMPL=3  P3 : one POSIX mutex per list node (hand-over-hand locking)
 *   IMPL=4  P4 : one POSIX read/write lock per list node
 *   IMPL=5  P5 : one cmpxchg spin lock per list node (hand-over-hand locking)
 *   IMPL=6  P6 : optimistic locking, one cmpxchg spin lock per list node
 *
 * Usage: ./pX <op_file> <num_ops> <num_threads>
 *
 * op_file: one "op u v" line per operation, op 0 = insert, 1 = delete,
 *   2 = search (u,v).  The initial graph has DEFAULT_NV vertices and
 *   DEFAULT_NE distinct random edges generated deterministically from
 *   DEFAULT_SEED, so every program and every experiment starts from the same
 *   graph (n grows to max vertex id + 1 if the file uses larger ids).
 *   An optional first line "n m seed" overrides these defaults.
 *
 * Common optimisations (all versions, see the report for the discussion):
 *   - lists are kept sorted so every operation stops early;
 *   - every list starts with a sentinel head node and ends at one shared,
 *     immutable tail sentinel (v = INT_MAX), which removes all head/NULL
 *     special cases (and the race on 'head' in the per-node lock versions);
 *   - list headers are cache-line aligned (no false sharing between lists);
 *   - the initial graph's nodes are laid out contiguously, list by list;
 *   - nodes are taken from per-thread pools built before the timer starts,
 *     so no malloc/free (and no lock initialisation) happens in the timed region.
 */
/* POSIX only (barriers, clock_gettime). Not _GNU_SOURCE: on glibc >= 2.38 it
 * redirects atoi/atol to __isoc23_strtol and the binary then needs glibc 2.38. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include <time.h>

#ifndef IMPL
#define IMPL 1
#endif
#if IMPL < 0 || IMPL > 6
#error "IMPL must be 0..6"
#endif

#define CACHE_LINE 64

/* initial graph when the op file has no "n m seed" line (-D to override) */
#ifndef DEFAULT_NV
#define DEFAULT_NV   10000
#endif
#ifndef DEFAULT_NE
#define DEFAULT_NE   2000000L
#endif
#ifndef DEFAULT_SEED
#define DEFAULT_SEED 433ULL
#endif
#define MAX_NV       40000              /* build_graph uses an n*n-bit bitmap */

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_FIND   2

#define cpu_relax()    asm volatile("pause" ::: "memory")
#define compiler_bar() asm volatile("" ::: "memory")

/* ------------------------------------------------------------------ */
/* The CompareAndSet developed in class.  It has no memory clobber, so */
/* the lock functions below add explicit compiler barriers.            */
/* ------------------------------------------------------------------ */
unsigned char CompareAndSet(int oldVal, int newVal, int *ptr)
{
	int oldValOut;
	unsigned char result;
	asm("lock cmpxchgl %4, %1 \n setzb %0"
		:"=qm"(result),  "+m" (*ptr), "=a" (oldValOut)
		:"a" (oldVal),  "r" (newVal)
		: );
	return result;
}

#if IMPL == 5 || IMPL == 6
/* Test-and-test-and-set lock with exponential backoff (lecture 4).
 * Uncontended acquire = one cmpxchg.  Waiters spin on a plain load, which
 * hits in their own cache until the holder's release invalidates the line,
 * and back off exponentially after a failed cmpxchg. */
#define BACKOFF_MIN 4
#define BACKOFF_MAX 1024

static inline void spin_acquire(int *lock)
{
	unsigned delay = BACKOFF_MIN, i;

	if (CompareAndSet(0, 1, lock)) {
		compiler_bar();
		return;
	}
	for (;;) {
		while (*(volatile int *)lock) cpu_relax();
		if (CompareAndSet(0, 1, lock)) break;
		for (i = 0; i < delay; i++) cpu_relax();
		if (delay < BACKOFF_MAX) delay <<= 1;
	}
	compiler_bar();
}

static inline void spin_release(int *lock)
{
	compiler_bar();                 /* critical-section stores stay before the release */
	*(volatile int *)lock = 0;      /* x86 TSO: a plain store is a release */
}
#endif

/* ================================================================== */
/*                         Data structures                            */
/* ================================================================== */
#if IMPL == 6
/* traversals run without locks, so 'next' must be re-read from memory */
#define NEXT_T struct node_s *volatile
#else
#define NEXT_T struct node_s *
#endif

typedef struct node_s {
	int v;
#if IMPL == 3
	pthread_mutex_t lock;
#elif IMPL == 4
	pthread_rwlock_t lock;
#elif IMPL == 5 || IMPL == 6
	int lock;
#endif
#if IMPL == 6
	volatile int marked;            /* 1 once the node is logically deleted */
#endif
	NEXT_T next;
} node_t;

/* One adjacency list.  Aligned to a cache line so the head sentinels (and
 * per-list mutexes) of neighbouring lists never share a line. */
typedef struct list_s {
#if IMPL == 2
	pthread_mutex_t lock;
#endif
	node_t head;                    /* sentinel, v = INT_MIN, never removed */
} __attribute__((aligned(CACHE_LINE))) list_t;

typedef struct op_s {
	int op, u, v;
} op_t;

/* Per-thread state, one cache line apart. */
typedef struct tctx_s {
	int id;
	long lo, hi;                    /* this thread's slice of the op list */
	node_t *slab;                   /* pre-allocated node pool */
	long used, cap;
	node_t *free_list;              /* deleted nodes ready for reuse (not P6) */
	long ins_ok, del_ok, found;
} __attribute__((aligned(CACHE_LINE))) tctx_t;

static list_t *adj;
static node_t TAIL;                     /* shared tail sentinel, v = INT_MAX, immutable */
static node_t *init_nodes;
static int NV;
static long NE;
static unsigned long long SEED;

static op_t *ops;
static long op_count;
static int num_threads;

static pthread_barrier_t bar;
static struct timespec t0, t1;

#if IMPL == 1
static pthread_mutex_t graph_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

static void node_init(node_t *n, int v)
{
	n->v = v;
#if IMPL == 3
	pthread_mutex_init(&n->lock, NULL);
#elif IMPL == 4
	pthread_rwlock_init(&n->lock, NULL);
#elif IMPL == 5 || IMPL == 6
	n->lock = 0;
#endif
#if IMPL == 6
	n->marked = 0;
#endif
	n->next = NULL;
}

/* ------------------------------------------------------------------ */
/* Node pool.  Each thread pre-allocates (and pre-initialises the lock */
/* of) one node per insert in its slice, before the timer starts.      */
/* P0-P5 recycle deleted nodes: a node is unlinked while its           */
/* predecessor is held exclusively, so nobody else can still be on it. */
/* P6 never reuses a node: lock-free traversals may still be reading   */
/* a deleted node (lecture 4), so deleted nodes are only reclaimed at  */
/* exit, outside the timed region.                                     */
/* ------------------------------------------------------------------ */
static void pool_init(tctx_t *c)
{
	long i, cap = 1;

	for (i = c->lo; i < c->hi; i++)
		if (ops[i].op == OP_INSERT) cap++;
	c->slab = (node_t *)malloc(cap * sizeof(node_t));
	assert(c->slab != NULL);
	for (i = 0; i < cap; i++) node_init(&c->slab[i], 0);   /* also touches every page */
	c->cap = cap;
	c->used = 0;
	c->free_list = NULL;
}

static inline node_t *alloc_node(tctx_t *c)
{
#if IMPL != 6
	if (c->free_list) {
		node_t *n = c->free_list;
		c->free_list = n->next;
		return n;
	}
#endif
	assert(c->used < c->cap);
	return &c->slab[c->used++];
}

static inline void retire_node(tctx_t *c, node_t *n)
{
#if IMPL != 6
	n->next = c->free_list;
	c->free_list = n;
#else
	(void)c; (void)n;
#endif
}

/* ================================================================== */
/*                     List operations per version                    */
/* ================================================================== */
#if IMPL <= 2
/* Plain sorted-list operations; SEQ uses them directly, P1/P2 wrap them
 * in the graph / list mutex. */
static inline int ll_insert(node_t *head, int v, tctx_t *c)
{
	node_t *pred = head, *curr = head->next, *n;

	while (curr->v < v) {
		pred = curr;
		curr = curr->next;
	}
	if (curr->v == v) return 0;
	n = alloc_node(c);
	n->v = v;
	n->next = curr;
	pred->next = n;
	return 1;
}

static inline int ll_delete(node_t *head, int v, tctx_t *c)
{
	node_t *pred = head, *curr = head->next;

	while (curr->v < v) {
		pred = curr;
		curr = curr->next;
	}
	if (curr->v != v) return 0;
	pred->next = curr->next;
	retire_node(c, curr);
	return 1;
}

static inline int ll_find(node_t *head, int v)
{
	node_t *curr = head->next;

	while (curr->v < v) curr = curr->next;
	return curr->v == v;
}

#if IMPL == 0
#define GLOCK(L)
#define GUNLOCK(L)
#elif IMPL == 1
#define GLOCK(L)   pthread_mutex_lock(&graph_lock)
#define GUNLOCK(L) pthread_mutex_unlock(&graph_lock)
#else
#define GLOCK(L)   pthread_mutex_lock(&(L)->lock)
#define GUNLOCK(L) pthread_mutex_unlock(&(L)->lock)
#endif

static int graph_insert(list_t *L, int v, tctx_t *c)
{
	int r;
	GLOCK(L);
	r = ll_insert(&L->head, v, c);
	GUNLOCK(L);
	return r;
}

static int graph_delete(list_t *L, int v, tctx_t *c)
{
	int r;
	GLOCK(L);
	r = ll_delete(&L->head, v, c);
	GUNLOCK(L);
	return r;
}

static int graph_find(list_t *L, int v)
{
	int r;
	GLOCK(L);
	r = ll_find(&L->head, v);
	GUNLOCK(L);
	return r;
}

#elif IMPL == 3 || IMPL == 5
/* ------------------------------------------------------------------ *
 * P3 / P5: hand-over-hand (lock coupling) per-node locking.  The lock
 * of the next node is taken before the lock of the current node is
 * released, so two operations can never pass each other and a node is
 * never freed while somebody is inside it.  All threads take locks in
 * list order, so there is no deadlock.  The shared tail sentinel is
 * immutable and never removed, so it is never locked.
 * ------------------------------------------------------------------ */
#if IMPL == 3
#define NLOCK(n)   pthread_mutex_lock(&(n)->lock)
#define NUNLOCK(n) pthread_mutex_unlock(&(n)->lock)
#else
#define NLOCK(n)   spin_acquire(&(n)->lock)
#define NUNLOCK(n) spin_release(&(n)->lock)
#endif
#define NLOCK_T(n)   do { if ((n) != &TAIL) NLOCK(n); } while (0)
#define NUNLOCK_T(n) do { if ((n) != &TAIL) NUNLOCK(n); } while (0)

/* Returns with pred and curr locked, pred->v < v <= curr->v. */
static inline node_t *hoh_locate(list_t *L, int v, node_t **pcurr)
{
	node_t *pred = &L->head, *curr;

	NLOCK(pred);
	curr = pred->next;
	NLOCK_T(curr);
	while (curr->v < v) {
		NUNLOCK(pred);
		pred = curr;
		curr = curr->next;
		NLOCK_T(curr);
	}
	*pcurr = curr;
	return pred;
}

static int graph_insert(list_t *L, int v, tctx_t *c)
{
	node_t *curr, *n, *pred = hoh_locate(L, v, &curr);

	if (curr->v == v) {
		NUNLOCK_T(curr);
		NUNLOCK(pred);
		return 0;
	}
	/* pred's lock alone keeps curr in place (deleting curr or inserting
	 * before it both need pred's lock), so release curr early */
	NUNLOCK_T(curr);
	n = alloc_node(c);
	n->v = v;
	n->next = curr;
	pred->next = n;
	NUNLOCK(pred);
	return 1;
}

static int graph_delete(list_t *L, int v, tctx_t *c)
{
	node_t *curr, *pred = hoh_locate(L, v, &curr);

	if (curr->v != v) {
		NUNLOCK_T(curr);
		NUNLOCK(pred);
		return 0;
	}
	pred->next = curr->next;
	NUNLOCK(curr);
	NUNLOCK(pred);
	retire_node(c, curr);
	return 1;
}

static int graph_find(list_t *L, int v)
{
	node_t *pred = &L->head, *curr;
	int r;

	NLOCK(pred);
	curr = pred->next;
	NLOCK_T(curr);
	NUNLOCK(pred);
	while (curr->v < v) {
		pred = curr;
		curr = curr->next;
		NLOCK_T(curr);
		NUNLOCK(pred);
	}
	r = (curr->v == v);
	NUNLOCK_T(curr);
	return r;
}

#elif IMPL == 4
/* ------------------------------------------------------------------ *
 * P4: per-node POSIX read/write locks.
 * Searches couple READ locks hand-over-hand, so any number of them can
 * walk the same nodes together.  Inserts and deletes also traverse with
 * READ locks (instead of write-locking every node on the way) and only
 * the node that will be modified is write-locked:
 *   - while walking we keep the read lock of the node before 'pred'
 *     ('gp'); that blocks the removal of pred and insertions in front of
 *     it, so pred can safely be upgraded (read unlock -> write lock);
 *   - after the upgrade pred->next is re-checked; if a smaller key was
 *     inserted meanwhile we simply keep walking.
 * Locks are always requested in list order (the upgrade re-requests a
 * node while holding only nodes in front of it), so there is no deadlock.
 * ------------------------------------------------------------------ */
#define RD(n) do { if ((n) != &TAIL) pthread_rwlock_rdlock(&(n)->lock); } while (0)
#define WR(n) do { if ((n) != &TAIL) pthread_rwlock_wrlock(&(n)->lock); } while (0)
#define UN(n) do { if ((n) != &TAIL) pthread_rwlock_unlock(&(n)->lock); } while (0)

/* Returns pred write-locked with pred->v < v <= pred->next->v. */
static node_t *rw_locate(list_t *L, int v)
{
	node_t *gp = NULL, *pred = &L->head, *curr;

	RD(pred);
	for (;;) {
		curr = pred->next;
		if (curr->v < v) {
			RD(curr);
			if (gp) UN(gp);
			gp = pred;
			pred = curr;
			continue;
		}
		UN(pred);
		WR(pred);
		curr = pred->next;
		if (curr->v >= v) break;
		/* a smaller key was linked after pred during the upgrade */
		RD(curr);
		if (gp) UN(gp);
		gp = pred;
		pred = curr;
	}
	if (gp) UN(gp);
	return pred;
}

static int graph_insert(list_t *L, int v, tctx_t *c)
{
	node_t *pred = rw_locate(L, v), *curr = pred->next, *n;

	if (curr->v == v) {
		UN(pred);
		return 0;
	}
	n = alloc_node(c);
	n->v = v;
	n->next = curr;
	pred->next = n;
	UN(pred);
	return 1;
}

static int graph_delete(list_t *L, int v, tctx_t *c)
{
	node_t *pred = rw_locate(L, v), *curr = pred->next;

	if (curr->v != v) {
		UN(pred);
		return 0;
	}
	WR(curr);                       /* wait for readers still inside curr */
	pred->next = curr->next;
	UN(curr);
	UN(pred);
	retire_node(c, curr);
	return 1;
}

static int graph_find(list_t *L, int v)
{
	node_t *pred = &L->head, *curr;
	int r;

	RD(pred);
	curr = pred->next;
	RD(curr);
	UN(pred);
	while (curr->v < v) {
		pred = curr;
		curr = curr->next;
		RD(curr);
		UN(pred);
	}
	r = (curr->v == v);
	UN(curr);
	return r;
}

#elif IMPL == 6
/* ------------------------------------------------------------------ *
 * P6: optimistic locking (lecture 4) with a 'deleted' mark per node.
 *   - every operation first walks the list WITHOUT locks to find
 *     (pred, curr);
 *   - search takes no lock at all: found iff curr->v == v && !marked;
 *   - insert of a present edge and delete of an absent edge are decided
 *     exactly like a search, also without locks (they change nothing);
 *   - insert locks only pred: deleting curr or linking a node in front
 *     of it both need pred's lock, so pred's lock alone pins the window;
 *     validation: !pred->marked && pred->next == curr;
 *   - delete locks pred then curr (list order), validates
 *     !pred->marked && !curr->marked && pred->next == curr, sets the mark
 *     (linearisation point) and unlinks;
 *   - on a failed validation the walk restarts from pred if pred is
 *     still in the list, otherwise from the head;
 *   - deleted nodes are never freed or reused during the run, so a
 *     lock-free walker that stands on one still finds valid next
 *     pointers that lead back into the list.
 * ------------------------------------------------------------------ */
static int graph_insert(list_t *L, int v, tctx_t *c)
{
	node_t *start = &L->head, *pred, *curr, *n;
	int r;

	for (;;) {
		pred = start;
		curr = pred->next;
		while (curr->v < v) {
			pred = curr;
			curr = curr->next;
		}
		if (curr->v == v && !curr->marked) return 0;
		spin_acquire(&pred->lock);
		if (!pred->marked && pred->next == curr) {
			r = 0;
			if (curr->v != v) {
				n = alloc_node(c);
				n->v = v;
				n->next = curr;
				compiler_bar();         /* node is complete before it is published */
				pred->next = n;
				r = 1;
			}
			spin_release(&pred->lock);
			return r;
		}
		spin_release(&pred->lock);
		start = pred->marked ? &L->head : pred;
	}
}

static int graph_delete(list_t *L, int v, tctx_t *c)
{
	node_t *start = &L->head, *pred, *curr;

	for (;;) {
		pred = start;
		curr = pred->next;
		while (curr->v < v) {
			pred = curr;
			curr = curr->next;
		}
		if (curr->v != v || curr->marked) return 0;
		spin_acquire(&pred->lock);
		spin_acquire(&curr->lock);
		if (!pred->marked && !curr->marked && pred->next == curr) {
			curr->marked = 1;
			compiler_bar();
			pred->next = curr->next;
			spin_release(&curr->lock);
			spin_release(&pred->lock);
			retire_node(c, curr);
			return 1;
		}
		spin_release(&curr->lock);
		spin_release(&pred->lock);
		start = pred->marked ? &L->head : pred;
	}
}

static int graph_find(list_t *L, int v)
{
	node_t *curr = L->head.next;

	while (curr->v < v) curr = curr->next;
	return curr->v == v && !curr->marked;
}
#endif

/* ================================================================== */
/*                          Worker threads                            */
/* ================================================================== */
static void run_ops(tctx_t *c)
{
	long i, ins = 0, del = 0, fnd = 0;

	for (i = c->lo; i < c->hi; i++) {
		const op_t *o = &ops[i];
		list_t *L = &adj[o->u];

		if (o->op == OP_INSERT) ins += graph_insert(L, o->v, c);
		else if (o->op == OP_DELETE) del += graph_delete(L, o->v, c);
		else fnd += graph_find(L, o->v);
	}
	c->ins_ok = ins;
	c->del_ok = del;
	c->found = fnd;
}

static void *worker(void *param)
{
	tctx_t *c = (tctx_t *)param;

	pool_init(c);                           /* not timed */
	pthread_barrier_wait(&bar);             /* everybody is ready */
	if (c->id == 0) clock_gettime(CLOCK_MONOTONIC, &t0);
	pthread_barrier_wait(&bar);             /* start together */
	run_ops(c);
	pthread_barrier_wait(&bar);             /* everybody is done */
	if (c->id == 0) clock_gettime(CLOCK_MONOTONIC, &t1);
	return NULL;
}

/* ================================================================== */
/*                     Set-up, verification, main                     */
/* ================================================================== */
static uint64_t splitmix64(uint64_t *s)
{
	uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static uint64_t edge_hash(int u, int v)
{
	uint64_t s = ((uint64_t)(unsigned)u << 32) | (unsigned)v;
	return splitmix64(&s);
}

/* Initial graph: m distinct random edges (u != v) from a fixed seed, so it
 * is identical for every program and every run.  Each list's nodes are
 * allocated contiguously and in sorted order. */
static void build_graph(void)
{
	uint64_t words = ((uint64_t)NV * NV + 63) / 64, s = SEED, idx;
	uint64_t *bits = (uint64_t *)calloc(words, sizeof(uint64_t));
	long cnt = 0, k = 0;
	int u, v;
	node_t *pred, *n;

	assert(bits != NULL);
	while (cnt < NE) {
		u = (int)(splitmix64(&s) % (uint64_t)NV);
		v = (int)(splitmix64(&s) % (uint64_t)NV);
		if (u == v) continue;
		idx = (uint64_t)u * NV + v;
		if ((bits[idx >> 6] >> (idx & 63)) & 1) continue;
		bits[idx >> 6] |= 1ULL << (idx & 63);
		cnt++;
	}

	node_init(&TAIL, INT_MAX);
	adj = (list_t *)aligned_alloc(CACHE_LINE, (size_t)NV * sizeof(list_t));
	init_nodes = (node_t *)malloc((NE > 0 ? NE : 1) * sizeof(node_t));
	assert(adj != NULL && init_nodes != NULL);

	for (u = 0; u < NV; u++) {
		memset(&adj[u], 0, sizeof(list_t));
#if IMPL == 2
		pthread_mutex_init(&adj[u].lock, NULL);
#endif
		node_init(&adj[u].head, INT_MIN);
		pred = &adj[u].head;
		for (v = 0; v < NV; v++) {
			idx = (uint64_t)u * NV + v;
			if (!((bits[idx >> 6] >> (idx & 63)) & 1)) continue;
			n = &init_nodes[k++];
			node_init(n, v);
			pred->next = n;
			pred = n;
		}
		pred->next = &TAIL;
	}
	assert(k == NE);
	free(bits);
}

/* Structural check after the run: every list strictly sorted, values in
 * range, no reachable deleted node, and
 * final edges == initial edges + successful inserts - successful deletes. */
static int verify(long ins, long del, long *edges, uint64_t *chk)
{
	long cnt = 0, limit = NE + op_count + 1;
	uint64_t h = 0;
	int u, prev, ok = 1;
	node_t *n;

	for (u = 0; u < NV && ok; u++) {
		prev = INT_MIN;
		for (n = adj[u].head.next; n != &TAIL; n = n->next) {
			if (n == NULL || n->v <= prev || n->v < 0 || n->v >= NV) { ok = 0; break; }
#if IMPL == 6
			if (n->marked) { ok = 0; break; }
#endif
			if (++cnt > limit) { ok = 0; break; }
			h += edge_hash(u, n->v);
			prev = n->v;
		}
	}
	if (cnt != NE + ins - del) ok = 0;
	*edges = cnt;
	*chk = h;
	return ok;
}

/* Fast loader for the op file (not timed). */
static int next_num(char **pp, char *end, long long *out)
{
	char *p = *pp;
	long long x = 0;
	int neg = 0;

	while (p < end && !(*p >= '0' && *p <= '9') && *p != '-') p++;
	if (p >= end) return 0;
	if (*p == '-') { neg = 1; p++; }
	while (p < end && *p >= '0' && *p <= '9') x = x * 10 + (*p++ - '0');
	*pp = p;
	*out = neg ? -x : x;
	return 1;
}

static void load_ops(const char *path)
{
	FILE *fp = fopen(path, "rb");
	long len, i;
	long long a, b, c, max_id = 0;
	char *buf, *p, *end;
	int has_header;

	if (!fp) { perror(path); exit(1); }
	fseek(fp, 0, SEEK_END);
	len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	buf = (char *)malloc(len + 1);
	assert(buf != NULL);
	if ((long)fread(buf, 1, len, fp) != len) { fprintf(stderr, "Read error on %s\n", path); exit(1); }
	fclose(fp);
	p = buf;
	end = buf + len;

	/* Optional "n m seed" line.  Its first number is n >= 3, while an
	 * operation line starts with an op code (0..2), so the two cannot be
	 * confused.  Without it the default graph is used. */
	if (!next_num(&p, end, &a) || !next_num(&p, end, &b) || !next_num(&p, end, &c)) {
		fprintf(stderr, "%s: empty operation file\n", path);
		exit(1);
	}
	has_header = a > 2;
	if (has_header) {
		NV = (int)a; NE = (long)b; SEED = (unsigned long long)c;
		if (NV > MAX_NV || NE < 0 || NE > (long)NV * (NV - 1) / 2) {
			fprintf(stderr, "Bad graph line: need 3 <= n <= %d and 0 <= m <= n(n-1)/2\n", MAX_NV);
			exit(1);
		}
	} else {
		p = buf;                        /* the first line is an operation */
		NV = DEFAULT_NV; NE = DEFAULT_NE; SEED = DEFAULT_SEED;
	}

	ops = (op_t *)malloc(op_count * sizeof(op_t));
	assert(ops != NULL);
	for (i = 0; i < op_count; i++) {
		if (!next_num(&p, end, &a) || !next_num(&p, end, &b) || !next_num(&p, end, &c)) {
			fprintf(stderr, "%s holds only %ld operations, %ld requested\n", path, i, op_count);
			exit(1);
		}
		if (a < 0 || a > 2 || b < 0 || c < 0 || b >= MAX_NV || c >= MAX_NV ||
		    (has_header && (b >= NV || c >= NV))) {
			fprintf(stderr, "Bad operation on line %ld: %lld %lld %lld\n", i + 1 + has_header, a, b, c);
			exit(1);
		}
		if (b > max_id) max_id = b;
		if (c > max_id) max_id = c;
		ops[i].op = (int)a;
		ops[i].u = (int)b;
		ops[i].v = (int)c;
	}
	free(buf);
	if (max_id >= NV) NV = (int)max_id + 1;  /* only without the "n m seed" line */
}

static const char *impl_name(void)
{
	static const char *names[] = {
		"SEQ (no locks)", "P1 (graph mutex)", "P2 (list mutex)", "P3 (node mutex)",
		"P4 (node rwlock)", "P5 (node cmpxchg lock)", "P6 (optimistic, cmpxchg lock)"
	};
	return names[IMPL];
}

int main(int argc, char *argv[])
{
	int i;
	long ins = 0, del = 0, fnd = 0, edges, us;
	uint64_t chk;
	pthread_t *tid;
	tctx_t *ctx;
	int ok;

	if (argc != 4) {
		fprintf(stderr, "Usage: %s <op_file> <num_ops> <num_threads>\n", argv[0]);
		return 1;
	}
	op_count = atol(argv[2]);
	num_threads = atoi(argv[3]);
	if (op_count < 1 || num_threads < 1) {
		fprintf(stderr, "num_ops and num_threads must be positive\n");
		return 1;
	}
#if IMPL == 0
	if (num_threads != 1) {
		fprintf(stderr, "SEQ baseline always runs with 1 thread\n");
		num_threads = 1;
	}
#endif

	/* ---- allocation / initialisation (not timed) ---- */
	load_ops(argv[1]);
	build_graph();

	ctx = (tctx_t *)aligned_alloc(CACHE_LINE, num_threads * sizeof(tctx_t));
	tid = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	assert(ctx != NULL && tid != NULL);
	memset(ctx, 0, num_threads * sizeof(tctx_t));
	for (i = 0; i < num_threads; i++) {
		ctx[i].id = i;
		ctx[i].lo = op_count * i / num_threads;
		ctx[i].hi = op_count * (i + 1) / num_threads;
	}
	pthread_barrier_init(&bar, NULL, num_threads);

	/* ---- timed region: between the 2nd and 3rd barrier in worker() ---- */
	for (i = 1; i < num_threads; i++)
		pthread_create(&tid[i], NULL, worker, &ctx[i]);
	worker(&ctx[0]);
	for (i = 1; i < num_threads; i++)
		pthread_join(tid[i], NULL);

	us = (t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_nsec - t0.tv_nsec) / 1000;
	if (us < 1) us = 1;

	for (i = 0; i < num_threads; i++) {
		ins += ctx[i].ins_ok;
		del += ctx[i].del_ok;
		fnd += ctx[i].found;
	}
	ok = verify(ins, del, &edges, &chk);

	printf("Impl: %s, vertices: %d, initial edges: %ld, ops: %ld, threads: %d\n",
	       impl_name(), NV, NE, op_count, num_threads);
	printf("Successful inserts: %ld, deletes: %ld, searches found: %ld\n", ins, del, fnd);
	printf("Final edges: %ld, checksum: %016llx\n", edges, (unsigned long long)chk);
	printf("VERIFY: %s\n", ok ? "OK" : "FAILED");
	printf("Time: %ld microseconds, throughput: %.2f ops per second\n",
	       us, (double)op_count * 1000000.0 / (double)us);

	pthread_barrier_destroy(&bar);
	for (i = 0; i < num_threads; i++) free(ctx[i].slab);
	free(ctx); free(tid); free(ops); free(init_nodes); free(adj);
	return ok ? 0 : 2;
}
