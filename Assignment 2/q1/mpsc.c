/*
 * CS433 Assignment 2, Problem 1
 * Multi-producer single-consumer bounded buffer (N slots, P producers,
 * n values per producer), built on four different semaphore implementations
 * selected at compile time with -DIMPL=<1..4>:
 *
 *   IMPL=1  P1: lock-free FIFO semaphore (two ticket counters, cmpxchg only)
 *   IMPL=2  P2: lock-based busy-waiting semaphore (pthread mutex inside it)
 *   IMPL=3  P3: Linux POSIX semaphore (sem_t)
 *   IMPL=4  P4: condition variables (mutex used only to guard the condvars)
 *
 * Usage: ./pX N P n
 *   stdout : the n*P values in the order the consumer consumed them
 *   stderr : total time and a self-check ("VERIFY: OK")
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <assert.h>
#include <sys/time.h>

#ifndef IMPL
#define IMPL 1
#endif

#define CACHE_LINE 64

/* ------------------------------------------------------------------ */
/* Atomic primitive: the only atomic instruction used by the lock-free */
/* parts of the program.  This is the CompareAndSet developed in class */
/* (prog1_cmpxchg_lock.c / prog1_lock_free.c).  It has no memory       */
/* clobber, so explicit compiler_bar()s are placed where ordinary      */
/* accesses must not be reordered around it.                           */
/* ------------------------------------------------------------------ */
unsigned char CompareAndSet(int oldVal, int newVal, int *ptr)
{
  int oldValOut;
  unsigned char result;

  asm(
		"lock cmpxchgl %4, %1 \n setzb %0"
		:"=qm"(result),  "+m" (*ptr), "=a" (oldValOut)
		:"a" (oldVal),  "r" (newVal)
		:
	);

  return result;
}

#define cpu_relax()    asm volatile("pause" ::: "memory")
#define compiler_bar() asm volatile("" ::: "memory")

/* fetch-and-add built from CompareAndSet (lock-free CAS loop) */
static inline unsigned fetch_add(volatile unsigned *p, unsigned d)
{
    unsigned old;
    do {
        old = *p;
    } while (!CompareAndSet((int)old, (int)(old + d), (int *)p));
    return old;
}

/* ================================================================== */
/*                       Semaphore implementations                    */
/* ================================================================== */
#if IMPL == 1
/* ---------- P1: lock-free FIFO semaphore -------------------------- *
 * ticket : number of wait calls so far   (fetch_add)
 * grant  : initial value + number of posts so far (fetch_add)
 * The waiter holding ticket t owns the t-th permit and may proceed once
 * grant > t, so permits are handed out strictly in ticket (FIFO) order.
 */
typedef struct {
    volatile unsigned ticket; char p0[CACHE_LINE - sizeof(unsigned)];
    volatile unsigned grant;  char p1[CACHE_LINE - sizeof(unsigned)];
} lf_state;

typedef struct { lf_state *st; } ssem_t;

static void ssem_init(ssem_t *s, unsigned value)
{
    s->st = (lf_state *)aligned_alloc(CACHE_LINE, sizeof(lf_state));
    assert(s->st != NULL);
    memset(s->st, 0, sizeof(lf_state));
    s->st->grant = value;
}

static void ssem_wait(ssem_t *s)
{
    lf_state *st = s->st;
    unsigned t = fetch_add(&st->ticket, 1);
    while ((int)(st->grant - t) <= 0) cpu_relax();   /* wraparound-safe grant > t */
    compiler_bar();
}

static void ssem_post(ssem_t *s)
{
    compiler_bar();
    (void)fetch_add(&s->st->grant, 1);
}

static void ssem_destroy(ssem_t *s) { free(s->st); s->st = NULL; }

#elif IMPL == 2
/* ---------- P2: lock-based busy-waiting semaphore ----------------- */
typedef struct {
    pthread_mutex_t lock;
    int count;
} ssem_t;

static void ssem_init(ssem_t *s, unsigned value)
{
    pthread_mutex_init(&s->lock, NULL);
    s->count = (int)value;
}

static void ssem_wait(ssem_t *s)
{
    for (;;) {
        pthread_mutex_lock(&s->lock);
        if (s->count > 0) {
            s->count--;
            pthread_mutex_unlock(&s->lock);
            return;
        }
        pthread_mutex_unlock(&s->lock);
        cpu_relax();
    }
}

static void ssem_post(ssem_t *s)
{
    pthread_mutex_lock(&s->lock);
    s->count++;
    pthread_mutex_unlock(&s->lock);
}

static void ssem_destroy(ssem_t *s) { pthread_mutex_destroy(&s->lock); }

#elif IMPL == 3
/* ---------- P3: Linux semaphore ----------------------------------- */
typedef struct { sem_t s; } ssem_t;

static void ssem_init(ssem_t *s, unsigned value) { sem_init(&s->s, 0, value); }
static void ssem_wait(ssem_t *s) { while (sem_wait(&s->s) != 0) ; }
static void ssem_post(ssem_t *s) { sem_post(&s->s); }
static void ssem_destroy(ssem_t *s) { sem_destroy(&s->s); }

#elif IMPL == 4
/* ---------- P4: condition variables -------------------------------- *
 * The mutex is used only to protect the condition variable's predicate
 * (the counter).  All other shared state in the program stays lock-free.
 */
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  cv;
    int count;
} ssem_t;

static void ssem_init(ssem_t *s, unsigned value)
{
    pthread_mutex_init(&s->lock, NULL);
    pthread_cond_init(&s->cv, NULL);
    s->count = (int)value;
}

static void ssem_wait(ssem_t *s)
{
    pthread_mutex_lock(&s->lock);
    while (s->count == 0) pthread_cond_wait(&s->cv, &s->lock);
    s->count--;
    pthread_mutex_unlock(&s->lock);
}

static void ssem_post(ssem_t *s)
{
    pthread_mutex_lock(&s->lock);
    s->count++;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->lock);
}

static void ssem_destroy(ssem_t *s)
{
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->lock);
}
#else
#error "IMPL must be 1..4"
#endif

/* ================================================================== */
/*                    Multi-producer single-consumer                  */
/* ================================================================== */
/* One slot per cache line so producers writing neighbouring slots do not
 * false-share.  'ready' is set by the producer after writing 'val' and
 * cleared by the consumer after reading it. */
typedef struct {
    int val;
    volatile int ready;
    char pad[CACHE_LINE - 2 * sizeof(int)];
} slot_t;

static slot_t *slots;
static int *out;                         /* consumer's result buffer */
static ssem_t empty_sem, full_sem;
static volatile unsigned tail;           /* next slot index to claim (CAS) */
static long N, P, n;

static void *producer(void *arg)
{
    long p = (long)arg, k;
    for (k = 0; k < n; k++) {
        ssem_wait(&empty_sem);                       /* an empty slot exists */
        unsigned t = fetch_add(&tail, 1);            /* claim a distinct slot */
        slot_t *s = &slots[t % (unsigned)N];
        s->val = (int)(p * n + k);
        compiler_bar();                              /* x86 TSO: stores stay ordered */
        s->ready = 1;                                /* publish */
        ssem_post(&full_sem);
    }
    return NULL;
}

static void consume_all(void)
{
    long total = n * P, i;
    unsigned head = 0;
    for (i = 0; i < total; i++) {
        ssem_wait(&full_sem);
        slot_t *s = &slots[head++ % (unsigned)N];
        /* A later-claimed slot may be published before an earlier one, so
         * wait for this particular slot's producer to finish writing. */
        while (!s->ready) cpu_relax();
        compiler_bar();
        out[i] = s->val;
        compiler_bar();
        s->ready = 0;
        ssem_post(&empty_sem);
    }
}

int main(int argc, char *argv[])
{
    struct timeval t0, t1;
    pthread_t *th;
    long i;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s N P n\n", argv[0]);
        return 1;
    }
    N = atol(argv[1]); P = atol(argv[2]); n = atol(argv[3]);
    assert(N >= 1 && P >= 1 && n >= 1);
    assert(n * P < (1L << 31));

    /* ---- allocation / initialisation (not timed) ---- */
    slots = (slot_t *)aligned_alloc(CACHE_LINE, N * sizeof(slot_t));
    out   = (int *)malloc(n * P * sizeof(int));
    th    = (pthread_t *)malloc(P * sizeof(pthread_t));
    assert(slots && out && th);
    memset(slots, 0, N * sizeof(slot_t));
    tail = 0;
    ssem_init(&empty_sem, (unsigned)N);
    ssem_init(&full_sem, 0);

    /* ---- timed region ---- */
    gettimeofday(&t0, NULL);
    for (i = 0; i < P; i++) pthread_create(&th[i], NULL, producer, (void *)i);
    consume_all();
    for (i = 0; i < P; i++) pthread_join(th[i], NULL);
    gettimeofday(&t1, NULL);

    fprintf(stderr, "Total time: %ld microseconds\n",
            (long)((t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_usec - t0.tv_usec)));

    /* ---- self check: every producer's values arrive complete and in order ---- */
    long *next = (long *)calloc(P, sizeof(long));
    int ok = 1;
    for (i = 0; i < n * P && ok; i++) {
        long p = out[i] / n, k = out[i] % n;
        if (p < 0 || p >= P || k != next[p]++) ok = 0;
    }
    fprintf(stderr, "VERIFY: %s\n", ok ? "OK" : "FAILED");

    /* ---- the consumer's printed output ---- */
    for (i = 0; i < n * P; i++) printf("%d\n", out[i]);

    ssem_destroy(&empty_sem);
    ssem_destroy(&full_sem);
    free(next); free(th); free(out); free(slots);
    return ok ? 0 : 2;
}
