/*
 * chi = chromatic number of G (vertices only).
 * Fact A: answer >= chi.
 * Fact B: if vertices use colours 0..K-1 with K ODD, colour every edge (u,v) with
 *         (c(u)+c(v)) * inv2 mod K, inv2 = (K+1)/2. This satisfies (ii) and (iii).
 * So: m == 0 or chi odd -> answer = chi.
 *     chi even, m > 0   -> answer is chi or chi+1; ONE exact test "is G' chi-colourable?"
 *                          on the conflict graph G' decides it.
 * Both tests use one parallel backtracking: first d nodes are enumerated into prefixes
 * that threads take dynamically; a shared atomic flag stops the search when one wins.
 */
#include <omp.h>
#include <sys/time.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

#define PREFIX_PER_THREAD 50

vector<long long> parseInts(const string& s) {
    vector<long long> v;
    long long cur = 0;
    bool in = false;
    for (char ch : s) {
        if (ch >= '0' && ch <= '9') { cur = cur * 10 + (ch - '0'); in = true; }
        else if (in) { v.push_back(cur); cur = 0; in = false; }
    }
    if (in) v.push_back(cur);
    return v;
}

// Line 1: n.  Line k+2: neighbours i > k of vertex k.  E sorted by (u,v), no duplicates.
bool readGraph(const char* fn, int& n, vector<pair<int, int>>& E) {
    ifstream in(fn);
    if (!in) return false;
    string line;
    long long nn = -1;
    while (getline(in, line)) {
        vector<long long> v = parseInts(line);
        if (!v.empty()) { nn = v[0]; break; }
    }
    if (nn < 0) return false;
    n = (int)nn;
    for (int k = 0; k < n && getline(in, line); ++k)
        for (long long x : parseInts(line)) {
            if (x >= n || x == k) continue;
            E.push_back({(int)min<long long>(k, x), (int)max<long long>(k, x)});
        }
    sort(E.begin(), E.end());
    E.erase(unique(E.begin(), E.end()), E.end());
    return true;
}

// Nodes renamed by position in the colouring order; backAdj = earlier neighbours.
struct Problem {
    int N = 0;
    vector<int> order;
    vector<int> backOff, backAdj;
};

// Order: most already-ordered neighbours first, ties by degree.
Problem makeProblem(const vector<vector<int>>& adj) {
    Problem P;
    int N = (int)adj.size();
    P.N = N;
    vector<int> posOf(N, -1), conn(N, 0);
    for (int step = 0; step < N; ++step) {
        int best = -1;
        for (int v = 0; v < N; ++v) {
            if (posOf[v] != -1) continue;
            if (best == -1 || conn[v] > conn[best] ||
                (conn[v] == conn[best] && adj[v].size() > adj[best].size()))
                best = v;
        }
        posOf[best] = step;
        P.order.push_back(best);
        for (int w : adj[best]) conn[w]++;
    }
    P.backOff.assign(N + 1, 0);
    for (int p = 0; p < N; ++p) {
        for (int w : adj[P.order[p]])
            if (posOf[w] < p) P.backAdj.push_back(posOf[w]);
        P.backOff[p + 1] = (int)P.backAdj.size();
    }
    return P;
}

atomic<bool> found(false);

// Colour positions pos..N-1; positions 0..pos-1 already use exactly 'used' colours.
bool dfs(const Problem& P, int pos, int used, int k, vector<int>& col) {
    if (found.load(memory_order_relaxed)) return false;
    if (pos == P.N) return true;
    int limit = min(k - 1, used);  // symmetry breaking
    for (int c = 0; c <= limit; ++c) {
        bool ok = true;
        for (int i = P.backOff[pos]; i < P.backOff[pos + 1]; ++i)
            if (col[P.backAdj[i]] == c) { ok = false; break; }
        if (!ok) continue;
        col[pos] = c;
        if (dfs(P, pos + 1, max(used, c + 1), k, col)) return true;
    }
    return false;
}

// Same search, forbidden colours collected once in a bitmask (k <= 64).
bool dfsMask(const Problem& P, int pos, int used, int k, vector<int>& col) {
    if (found.load(memory_order_relaxed)) return false;
    if (pos == P.N) return true;
    uint64_t forbidden = 0;
    for (int i = P.backOff[pos]; i < P.backOff[pos + 1]; ++i)
        forbidden |= 1ULL << col[P.backAdj[i]];
    int limit = min(k - 1, used);
    for (int c = 0; c <= limit; ++c) {
        if ((forbidden >> c) & 1) continue;
        col[pos] = c;
        if (dfsMask(P, pos + 1, max(used, c + 1), k, col)) return true;
    }
    return false;
}

// Is the graph k-colourable? If yes, result[p] = colour of position p.
bool solveK(const Problem& P, int k, int n_threads, vector<int>& result) {
    int N = P.N;
    vector<vector<int>> prefixes(1);  // all valid colourings of the first d positions
    size_t target = (size_t)PREFIX_PER_THREAD * n_threads;
    int d = 0;
    while (d < N && prefixes.size() < target) {
        vector<vector<int>> next;
        for (const vector<int>& p : prefixes) {
            int used = 0;
            for (int c : p) used = max(used, c + 1);
            for (int c = 0; c <= min(k - 1, used); ++c) {
                bool ok = true;
                for (int i = P.backOff[d]; i < P.backOff[d + 1]; ++i)
                    if (p[P.backAdj[i]] == c) { ok = false; break; }
                if (!ok) continue;
                vector<int> q = p;
                q.push_back(c);
                next.push_back(move(q));
            }
        }
        prefixes.swap(next);
        ++d;
        if (prefixes.empty()) return false;
    }

    found.store(false);
    long long cnt = (long long)prefixes.size();
    result.assign(N, 0);
#pragma omp parallel num_threads(n_threads)
    {
        vector<int> col(N, -1);  // private colour array
#pragma omp for schedule(dynamic, 1)
        for (long long i = 0; i < cnt; ++i) {
            if (found.load(memory_order_relaxed)) continue;
            const vector<int>& p = prefixes[i];
            int used = 0;
            for (int j = 0; j < d; ++j) { col[j] = p[j]; used = max(used, p[j] + 1); }
            bool ok = (k <= 64) ? dfsMask(P, d, used, k, col) : dfs(P, d, used, k, col);
            if (ok) {
#pragma omp critical(result_lock)
                {
                    if (!found.load()) { result = col; found.store(true); }
                }
            }
        }
    }
    return found.load();
}

// Conflict graph G'. Nodes: vertex u -> u, edge e -> n+e. Conditions (i), (ii), (iii).
vector<vector<int>> buildConflictGraph(int n, const vector<pair<int, int>>& E) {
    int m = (int)E.size(), N = n + m;
    vector<int> upOff(n + 1, 0), upAdj(m);  // E sorted: edge (u, upAdj[a]) has id a
    for (int e = 0; e < m; ++e) { upOff[E[e].first + 1]++; upAdj[e] = E[e].second; }
    for (int u = 0; u < n; ++u) upOff[u + 1] += upOff[u];

    vector<pair<int, int>> conflicts;
    for (int e = 0; e < m; ++e) {
        int u = E[e].first, v = E[e].second, en = n + e;
        conflicts.push_back({u, v});
        conflicts.push_back({u, en});
        conflicts.push_back({v, en});
    }
#pragma omp parallel
    {
        vector<pair<int, int>> loc;  // private per thread
#pragma omp for schedule(dynamic, 8)
        for (int u = 0; u < n; ++u)
            for (int a = upOff[u]; a < upOff[u + 1]; ++a) {
                int v = upAdj[a];
                int p = a + 1, q = upOff[v];  // common higher neighbours w of u and v
                while (p < upOff[u + 1] && q < upOff[v + 1]) {
                    if (upAdj[p] < upAdj[q]) ++p;
                    else if (upAdj[p] > upAdj[q]) ++q;
                    else {  // triangle u<v<w with edges a=(u,v), p=(u,w), q=(v,w)
                        loc.push_back({n + a, n + p});
                        loc.push_back({n + p, n + q});
                        loc.push_back({n + a, n + q});
                        ++p; ++q;
                    }
                }
            }
#pragma omp critical(merge)
        conflicts.insert(conflicts.end(), loc.begin(), loc.end());
    }
    vector<vector<int>> adj(N);
    for (auto& pr : conflicts) {
        adj[pr.first].push_back(pr.second);
        adj[pr.second].push_back(pr.first);
    }
    return adj;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Incorrect number of arguments! Provide input file, output file and number of threads!\n");
        exit(1);
    }
    int n_threads = max(1, atoi(argv[3]));
    omp_set_dynamic(0);
    omp_set_num_threads(n_threads);

    int n = 0;
    vector<pair<int, int>> E;
    FILE* out_fp = fopen(argv[2], "w+");
    if (out_fp == NULL || !readGraph(argv[1], n, E)) {
        printf("Error opening file\n");
        exit(1);
    }
    int m = (int)E.size();

    struct timeval tv0, tv1, tv2;
    struct timezone tz0, tz1, tz2;
    gettimeofday(&tv0, &tz0);

    /* ---- Step 1: chromatic number chi of G and a vertex colouring ---- */
    vector<vector<int>> gadj(n);
    for (auto& e : E) { gadj[e.first].push_back(e.second); gadj[e.second].push_back(e.first); }
    Problem PV = makeProblem(gadj);
    vector<int> vcol(n, 0), res;
    int chi = 0;
    for (int k = 1; k <= n; ++k)
        if (solveK(PV, k, n_threads, res)) {
            chi = k;
            for (int p = 0; p < n; ++p) vcol[PV.order[p]] = res[p];
            break;
        }

    gettimeofday(&tv1, &tz1);

    /* ---- Step 2: colour the edges ---- */
    vector<int> ecol(m, 0);
    int answer = chi;
    bool useFormula = true;
    if (m > 0 && chi % 2 == 0) {
        // chi even: is chi enough, or do we need chi+1?  One exact test on G'.
        vector<vector<int>> cadj = buildConflictGraph(n, E);
        Problem PG = makeProblem(cadj);
        if (solveK(PG, chi, n_threads, res)) {
            for (int p = 0; p < PG.N; ++p) {
                int node = PG.order[p];
                if (node < n) vcol[node] = res[p]; else ecol[node - n] = res[p];
            }
            useFormula = false;
        } else {
            answer = chi + 1;
        }
    }
    if (useFormula && m > 0) {
        long long K = answer, inv2 = (K + 1) / 2;  // K is odd here
        for (int e = 0; e < m; ++e)
            ecol[e] = (int)(((long long)(vcol[E[e].first] + vcol[E[e].second]) * inv2) % K);
    }

    gettimeofday(&tv2, &tz2);

    /* ---- output ---- */
    for (int u = 0; u < n; ++u) fprintf(out_fp, "%d, %d\n", u, vcol[u]);
    for (int e = 0; e < m; ++e) fprintf(out_fp, "(%d, %d), %d\n", E[e].first, E[e].second, ecol[e]);
    fprintf(out_fp, "%d\n", answer);

    printf("G: n=%d m=%d chi(G)=%d colours=%d threads=%d\n", n, m, chi, answer, n_threads);
    printf("Vertex colouring: %ld microseconds\n",
           (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));
    printf("Edge step: %ld microseconds\n",
           (tv2.tv_sec - tv1.tv_sec) * 1000000 + (tv2.tv_usec - tv1.tv_usec));
    printf("Time: %ld microseconds\n",
           (tv2.tv_sec - tv0.tv_sec) * 1000000 + (tv2.tv_usec - tv0.tv_usec));

    fclose(out_fp);
    return 0;
}