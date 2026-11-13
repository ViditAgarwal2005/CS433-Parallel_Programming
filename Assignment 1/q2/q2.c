#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

#define C 16

double **A;
double *y, *x;

int min(int a, int b) {
    return ((a < b) ? a : b);
}

int max(int a, int b) {
    return ((a > b) ? a : b);
}

int main(int argc, char **argv) {

    FILE *in_fp, *out_fp;

    if (argc != 4) {
        printf("Incorrect number of arguments! Provide input file, output file and number of threads!\n");
        exit(1);
    }

    int n_threads = atoi(argv[3]);

    out_fp = fopen(argv[2], "w+");
    in_fp = fopen(argv[1], "r");

    if (in_fp == NULL || out_fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    /*
        Matrix size is N = 2n + 1
    */
    int n;
    int ret = fscanf(in_fp, "%d", &n);

    if (ret == EOF) {
        printf("Error in reading from file\n");
        exit(1);
    }

    int N = 2 * n + 1;

    /*
        A is an hourglass matrix.
        We store only the non-zero part of every row.
    */
    A = (double **)malloc(N * sizeof(double *));
    assert(A != NULL);

    for (int row = 0; row < N; row++) {

        int start = min(row, N - 1 - row);
        int end = max(row, N - 1 - row);

        int length = end - start + 1;

        A[row] = (double *)malloc(length * sizeof(double));
        assert(A[row] != NULL);

        for (int col = start; col <= end; col++) {

            ret = fscanf(in_fp, "%lf", &A[row][col - start]);

            if (ret == EOF) {
                printf("Error in reading matrix\n");
                exit(1);
            }
        }
    }

    y = (double *)malloc(N * sizeof(double));
    x = (double *)malloc(N * sizeof(double));

    assert(y != NULL);
    assert(x != NULL);

    for (int row = 0; row < N; row++) {

        ret = fscanf(in_fp, "%lf", &y[row]);

        if (ret == EOF) {
            printf("Error in reading y\n");
            exit(1);
        }

        x[row] = 0.0;
    }

    /*
        Number of pairs outside the centre is n.

        We process C pairs at a time.
    */
    int times = (n / C) + ((n % C) ? 1 : 0);

    struct timeval tv0, tv1;
    struct timezone tz0, tz1;

    gettimeofday(&tv0, &tz0);

    /*

        Each iteration consists of:

        1. Solve a small hourglass block sequentially
           using single.

        2. Update all outer rows in parallel
           using omp for.
    */
#pragma omp parallel num_threads(n_threads)
    {

        /*
            First solve the centre.

            x[n] = y[n] / A[n][n]

            Since the centre row contains only one element,
            A[n][n] is A[n][0].
        */
#pragma omp single
        {
            x[n] = y[n] / A[n][0];
        }

        /*
            Remove the centre contribution from all
            rows outside the centre.

            For pair (row, N-1-row), the centre is at
            offset n-row in both stored rows.
        */
#pragma omp for
        for (int row = 0; row < n; row++) {

            int right = N - 1 - row;

            int center_offset = n - row;

            y[row] -= A[row][center_offset] * x[n];

            y[right] -= A[right][center_offset] * x[n];
        }

        /*
            Process blocks from the centre outward.
        */
        for (int block = 0; block < times; block++) {

            int inner_left = n - block * C - 1;

            int outer_left = max(0, n - (block + 1) * C);

            /*
                -----------------------------
                TRIANGLE / SMALL BLOCK SOLVE
                -----------------------------
            */

#pragma omp single
            {

                for (int left = inner_left;
                     left >= outer_left;
                     left--) {

                    int right = N - 1 - left;

                    /*
                        Row 'left' starts at column 'left'.

                        Therefore:

                            A[left][0]       = A[left][left]
                            A[left][right-left] = A[left][right]
                    */

                    double a = A[left][0];

                    double b = A[left][right - left];

                    /*
                        Row 'right' starts at column 'left'.

                        Therefore:

                            A[right][0]       = A[right][left]
                            A[right][right-left] = A[right][right]
                    */

                    double c = A[right][0];

                    double d = A[right][right - left];

                    /*
                        Solve:

                            a*x[left] + b*x[right] = y[left]
                            c*x[left] + d*x[right] = y[right]

                        determinant = ad - bc
                    */

                    double determinant = a * d - b * c;

                    double old_y_left = y[left];
                    double old_y_right = y[right];

                    x[left] =
                        (old_y_left * d - b * old_y_right)
                        / determinant;

                    x[right] =
                        (a * old_y_right - c * old_y_left)
                        / determinant;

                    /*
                        Update the remaining rows inside this
                        small block.

                    */

                    for (int row = left - 1;
                         row >= outer_left;
                         row--) {

                        int row_right = N - 1 - row;

                        /*
                            Row 'row' starts at column 'row'.

                            Contribution from x[left].
                        */
                        y[row] -=
                            A[row][left - row] * x[left];

                        /*
                            Contribution from x[right].
                        */
                        y[row] -=
                            A[row][right - row] * x[right];

                        /*
                            The corresponding right-side row
                            also gets both contributions.
                        */
                        y[row_right] -=
                            A[row_right][left - row] * x[left];

                        y[row_right] -=
                            A[row_right][right - row] * x[right];
                    }
                }
            }

            /*
                -----------------------------
                RECTANGLE / OUTER UPDATE
                -----------------------------

                The block has now been solved.

                All rows outside the block are independent,
                so they can be updated in parallel.

                We subtract the contribution of every x
                computed in this block.
            */

#pragma omp for
            for (int row = 0; row < outer_left; row++) {

                int right = N - 1 - row;

                /*
                    Left half of the solved block.

                    Example:

                        x[n-16], ..., x[n-1]
                */
                for (int col = outer_left;
                     col <= inner_left;
                     col++) {

                    y[row] -=
                        A[row][col - row] * x[col];

                    y[right] -=
                        A[right][col - row] * x[col];
                }

                /*
                    Right half of the solved block.

                    Example:

                        x[n+1], ..., x[n+16]
                */
                int right_block_start =
                    N - 2 - inner_left;

                int right_block_end =
                    N - 1 - outer_left;

                for (int col = right_block_start;
                     col <= right_block_end;
                     col++) {

                    y[row] -=
                        A[row][col - row] * x[col];

                    y[right] -=
                        A[right][col - row] * x[col];
                }
            }
        }
    }

    gettimeofday(&tv1, &tz1);

    /*
        Print solution.
    */
    for (int i = 0; i < N; i++) {
        fprintf(out_fp, "%lf ", x[i]);
    }

    fprintf(out_fp, "\n");

    printf("Time: %ld microseconds\n",
           (tv1.tv_sec - tv0.tv_sec) * 1000000 +
           (tv1.tv_usec - tv0.tv_usec));

    /*
        Free memory.
    */
    for (int row = 0; row < N; row++) {
        free(A[row]);
    }

    free(A);
    free(x);
    free(y);

    fclose(in_fp);
    fclose(out_fp);

    return 0;
}