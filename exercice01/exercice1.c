#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#define MAX_STRIDE 20

int main()
{
    // Simple stride experiment: keep the number of touches constant (N) while
    // increasing the distance between successive accesses to highlight cache effects.
    int N = 1000000;
    double *a;
    a = malloc(N * MAX_STRIDE * sizeof(double));
    double sum, rate, msec, start, end;

    // Initialize the whole buffer so pages are mapped and values are defined.
    for (int i = 0; i < N * MAX_STRIDE; i++)
        a[i] = 1.;

    printf("stride , sum, time (msec), rate (MB/s)\n");

    for (int i_stride = 1; i_stride <= MAX_STRIDE; i_stride++)
    {
        sum = 0.0;
        start = (double)clock() / CLOCKS_PER_SEC;

        // Visit exactly N elements but with a varying stride.
        for (int i = 0; i < N * i_stride; i += i_stride)
            sum += a[i];

        end = (double)clock() / CLOCKS_PER_SEC;
        msec = (end - start) * 1000.0; // Elapsed time in milliseconds.
        rate = sizeof(double) * N * (1000.0 / msec) / (1024 * 1024);

        printf("%d, %f, %f, %f\n", i_stride, sum, msec, rate);
    }
    free(a);
}
