#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#define R1 512 // number of rows in Matrix-1
#define C1 512 // number of columns in Matrix-1
#define R2 512 // number of rows in Matrix-2
#define C2 512 // number of columns in Matrix-2



int main() {
    // Compare two loop orderings for dense matrix multiplication (512x512)
    // and measure how access patterns impact cache behavior.

    // Use dynamic allocation to avoid large stack arrays.
    double **m1 = (double **)malloc(R1 * sizeof(double *));
    double **m2 = (double **)malloc(R2 * sizeof(double *));
    double **result_ijk = (double **)malloc(R1 * sizeof(double *));
    double **result_ikj = (double **)malloc(R1 * sizeof(double *));
    
    for (int i = 0; i < R1; i++) {
        m1[i] = (double *)malloc(C1 * sizeof(double));
        result_ijk[i] = (double *)malloc(C2 * sizeof(double));
        result_ikj[i] = (double *)malloc(C2 * sizeof(double));
    }
    for (int i = 0; i < R2; i++) {
        m2[i] = (double *)malloc(C2 * sizeof(double));
    }

    // Fill inputs with small pseudo-random values (not focusing on numerical accuracy here).
    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C1; j++) {
            m1[i][j] = (double)(rand() % 10) + 1.0;
        }
    }
    
    for (int i = 0; i < R2; i++) {
        for (int j = 0; j < C2; j++) {
            m2[i][j] = (double)(rand() % 10) + 1.0;
        }
    }


     if (C1 != R2) {
        printf("The number of columns in Matrix-1 must be "
               "equal to the number of rows in "
               "Matrix-2\n");
        exit(EXIT_FAILURE);
    }
    
    // Write results to a CSV-like text file for later plotting/reporting.
    FILE *fp = fopen("mxm_results.txt", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        exit(EXIT_FAILURE);
    }
    
    fprintf(fp, "Matrix Multiplication Performance Analysis\n");
    fprintf(fp, "Matrix size: %d x %d\n\n", R1, C2);
    fprintf(fp, "Version, Time (msec), Bandwidth (MB/s)\n");
    
    printf("Matrix Multiplication Performance Analysis\n");
    printf("Matrix size: %d x %d\n\n", R1, C2);
    printf("Version, Time (msec), Bandwidth (MB/s)\n");

    // Clear output matrices before timing.
    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C2; j++) {
            result_ijk[i][j] = 0;
            result_ikj[i][j] = 0;
        }
    }

    double rate, msec, start, end;
    long long total_ops = 4LL * R1 * R2 * C2; // 4 memory ops per iteration (3 reads + 1 write)
    long long total_bytes = total_ops * sizeof(double);

    // ===== Version 1: i-j-k loop order =====
    // B is accessed column-wise (poor spatial locality in row-major storage).
    start = (double)clock() / CLOCKS_PER_SEC;

    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C2; j++) {
            for (int k = 0; k < R2; k++) {
                result_ijk[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    end = (double)clock() / CLOCKS_PER_SEC;
    msec = (end - start) * 1000.0;
    rate = total_bytes * (1000.0 / msec) / (1024 * 1024);

    printf("i-j-k (Standard), %.4f, %.2f\n", msec, rate);
    fprintf(fp, "i-j-k (Standard), %.4f, %.2f\n", msec, rate);

    // ===== Version 2: i-k-j loop order =====
    // Inner loop walks through B[k][j] contiguously, which is typically cache-friendly.
    start = (double)clock() / CLOCKS_PER_SEC;

    for (int i = 0; i < R1; i++) {
        for (int k = 0; k < R2; k++) {
            for (int j = 0; j < C2; j++) {
                result_ikj[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    end = (double)clock() / CLOCKS_PER_SEC;
    msec = (end - start) * 1000.0;
    rate = total_bytes * (1000.0 / msec) / (1024 * 1024);

    printf("i-k-j (Optimized), %.4f, %.2f\n", msec, rate);
    fprintf(fp, "i-k-j (Optimized), %.4f, %.2f\n", msec, rate);

    fclose(fp);
    printf("\nResults saved to mxm_results.txt\n");

    // Release all allocated rows and pointer tables.
    for (int i = 0; i < R1; i++) {
        free(m1[i]);
        free(result_ijk[i]);
        free(result_ikj[i]);
    }
    for (int i = 0; i < R2; i++) {
        free(m2[i]);
    }
    free(m1);
    free(m2);
    free(result_ijk);
    free(result_ikj);



    return 0;
}
