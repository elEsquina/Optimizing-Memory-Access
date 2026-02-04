#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#define N 512  // Square matrix dimension (N x N).

int min(int a, int b) {
    return (a < b) ? a : b;
}

// Blocked (tiled) matrix multiplication.
void matrix_multiply_blocked(double **A, double **B, double **C, int n, int block_size) {
    // Iterate over submatrices so the inner work reuses cache lines more effectively.
    for (int ii = 0; ii < n; ii += block_size) {        // Block row index (A and C).
        for (int jj = 0; jj < n; jj += block_size) {    // Block column index (B and C).
            for (int kk = 0; kk < n; kk += block_size) {// Block index used for accumulation.
                
                // Compute C's current tile using the corresponding tiles of A and B.
                for (int i = ii; i < min(ii + block_size, n); i++) {
                    for (int k = kk; k < min(kk + block_size, n); k++) {
                        for (int j = jj; j < min(jj + block_size, n); j++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

// Unblocked multiplication (used as a reference point).
void matrix_multiply_standard(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // Allocate A, B, and C as dynamic 2D arrays.
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }
    
    // Fill A and B with deterministic pseudo-random values so runs are comparable.
    srand(42);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(rand() % 10) + 1.0;
            B[i][j] = (double)(rand() % 10) + 1.0;
            C[i][j] = 0.0;
        }
    }
    
    // Save measurements in a simple CSV-like text file.
    FILE *fp = fopen("mxm_bloc_results.txt", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        exit(EXIT_FAILURE);
    }
    
    fprintf(fp, "Block Matrix Multiplication Performance Analysis\n");
    fprintf(fp, "Matrix size: %d x %d\n\n", N, N);
    fprintf(fp, "Block Size, Time (msec), Bandwidth (MB/s), Speedup vs Standard\n");
    
    printf("Block Matrix Multiplication Performance Analysis\n");
    printf("Matrix size: %d x %d\n\n", N, N);
    printf("Block Size, Time (msec), Bandwidth (MB/s), Speedup\n");
    
    double standard_time = 0;
    long long total_ops = 4LL * N * N * N; // Rough traffic estimate: 3 loads + 1 store per multiply-add.
    long long total_bytes = total_ops * sizeof(double);
    
    // Sweep a few block sizes (powers of two).
    int block_sizes[] = {8, 16, 32, 64, 128, 256};
    int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    for (int bs_idx = 0; bs_idx < num_sizes; bs_idx++) {
        int block_size = block_sizes[bs_idx];
        
        // Clear C before each timed run.
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = 0.0;
            }
        }
        
        // Timing (CPU time via clock()).
        double start = (double)clock() / CLOCKS_PER_SEC;
        
        if (block_size == 512) {
            // If block_size equals N, the blocked routine degenerates to the unblocked i-k-j order.
            matrix_multiply_standard(A, B, C, N);
        } else {
            matrix_multiply_blocked(A, B, C, N, block_size);
        }
        
        double end = (double)clock() / CLOCKS_PER_SEC;
        double msec = (end - start) * 1000.0;
        double bandwidth = total_bytes * (1000.0 / msec) / (1024 * 1024);
        
        // Keep a baseline to compute speedup (first configuration used as reference here).
        if (block_size == 512 || bs_idx == 0) {
            standard_time = msec;
        }
        
        double speedup = standard_time / msec;
        
        printf("%4d, %10.2f, %12.2f, %6.2fx\n", block_size, msec, bandwidth, speedup);
        fprintf(fp, "%4d, %10.2f, %12.2f, %6.2fx\n", block_size, msec, bandwidth, speedup);
    }
    
    // Finally, run the unblocked version once (recorded separately in the output).
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    double start = (double)clock() / CLOCKS_PER_SEC;
    matrix_multiply_standard(A, B, C, N);
    double end = (double)clock() / CLOCKS_PER_SEC;
    double msec = (end - start) * 1000.0;
    double bandwidth = total_bytes * (1000.0 / msec) / (1024 * 1024);
    
    printf("Standard (no blocking), %10.2f, %12.2f, %6.2fx\n", msec, bandwidth, 1.0);
    fprintf(fp, "Standard (no blocking), %10.2f, %12.2f, %6.2fx\n", msec, bandwidth, 1.0);
    
    fclose(fp);
    printf("\nResults saved to mxm_bloc_results.txt\n");
    
    // Free memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    
    return 0;
}
