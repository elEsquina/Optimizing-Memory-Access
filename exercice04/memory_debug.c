#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 5

int* allocate_array(int size) {
    // Allocate an integer buffer and fail fast if allocation fails.
    int *arr = (int*)malloc(size * sizeof(int));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return arr;
}

void initialize_array(int *arr, int size) {
    // Populate with predictable values (useful for debugging/validation).
    if (!arr) return;
    for (int i = 0; i < size; i++) {
        arr[i] = i * 10;
    }
}

void print_array(int *arr, int size) {
    // Pretty-print the current contents.
    if (!arr) return;
    printf("Array elements: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int* duplicate_array(int *arr, int size) {
    // Create a heap-allocated copy of the array.
    if (!arr) return NULL;
    int *copy = (int*)malloc(size * sizeof(int));
    if (!copy) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    memcpy(copy, arr, size * sizeof(int));
    return copy;
}

void free_memory(int *arr) {
    // Centralized free helper (kept trivial on purpose).
    if (arr) {
        free(arr);
    }
}

int main() {
    // Allocate → initialize → duplicate → free (Valgrind should report no leaks).
    int *array = allocate_array(SIZE);
    initialize_array(array, SIZE);
    print_array(array, SIZE);
    int *array_copy = duplicate_array(array, SIZE);
    print_array(array_copy, SIZE);
    free_memory(array);
    free_memory(array_copy);
    return 0; // Memory leaks fixed!
}
