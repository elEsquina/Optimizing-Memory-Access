# TP1 - Optimizing Memory Access (Lab Report)

**Course:** Parallel Computing  
**Student:** Othmane Azoubi  
**Date:** January 2026  
**Goal:** Study how memory access patterns affect performance, then apply simple transformations (loop order and blocking) to improve cache locality. A short memory-debugging task is included to practice leak detection.

---

## Exercise 1 - Stride experiment + compiler optimization

### What I implemented
The program `exercice01/exercice1.c` iterates over a large array while varying the **stride** (1 to 20). For each stride, it touches **exactly N elements**, but the distance between consecutive accesses changes:

```c
for (int i = 0; i < N * i_stride; i += i_stride)
    sum += a[i];
```

Stride 1 is contiguous (best spatial locality). Larger strides skip elements, so fewer useful values are brought per cache line, which increases misses.

### How I ran it
I compiled the same source with and without optimizations and redirected the CSV-like output into files:

```bash
gcc -O0 exercice01/exercice1.c -o exercice1_O0.exe
gcc -O2 exercice01/exercice1.c -o exercice1_O2.exe
./exercice1_O0.exe > results_O0.txt
./exercice1_O2.exe > results_O2.txt
```

Plotting (from the repository root):

```bash
python3 exercice01/plot_results.py --o0 results_O0.txt --o2 results_O2.txt --output exercice01/stride_analysis.png --no-show
```

### Results
![Stride Analysis](exercice01/stride_analysis.png)

- Average execution time: **-O0 = 4.35 ms**, **-O2 = 3.35 ms** (**1.30× speedup**)  
- Average bandwidth: **-O0 = 2,038 MB/s**, **-O2 = 3,580 MB/s** (**+75.6%**)  

### Discussion
As the stride grows, performance drops because each access is less likely to reuse data already in cache. The `-O2` build is consistently faster due to better loop/code generation (e.g., fewer overhead instructions and improved register usage).

---

## Exercise 2 - Loop reordering in matrix multiplication

### What I implemented
`exercice02/mxm.c` computes `C = A × B` for two loop nest orders on 512×512 matrices:

- **i-j-k** (classic): `B[k][j]` is effectively accessed column-wise → poor locality
- **i-k-j**: inner loop walks across `B[k][j]` row-wise → better locality

### Results
| Version | Time (ms) | Bandwidth (MB/s) | Speedup |
|---|---:|---:|---:|
| i-j-k (Standard) | 247.00 | 16,583 | 1.00× |
| i-k-j (Optimized) | 62.00 | 66,065 | **3.98×** |

### Why the order matters
On row-major arrays, visiting `B[k][j]` with `j` in the innermost loop makes memory accesses mostly sequential. That improves cache line utilization and reduces misses, which explains the large speedup.

How to run:

```bash
gcc -O2 exercice02/mxm.c -o mxm
./mxm
```

---

## Exercise 3 - Block (tiled) matrix multiplication

### What I implemented
`exercice03/mxm_bloc.c` applies blocking (tiling): it multiplies submatrices so that parts of `A`, `B`, and `C` stay hot in cache while computing a tile.

### Results
I tested block sizes from 8 to 256 on 512×512 matrices:

![Block Size Analysis](exercice03/block_size_analysis.png)

| Block Size | Time (ms) | Bandwidth (MB/s) |
|---:|---:|---:|
| 8 | 126.00 | 32,508 |
| 16 | 103.00 | 39,767 |
| 32 | 151.00 | 27,126 |
| 64 | 109.00 | 37,578 |
| 128 | 105.00 | 39,010 |
| **256** | **85.00** | **48,188** |
| No blocking | 103.00 | 39,767 |

### Discussion
Blocking trades extra loop overhead for better locality. Here, larger blocks (up to 256) performed best on my machine, likely because the working set is still manageable (and benefits from higher-level caches).

How to run (from the repository root):

```bash
gcc -O2 exercice03/mxm_bloc.c -o mxm_bloc
./mxm_bloc
python3 exercice03/plot_block_analysis.py --input mxm_bloc_results.txt --output exercice03/block_size_analysis.png --no-show
```

---

## Exercise 4 - Memory leak detection with Valgrind

### What I implemented
`exercice04/memory_debug.c` allocates an array, duplicates it, prints both, and then frees all allocations. The goal is to ensure every `malloc()` is paired with a `free()`.

### How I checked it
One way to run Valgrind without installing it locally is using a Docker container:

```bash
docker run --rm -v ${PWD}:/workspace -w /workspace ubuntu:latest bash -c \
"apt-get update && apt-get install -y gcc valgrind && \
gcc -g -o memory_debug exercice04/memory_debug.c && \
valgrind --leak-check=full ./memory_debug"
```

Expected outcome: Valgrind reports **0 bytes in use at exit** and **0 errors**.

---

## Exercise 5 - HPL benchmark (theory)

Running HPL requires access to an HPC platform (cluster/supercomputer). I did not execute it here due to lack of access at the time of writing.

---

## References

- Exercise 1: `exercice01/exercice1.c`, `exercice01/plot_results.py`
- Exercise 2: `exercice02/mxm.c`
- Exercise 3: `exercice03/mxm_bloc.c`, `exercice03/plot_block_analysis.py`
- Exercise 4: `exercice04/memory_debug.c`
- Valgrind manual: https://valgrind.org/docs/manual/
