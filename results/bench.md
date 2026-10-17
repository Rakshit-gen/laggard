## Cost of reading a step

Machine: Apple M4, go1.25.4. One goroutine. Each job is warmed for 150 steps, then 200 steps are timed one call at a time.

| GPUs | layout (D x S x G) | operations per step | member records per step | Observe p50 | Observe p99 | per record | localizer memory |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | 4 x 4 x 8 | 2080 | 7296 | 0.080 ms | 0.094 ms | 11.0 ns | 0.39 MB |
| 1024 | 16 x 8 x 8 | 18496 | 62464 | 0.772 ms | 0.847 ms | 12.4 ns | 3.06 MB |
| 4096 | 32 x 16 x 8 | 77952 | 258048 | 3.439 ms | 3.804 ms | 13.3 ns | 12.34 MB |

Memory is heap growth from creating the localizer and running it, measured after a GC.

