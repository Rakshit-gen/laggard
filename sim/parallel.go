package sim

import (
	"runtime"
	"sync"
)

// parallel runs f for 0..n-1 on every core and returns the results in
// order.
func parallel[T any](n int, f func(int) T) []T {
	out := make([]T, n)
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range runtime.NumCPU() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				out[i] = f(i)
			}
		}()
	}
	for i := range n {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return out
}
