## Draining decisions

Each policy acts on its own alarms: the node it names is drained and the job restarts from a checkpoint on a spare, which costs the whole job a fixed number of seconds. 42 one-fault runs of 3000 steps, 3 per kind and size. Loss is extra job time over the same run with no fault; regret is extra time over the oracle, which knows the fault and picks the cheaper of draining the moment it starts and never draining. laggard is told how many steps the job has left and waits 50 steps before draining.

Restart costs 120 s (about 125 steps):

| policy | mean loss | mean regret | drains | wrong node or no fault | drained during a transient |
|---|---:|---:|---:|---:|---:|
| never drain | 4.20% | 2.67% | 0 | 0 | 0 |
| oracle | 1.53% | 0.00% | 9 | 0 | 0 |
| laggard | 1.97% | 0.45% | 14 | 0 | 0 |
| laggard, ski rental only | 2.44% | 0.92% | 9 | 0 | 0 |
| laggard, drain on first alarm | 4.58% | 3.06% | 45 | 6 | 6 |
| last-arriver blame | 2.72% | 1.19% | 22 | 0 | 6 |
| NVRx-style, defaults (0.7, 300 s) | 3.71% | 2.18% | 3 | 0 | 0 |
| NVRx-style, calibrated, 300 s | 4.90% | 3.38% | 21 | 0 | 0 |
| NVRx-style, calibrated, 30 s | 5.27% | 3.74% | 27 | 2 | 2 |

Restart costs 600 s (about 625 steps):

| policy | mean loss | mean regret | drains | wrong node or no fault | drained during a transient |
|---|---:|---:|---:|---:|---:|
| never drain | 4.20% | 0.71% | 0 | 0 | 0 |
| oracle | 3.49% | 0.00% | 3 | 0 | 0 |
| laggard | 3.56% | 0.08% | 3 | 0 | 0 |
| laggard, ski rental only | 4.94% | 1.45% | 3 | 0 | 0 |
| laggard, drain on first alarm | 22.51% | 19.02% | 45 | 6 | 6 |
| last-arriver blame | 11.48% | 8.00% | 22 | 0 | 6 |
| NVRx-style, defaults (0.7, 300 s) | 4.91% | 1.42% | 3 | 0 | 0 |
| NVRx-style, calibrated, 300 s | 13.27% | 9.78% | 21 | 0 | 0 |
| NVRx-style, calibrated, 30 s | 16.03% | 12.54% | 27 | 2 | 2 |

