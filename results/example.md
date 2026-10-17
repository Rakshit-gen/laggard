## One slow NIC

The NIC of rank 77 (replica 2, stage 1, position 5, node 9) runs at two thirds of its speed from step 800. Every DP all-reduce and pipeline transfer through it takes 1.5x as long. First alarm after the fault:

| detector | first alarm |
|---|---|
| laggard | step 801, 1 after the fault: NIC 77 on node 9 |
| step-time alert | none in 600 steps |
| last-arriver blame | none in 600 steps |
| NVRx-style, defaults (0.7, 300 s) | none in 600 steps |
| NVRx-style, calibrated, 300 s | none in 600 steps |
| NVRx-style, calibrated, 30 s | none in 600 steps |

