## Detection

Job: 128 GPUs (4 replicas x 4 pipeline stages x 8-way tensor parallel), 3000 steps per run. Thresholds were set on 32 healthy runs; false alarms are counted on 32 different healthy runs (96000 steps) and before the fault in every faulty run. Faulty runs: 84, one fault each, 6 per kind and size.

A fault counts as caught when the first alarm after it starts names its node. "Wrong node" means the first alarm named a healthy one. Transient faults count if flagged before they end plus 20 steps.

| detector | caught | wrong node first | no location | missed | median steps to catch | false alarms, healthy runs | false alarms, before the fault |
|---|---:|---:|---:|---:|---:|---:|---:|
| laggard | 98% | 0% | 0% | 2% | 3 | 0 | 2 |
| step-time alert | 0% | 0% | 27% | 73% | - | 1 | 0 |
| last-arriver blame | 52% | 0% | 0% | 48% | 14 | 0 | 0 |
| NVRx-style, defaults (0.7, 300 s) | 7% | 0% | 0% | 93% | 271 | 0 | 0 |
| NVRx-style, calibrated, 300 s | 50% | 0% | 0% | 50% | 296 | 0 | 0 |
| NVRx-style, calibrated, 30 s | 58% | 0% | 0% | 42% | 30 | 0 | 0 |

Caught, by kind of fault:

| detector | slow GPU | slow NIC | slow node | transient GPU |
|---|---:|---:|---:|---:|
| laggard | 97% | 100% | 94% | 100% |
| step-time alert | 0% | 0% | 0% | 0% |
| last-arriver blame | 60% | 58% | 0% | 100% |
| NVRx-style, defaults (0.7, 300 s) | 20% | 0% | 0% | 0% |
| NVRx-style, calibrated, 300 s | 80% | 0% | 100% | 0% |
| NVRx-style, calibrated, 30 s | 83% | 0% | 100% | 50% |

laggard by fault size. The first run of each persistent size has the fault from step 0; "kind right" means it also said GPU, NIC or node correctly.

| fault | slowdown | caught | kind right | median steps to catch | caught when present from step 0 |
|---|---:|---:|---:|---:|---:|
| slow GPU | 1.03x | 5 of 6 | 5 of 5 | 3 | 0 of 1 |
| slow GPU | 1.05x | 6 of 6 | 6 of 6 | 3 | 1 of 1 |
| slow GPU | 1.10x | 6 of 6 | 6 of 6 | 3 | 1 of 1 |
| slow GPU | 1.20x | 6 of 6 | 6 of 6 | 3 | 1 of 1 |
| slow GPU | 1.50x | 6 of 6 | 6 of 6 | 3 | 1 of 1 |
| slow NIC | 1.25x | 6 of 6 | 6 of 6 | 1 | 1 of 1 |
| slow NIC | 1.50x | 6 of 6 | 6 of 6 | 1 | 1 of 1 |
| slow NIC | 2.00x | 6 of 6 | 6 of 6 | 1 | 1 of 1 |
| slow NIC | 4.00x | 6 of 6 | 6 of 6 | 1 | 1 of 1 |
| slow node | 1.05x | 5 of 6 | 5 of 5 | 4 | 0 of 1 |
| slow node | 1.10x | 6 of 6 | 6 of 6 | 4 | 1 of 1 |
| slow node | 1.20x | 6 of 6 | 6 of 6 | 4 | 1 of 1 |
| transient GPU | 1.50x | 6 of 6 | 6 of 6 | 3 | - |
| transient GPU | 2.00x | 6 of 6 | 6 of 6 | 3 | - |

Thresholds after calibration:

- laggard: 10.6, 14.8, 4.47
- step-time alert: 0.128
- last-arriver blame: 0.06
- NVRx-style, defaults (0.7, 300 s): package defaults
- NVRx-style, calibrated, 300 s: 0.0435
- NVRx-style, calibrated, 30 s: 0.0565

