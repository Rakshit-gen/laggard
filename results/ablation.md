## Ablation

The detection experiment again, with one part of laggard switched off at a time. Each variant gets its own thresholds from the same healthy runs, so each is held to zero false alarms there.

| detector | caught | wrong node first | no location | missed | median steps to catch | false alarms, healthy runs | false alarms, before the fault |
|---|---:|---:|---:|---:|---:|---:|---:|
| laggard | 98% | 0% | 0% | 2% | 3 | 0 | 2 |
| GPU against every GPU, not its TP peers | 96% | 0% | 0% | 4% | 3 | 0 | 2 |
| NIC blamed whenever its DP group is slow | 77% | 20% | 0% | 2% | 3 | 1 | 7 |
| nodes compared without removing replica and stage effects | 98% | 0% | 0% | 2% | 3 | 0 | 0 |
| no cap on a step's contribution | 96% | 1% | 0% | 2% | 2 | 2 | 7 |
| no cap on a learned normal | 86% | 0% | 0% | 14% | 3 | 0 | 2 |

Caught, by kind of fault:

| detector | slow GPU | slow NIC | slow node | transient GPU |
|---|---:|---:|---:|---:|
| laggard | 97% | 100% | 94% | 100% |
| GPU against every GPU, not its TP peers | 93% | 100% | 94% | 100% |
| NIC blamed whenever its DP group is slow | 97% | 29% | 94% | 100% |
| nodes compared without removing replica and stage effects | 97% | 100% | 94% | 100% |
| no cap on a step's contribution | 97% | 96% | 94% | 100% |
| no cap on a learned normal | 83% | 83% | 83% | 100% |

Named the right kind of component, of the faults caught:

| variant | kind right |
|---|---:|
| laggard | 100% |
| GPU against every GPU, not its TP peers | 86% |
| NIC blamed whenever its DP group is slow | 100% |
| nodes compared without removing replica and stage effects | 100% |
| no cap on a step's contribution | 96% |
| no cap on a learned normal | 100% |

