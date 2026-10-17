# How laggard decides

This is the long version of the README's "the idea" section. It follows one step's trace through the localizer, says why each piece is there, and what happens without it. Numbers come from `results/`.

## What it reads

For every communication operation in a step, each member reports when it entered and when it left, read from its own host's clock. That's all. No GPU counters, no NCCL internals, no synchronized clocks.

The job in the simulator is 128 GPUs: 4 data-parallel replicas, each a 4-stage pipeline, each stage an 8-GPU tensor-parallel group filling one node. Every GPU has its own NIC. Three kinds of operations show up:

- **TP all-reduce**: inside one node, after every chunk of compute.
- **PP send/recv**: between neighbouring stages, through the two GPUs' NICs.
- **DP all-reduce**: the gradient sync at the end of the step, across the replicas, through each member's NIC.

## Why the obvious signals don't work

A synchronous job hides its stragglers. Every collective ends at the same moment for all its members, so every rank's step takes the same time. Per-rank step time tells you nothing.

What does differ is how each rank spent the step. Two common ways to use that:

- **Last-arriver blame.** In each collective, the member who arrived last made the others wait, so charge it their waiting. This works for a slow GPU. It fails for a slow NIC: the NIC's DP all-reduce runs long for all four of its members, so all four start the next step late and all four collect blame. Three of them are on healthy nodes.
- **Compute against the median rank** (the NVRx-style scores). This works for a GPU that is a lot slower than everyone else. It can't see a NIC at all, and comparing against the whole job mixes in variation that has nothing to do with hardware: each replica sees different data, so its compute moves by a few percent from step to step.

## Step 1: split every rank's time

For each rank, the localizer walks its operations in order. The gap between leaving one operation and entering the next is compute. Both ends of that subtraction come from the same clock, so clock offsets between hosts drop out. `TestClockOffsetsDontMatter` shifts one host by 250 ms and checks that every score is unchanged.

Inside each operation, the member who arrived last didn't wait, so its time inside is the transfer alone. That's `min(exit - enter)` over the members, again one clock per subtraction.

Each step then gives:

- compute seconds per rank,
- transfer seconds per DP group,
- transfer seconds per PP link, summed over the step's microbatches.

## Step 2: compare with the right peers

Each kind of evidence is compared with things that should look the same:

- **A GPU against its TP peers.** The eight GPUs in a node run the same layers on the same data, so data-dependent variation cancels exactly. The residual is `log(compute) - median(log(compute) over the node)`.
- **A node against other nodes.** Its compute is the median of its eight GPUs, which one slow GPU can't move. Nodes differ by replica (data) and by stage (layers), so one pass of median polish removes a replica effect and a stage effect first. In this simulator that turned out not to matter (see the ablation). It would with stronger data-dependent variation, and it costs nothing.
- **A DP group against the other DP groups at the same stage**, which move the same number of bytes.
- **A PP link against all PP links.**

Turning off the TP-peer comparison and comparing every GPU with the whole job still catches most slow GPUs. But a slow node then looks like eight slow GPUs, and laggard names the wrong kind of component for 14% of the faults it catches.

## Step 3: a CUSUM per component

Each component has a learned normal (`mu`, its usual residual) and a noise level (`sigma`, shared across its family, from the median absolute residual across components each step, smoothed). Each step:

```
z = clip((residual - mu) / sigma, -4, 4)
S = max(0, S + z - 1)
```

An alarm goes up when `S` crosses the family's threshold.

The clip at 4 is what stops one long hiccup (a GC pause, a page-fault storm) from raising an alarm alone. It takes at least a few slow steps in a row. Without the clip, the healthy runs set a higher threshold to stay quiet, and false alarms before the fault go from 2 to 7.

The allowance of 1 is a noise unit. My first version used 0.5. That made the statistic creep up on healthy hardware whenever the learned normal was off by half a noise unit, which a 50-step warm-up easily gets wrong. With 1 and a 100-step warm-up, healthy jobs stay quiet.

**Learning the normal.** `mu` starts as the median residual over the warm-up. After that it follows slow drift while the component is unsuspected (`S` below 5). My first version adapted only while `S` was exactly zero, which sounds equivalent but isn't: `S` only touches zero right after below-normal steps, so `mu` drifted low and healthy components alarmed.

**The cap on normal.** `mu` can't go above `log(1.05)`. A GPU that is already 20% slow when the job starts would otherwise learn its slowness as normal during warm-up and never be flagged. With the cap, anything over 5% slower than its peers is a fault however long it's been that way. Without the cap, laggard catches none of the 12 faults present from step 0 (10 of 12 with it), and 86% of faults overall instead of 98%. The price is that a GPU born 5% slow gets flagged. In the simulator, built-in GPU speeds vary by 1%, so that's rare. On real hardware, set the cap from the spread you see.

**Thresholds** are set per family (GPU, node, NIC) from healthy runs: the highest score each family reached across 32 healthy jobs, plus 5%. Every baseline gets the same treatment, so every detector is held to zero false alarms on those jobs.

## Step 4: a NIC is the one in all its slow groups

A DP group spans four NICs and a PP link spans two. When one NIC is slow, its DP group and both of its PP links are slow. Each of the other three NICs in that DP group has its PP links healthy, and the NICs at the far end of each slow PP link have their own DP groups healthy. So a NIC's score is the minimum of the CUSUMs of every group it belongs to. Only the bad NIC has all of them high.

This is the single biggest piece. Blaming a NIC whenever its DP group is slow catches 29% of slow NICs on the right node instead of 100%, and points at the wrong node first for 20% of all faults.

## Step 5: what is it costing the job?

Draining a node and restarting from a checkpoint costs the whole job a few minutes. Whether that's worth it depends on what the slow component costs the *job*, and that isn't its own extra time.

In a noisy job, some path is already the slowest every step: a replica with longer sequences, a GC pause somewhere, a congested link. A GPU that is 10% slow mostly finishes before that path does. In the simulator it adds 0.3% to the step. One that is 50% slow sets the pace and adds 17%. My first version charged the job the component's own extra seconds, about 20 times the real cost for the 10% GPU, and drained nodes that weren't worth draining.

laggard measures the cost instead by replaying the step. The trace has every member's compute before each operation and every operation's transfer time. A blocking collective's transfer starts when its last member arrives. So the step can be rebuilt operation by operation: once as recorded, and once with the suspect's compute (or its groups' transfers) scaled back to its learned normal. The difference in finish time is what the suspect cost the job this step, waiting included. `TestExcessIsTheJobsCost` checks this against the simulator's ground truth:

| fault | replay estimate | true cost |
|---|---:|---:|
| GPU 1.1x | 0.0038 s/step | 0.0032 s/step |
| GPU 1.5x | 0.172 | 0.165 |
| NIC 2x | 0.075 | 0.059 |
| node 1.2x | 0.067 | 0.068 |

The NIC estimate runs high because a slow DP all-reduce delays the next step too, and some of that overlaps with what the replay credits to this step.

A replay costs one pass over the step's records. It only runs for components that are alarmed, so a healthy step pays nothing extra.

## Step 6: drain or wait

Two things are unknown when an alarm goes up: whether the slowdown will end on its own, and how long the job has left.

- **If the job's remaining steps are known**, which they are for any training run with a fixed token budget: wait `Patience` steps (50 here, longer than the simulated transients), then drain if the current cost per step, multiplied by the steps left, exceeds a restart. The cost per step follows only the last five steps. My first version smoothed over 20, so after a transient ended the cost stayed high for a while and the node was drained after the fault was gone. `TestTransientIsWaitedOut` covers that.
- **If not**, fall back to ski rental: drain once the alarm has cost one restart's worth. That never pays more than twice what the best choice in hindsight would have, whatever the slowdown does next. It's the right rule when nothing is known, and a poor one when the horizon is known: its regret against the oracle was 0.92% with a 120 s restart and 1.45% with a 600 s one, against 0.45% and 0.08% for the horizon rule.

## What it costs to run

Observe on a 128-GPU step takes 0.08 ms at the median, about 11 ns per member record. At 4096 GPUs it's 3.4 ms, against steps that take on the order of a second. Memory is a few floats per component plus the warm-up samples: 0.4 MB at 128 GPUs and 12 MB at 4096. See `results/bench.md`.

## What this design doesn't do

- **It only knows the three kinds of component in the simulator.** A slow NVLink would show up as a TP transfer anomaly, which nothing scores yet. A whole slow switch or rail would make many NICs slow at once, and the intersection rule would blame none of them.
- **The replay assumes blocking collectives.** With overlapped communication, compute and transfer run at the same time and the gaps no longer equal compute.
- **One fault at a time.** Two slow components on the same node are drained together, which is right. Two on different nodes are handled one per restart.
- **Thresholds come from healthy runs of the same job.** A different model, batch size or cluster needs its own healthy runs to calibrate on.
