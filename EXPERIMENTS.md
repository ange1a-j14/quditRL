# Experiments log — amortized differentiable pulse planner

The amortized planner maps a target unitary `U_target` to a fixed-length pulse
sequence in one forward pass, trained by backpropagating the analytic gradient
of the gate infidelity through a differentiable `torch.matrix_exp` rollout
(`algos/amortized.py`). One network solves any target; no per-target
re-optimization.

## LEssons learned

### 1. Keep the rollout shallow (`seq_len ≈ 2–3·d`)
Backpropagating through too many stacked `matrix_exp` pulses weakens the
target-conditioning gradient. The optimizer then settles into the trivial
**target-independent** solution (emit a constant sequence), whose expected
fidelity against Haar targets is exactly `1/d²`. Fails to learn the intended moves.

- `d=3, seq_len=20` → stuck at infidelity ≈ 0.88 (fidelity ≈ 1/9 = 0.11). Dead.
- `d=3, seq_len=6`  → fidelity **0.9963** (infidelity 3.7e-3).

Parameter counting: `SU(d)` has `d²−1` real parameters and each pulse supplies
≈ `d` (`d−1` phases + 1 angle), so `seq_len ≈ 2·d` already over-parameterizes the
problem while staying shallow enough to train.

### 2. Target-difficulty curriculuum (essential for `d ≥ 4`)
Plain Haar training at `d=4` also got stuck at the `1/d²` minimum — the target
is too hard to ever produce a useful gradient from a random init. Fix: for the
first `curriculum_frac` of training, sample targets as **random pulse circuits
of annealed depth** (`curriculum_start_pulses → seq_len`). These are guaranteed
reachable and start near identity, giving a strong conditioning signal; training
then graduates to real Haar targets.

- `d=4, seq_len=12` plain Haar → fidelity ≈ 0.75 (≈ trivial floor). Dead.
- `d=4, seq_len=12` + curriculum → fidelity **0.877** (infidelity ≈ 0.12).

### 3. Cosine LR decay + gradient clipping
d=4 train fidelity (0.93) sat above eval (0.88) and was still descending at the
end of training. Added:
- **Cosine LR decay** to `lr·lr_min_frac` over training — lowers the
  end-of-training infidelity plateau and reduces eval noise.
- **Gradient clipping** (`grad_clip`, global norm) — stabilizes the deeper
  rollouts needed at higher `d`.

### Removed
The earlier PonderNet-style **halting head** (variable length + per-pulse
penalty) was dropped. With no penalty it collapsed all probability mass onto the
1-pulse prefix (`eval_pulses` pinned at 1.0), starving the planner. Fixed-length
training cannot collapse this way. (Early-exit can be revisited later as a
post-hoc threshold on prefix fidelity rather than a learned discrete head.)

## Results so far

Completed runs (cosine LR decay + grad clipping on):

| d | seq_len | curriculum | eval fidelity | eval infidelity | 1/d² floor |
|---|---------|------------|---------------|-----------------|-----------|
| 3 | 6       | no         | 0.9985        | 1.5e-3          | 0.111     |
| 4 | 12      | yes        | 0.891         | 0.109           | 0.0625    |
| 5 | 15      | yes        | 0.895         | 0.105           | 0.040     |

All of d=3–5 clear the trivial floor by a wide margin.

### High-d sweep (with identity-block init, `seq_len = 2·d`, `curriculum_frac=0.6`)

| d | seq_len | status | eval fidelity |
|---|---------|--------|---------------|
| 6 | 12      | validation only (full run not yet pulled) | 0.44 @ 2.5k iters, climbing |
| 7 | 14      | not yet pulled | — |
| 8 | 16      | partial (stopped at 42.4k / 70k iters) | ~0.44, still climbing |

Note: d=6/d=7 full-run CSVs are still on the VM (not pulled). d=8's local CSV is a
partial run that was interrupted before convergence, so its 0.44 is a lower bound.

### The d=6 collapse at seq_len=18 (what motivated identity init)

In the first sweep, **d=6 at seq_len=18 (=3·d) collapsed** straight to the 1/d²
floor and never recovered. `3·d` rollouts stay trainable through d=5 (seq 15) but
seq 18 crossed a threshold. Dropping
to `seq_len=2·d` did *not* help: d=6 stayed at the floor even at seq 12 (shallower
than d=5's working seq 15), and it failed even on near-identity depth-1/2
curriculum targets. That ruled out depth and pointed at a **barren plateau** —
at Hilbert dimension d=6 with ~12 stacked random unitary layers the infidelity
gradient vanishes at a random init.

### 4. Identity-block initialization
Scale down the planner's final pulse layer (`weight *= 1e-2`, `bias = 0`) so the
composed unitary starts at `U ≈ I`. The near-identity curriculum targets then sit
right next to the init with a strong, non-vanishing gradient. d=6 went from stuck
at 0.027 to **eval fidelity 0.44 within 2,500 iters** and climbing — plateau
escaped. (d=6–8 still use the shallow `seq_len=2·d` and `curriculum_frac=0.6`.)

## Reproduce on GCP

```bash
./scripts/gcp_run.sh run     # runs everything in scripts/experiments.txt
./scripts/gcp_run.sh poll    # pull metrics + plot for every experiment in the run
```

`poll` was fixed this session: it previously grepped the log *contents* as if they
were a filename (so it found no CSV and pulled nothing mid-sweep), and it only
ever pulled the last experiment. It now pulls a live CSV/plot for **every**
experiment in the latest log.

Sanity-check you're on the current code (not a stale build):
- output files have **no** `pulsepen` suffix,
- logs print `train_fid`/`eval_fid` with **no** `E[pulses]` field.
