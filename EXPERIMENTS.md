# Experiments log — amortized differentiable pulse planner

The amortized planner maps a target unitary `U_target` to a fixed-length pulse
sequence in one forward pass, trained by backpropagating the analytic gradient
of the gate infidelity through a differentiable `torch.matrix_exp` rollout
(`algos/amortized.py`). One network solves any target; no per-target
re-optimization.

## What changed and why it improved

### 1. Keep the rollout shallow (`seq_len ≈ 2–3·d`)
Backpropagating through too many stacked `matrix_exp` pulses weakens the
target-conditioning gradient. The optimizer then settles into the trivial
**target-independent** solution (emit a constant sequence), whose expected
fidelity against Haar targets is exactly `1/d²`.

- `d=3, seq_len=20` → stuck at infidelity ≈ 0.88 (fidelity ≈ 1/9 = 0.11). Dead.
- `d=3, seq_len=6`  → fidelity **0.9963** (infidelity 3.7e-3).

Parameter counting: `SU(d)` has `d²−1` real parameters and each pulse supplies
≈ `d` (`d−1` phases + 1 angle), so `seq_len ≈ 2·d` already over-parameterizes the
problem while staying shallow enough to train.

### 2. Target-difficulty curriculum (essential for `d ≥ 4`)
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

| d | seq_len | curriculum | eval fidelity | eval infidelity |
|---|---------|------------|---------------|-----------------|
| 3 | 6       | no         | 0.9963        | 3.7e-3          |
| 4 | 12      | yes        | 0.877         | 0.123           |

Trivial (target-independent) floor is `1/d²` = 0.111 (d=3), 0.0625 (d=4); both
runs clear it, confirming the planner conditions on the target.

## Reproduce

```bash
./scripts/gcp_run.sh run     # runs everything in scripts/experiments.txt
./scripts/gcp_run.sh poll    # pull latest metrics + plot
```

Sanity check that the current code is running (not a stale build):
- output files have **no** `pulsepen` suffix,
- logs print `train_fid`/`eval_fid` with **no** `E[pulses]` field.
