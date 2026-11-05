# Does a reactive optimizer need to react?

Replaying recorded [OPAL](https://github.com/xingjian-bai/opal) runs ([Reactive Optimizers](https://xingjianbai.com/reactive-optimizers/), Bai, Liang, Teoh, Sitzmann, 2026) without the agent.

OPAL lets a coding agent rewrite the optimizer (`updater.py`) and its diagnostics (`analysis.py`) during training.
The authors released every program version their agents wrote ([snow-leopard/opal-results](https://huggingface.co/datasets/snow-leopard/opal-results)),
but each setting is a single run. This project asks:

1. **Seed noise:** how much do the static recipe's results vary across seeds? Which OPAL gains are larger than that?
2. **Replay:** install the recorded programs at the recorded updates, with no agent, on new seeds. How much of the gain is left?
3. **Final program only:** install only the agent's last program, at the first check. Does *when* it changed things matter?
4. **Transfer:** replay a recorded run on a shifted task. Do the discovered recipes carry over?

Replay keeps the reactivity *written into* the programs (spike guards, batch gates) and removes the agent's live reasoning.
Separating those two is the point.

Everything here runs with no API calls.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/Kaggle: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
git clone https://github.com/xingjian-bai/opal third_party/opal   # example training scripts
python scripts/download_records.py                  # ~100 MB: small_scripts + overfit, no weights
python scripts/list_records.py                      # what each record holds
```

## Running

```bash
# static recipe (no OPAL)
python -m opal_replay.run --mode plain  --out runs/diff/plain-s1  third_party/opal/examples/diffusion_mnist.py --seed 1
# wrapped, never changed: must match plain exactly (sanity check)
python -m opal_replay.run --mode keep   --out runs/diff/keep-s1   third_party/opal/examples/diffusion_mnist.py --seed 1
# the recorded agent's programs, at the recorded updates
python -m opal_replay.run --mode replay --record data/opal-results/small_scripts/op-diff-499-b \
    --out runs/diff/replay-s1 third_party/opal/examples/diffusion_mnist.py --seed 1
# only the final program
python -m opal_replay.run --mode final  --record data/opal-results/small_scripts/op-diff-499-b \
    --out runs/diff/final-s1  third_party/opal/examples/diffusion_mnist.py --seed 1
```

Several seeds and modes at once (restartable; prints mean and sd per mode):

```bash
python scripts/sweep.py --task diff --seeds 0 1 2 3 4
```

Each run writes `result.json` (the script's metric), `opal/` (OPAL's own records; `opal report runs/.../opal` makes an HTML page)
and `replay.json` (which version went in at which update).

On Windows, set `PYTHONUTF8=1` first (`$env:PYTHONUTF8=1` in PowerShell, `set PYTHONUTF8=1` in cmd), or OPAL's
HTML report fails to write (`opal/report.py:904` writes without `encoding="utf-8"`; training is not affected).

## Results so far

**MNIST flow-matching diffusion** (`op-diff-499-b`, RTX 3050, 3 seeds, EMA validation loss, lower is better):

| Mode | Mean | SD | Seeds |
|---|---|---|---|
| plain (script's AdamW, warmup then constant LR) | 0.1697 | 0.0001 | 0.1696, 0.1698, 0.1697 |
| replay (14 recorded versions, recorded updates) | 0.1624 | 0.0001 | 0.1624, 0.1625, 0.1623 |
| final (v014 only, from update 50) | 0.1622 | 0.0000 | 0.1622, 0.1621, 0.1622 |
| *recorded OPAL run (H100, seed 0)* | *0.1623* | | |

The whole gain survives without the agent, and the final program alone, used from the start, does as well as the full sequence.
v014 is a static recipe (Muon + AdamW, warmup-hold-decay, per-layer LR multipliers). Here the agent worked as a
*recipe discoverer*. Reacting during the run was not needed to use what it found.

Ablation of v014 (`python scripts/make_variants.py` builds the programs; same task, seeds and hardware):

| Program | Mean | SD | What it changes |
|---|---|---|---|
| agent-first-v001 | 0.1623 | 0.0001 | the agent's first program, written at update 50: textbook Muon + AdamW + warmup-hold-decay |
| final-no-layer-mult | 0.1622 | 0.0002 | v014 without the per-layer LR multipliers |
| final-planned-decay | 0.1623 | 0.0002 | v014 with its original decay plan, not the one re-timed at update 2000 |
| adamw-wsd-lr1e-3 | 0.1654 | 0.0003 | AdamW only, v014's schedule shape, lr 1e-3 |
| adamw-wsd-lr2e-4 | 0.1707 | 0.0001 | AdamW only, v014's schedule shape, the script's lr 2e-4 |

```bash
python scripts/sweep.py --task diff --seeds 0 1 2 --modes \
  program:variants/diff/agent-first-v001 program:variants/diff/final-no-layer-mult \
  program:variants/diff/final-planned-decay program:variants/diff/adamw-wsd-lr2e-4 program:variants/diff/adamw-wsd-lr1e-3
```

On this task the agent's first program already reaches the full gain. The later edits it based on observing training
have no measurable effect, including the per-layer multipliers and the re-timed cooldown that it credited with gains
in its own comments. Muon matters: a re-tuned AdamW with the same schedule closes only part of the gap.

**Is 0.162 a floor?** No. v001 run for 12,000 steps (twice as long, 3 seeds) reaches 0.1602 (SD 0.0001), so 0.162 is not a
hard limit of this metric. This rules out the simplest floor explanation for the ablations; it does not show that a
better program exists at 6,000 steps. Validation loss also says nothing directly about sample quality.

**CIFAR-10 with 10% wrong labels** (`op-ofit-499-b`, RTX 3050, 3 seeds, final model on the clean test set):

| Run | Test accuracy | SD | Test loss | SD | Seeds (accuracy) |
|---|---|---|---|---|---|
| plain (static AdamW, lr 3e-3) | 73.58% | 0.88 | 2.4423 | 0.0904 | 72.97, 73.17, 74.59 |
| replay (24 recorded versions, recorded updates) | 80.77% | 0.98 | 0.6305 | 0.0441 | 79.71, 80.98, 81.63 |
| *recorded static (H100, seed 0)* | *73.53%* | | *2.4935* | | |
| *recorded OPAL run (H100, seed 0)* | *80.00%* | | *0.6576* | | |

```bash
python scripts/sweep.py --task overfit --seeds 0 1 2 --modes replay plain
```

The agent wrote these programs while watching seed 0, yet all four seeds it never saw beat seed 0 (79.71%; see the 5-seed
table below). Replay beat plain by 6.74, 7.81 and 7.04 accuracy points on seeds 0, 1 and 2. Timing chosen on one run
carried over to new runs with no agent present.

Limits: `--seed` and `--data-seed` change the initial weights and the
data order; the wrong labels and the validation split are fixed by the task, so new seeds do not draw a new set of wrong
labels. The test does not cover a different dataset or noise level.

The agent's final program is a fixed schedule: `v024`'s docstring says "fixed, function of the absolute update only".
Nothing in it reads the training signal. Its final model is the mean of six parameter averages taken over cycles of high
learning rate, which is cyclic SWA (decision 25 of the run, at update 28500, names it). So the live decisions shaped the
program, but running the program needs no reaction.

What did the long path add? `variants/overfit/` holds two hand-written programs. Replay and `swa-only` ran on 5 seeds,
the other two on the first 3:

| Program | Seeds | Accuracy | SD | Test loss | SD | Accuracy per seed | What it is |
|---|---|---|---|---|---|---|---|
| replay | 5 | 80.92% | 0.75 | 0.6204 | 0.0342 | 79.71, 80.98, 81.63, 81.43, 80.83 | all 24 versions |
| swa-only | 5 | 80.63% | 0.55 | 0.6412 | 0.0202 | 80.29, 79.95, 80.57, 81.14, 81.22 | only `v024`'s destination: SGD at lr 0.1 from update 50, its last six cycles, mean of their averages, then lr 0 |
| sgd-cosine | 3 | 74.77% | 0.45 | 2.0288 | 0.0196 | 74.29, 75.18, 74.85 | the textbook CIFAR recipe with the same lr and weight decay, no averaging |
| plain | 3 | 73.58% | 0.88 | 2.4423 | 0.0904 | 72.97, 73.17, 74.59 | the script's AdamW |

```bash
python scripts/sweep.py --task overfit --seeds 0 1 2 3 4 --modes replay program:variants/overfit/swa-only
python scripts/sweep.py --task overfit --seeds 0 1 2 --modes plain program:variants/overfit/sgd-cosine
```

Paired by seed, `swa-only` minus replay is -0.28 points of accuracy (95% CI -1.24 to +0.68, n=5; replay higher on 3 of 5)
and +0.021 test loss (95% CI -0.010 to +0.051; replay lower on 4 of 5). Neither gap is distinguishable from zero, and
neither interval rules out a real edge for replay of about a point. So the one-page program reaches nearly all of the
gain with none of the snapshots, restores or earlier phases, but it is not shown to be equal. The textbook recipe gains
1.20 points over plain on the same 3 seeds.

`sgd-cosine` also differs from `swa-only` in its learning-rate schedule, so it does not isolate averaging alone.
`swa-only` is not agent-free in origin: its learning rates and cycle lengths are the values the agent found. Its design
was chosen after reading `v024`, on the same task, so this is one case study and not a general result.

The numbers behind these tables are in `results/runs.csv` (`python scripts/export_results.py` rebuilds it from `runs/`).
`train_seconds` is not a clean timing: some runs shared the GPU or ran on battery.

## Recorded runs available

| Record | Task | Metric | Static | OPAL | Switches |
|---|---|---|---|---|---|
| `small_scripts/mnist-opal` | MNIST MLP | test acc ↑ | 0.9850 | 0.9863 | 6 |
| `small_scripts/op-cifar-499-b` | CIFAR-10 ResNet-18 | test acc ↑ | 0.9449 | 0.9456 | 3 |
| `small_scripts/op-diff-499-b` | MNIST flow-matching UNet | EMA val loss ↓ | 0.1696 | 0.1623 | 14 |
| `small_scripts/op-gpt-499-b` | 33.5M GPT, FineWeb | val loss ↓ | 4.4763 | 4.3419 | 8 |
| `overfit/op-ofit-499-b` | CIFAR-10, 10% wrong labels | test loss ↓ | 2.4935 | 0.6576 | 24 |
| `overfit/op-ofit-499-te2` | same, second agent run | test loss ↓ | 2.4935 | 0.6507 | 13 |

## Caveats

- The records were made with an unreleased OPAL commit (`499fddb`); replays use the public 0.3.10. Check a seed-0 replay against the recorded curve first.
- The originals ran on H100s. Different GPUs give slightly different numbers even with the same seed, so compare against static runs on *your* hardware, not against the recorded values.
- Agent programs can be tuned to the original hardware. In `mnist-opal` (run on a Mac CPU) the agent swapped Newton-Schulz
  for an exact SVD at v2, citing slow bf16 matmuls on that CPU. On an RTX 3050 the replay takes ~990 s against ~6 s for the
  plain run. Record wall-clock time alongside the metric.
- New OPAL agent runs (to measure OPAL's own seed variance) need a paid agent and are out of scope.

## Credits

OPAL and the run records are by Xingjian Bai, Jia-Hao Liang, Jayden Teoh and Vincent Sitzmann (MIT license).
`variants/` contains programs adapted from those run records and `analysis.py` files copied from them; `LICENSE-OPAL` holds
their license. Everything else here is under `LICENSE`.
