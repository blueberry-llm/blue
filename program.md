# autoresearch

This is an experiment to optimize a Mixture-of-Experts (MoE) model using autonomous AI research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Push the daily branch to GitHub**: `git push -u origin autoresearch/<tag>` — this creates a baseline on GitHub for the day's work.
4. **Read the in-scope files**: The repo is small. Read these files for full context:
   * `README.md` — repository context.
   * `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   * `train.py` — the file you modify. MoE model architecture, optimizer, training loop.
5. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
6. **Check `results.tsv` on master**: It already exists and contains the full history of all past experiments. Do not recreate it — just append new rows as experiments complete.
7. **Create reports directory if needed**: `mkdir reports` — this folder holds experiment report markdown files. It persists on master across days.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 40 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**

* Modify `train.py` — this is the only file you edit. Everything is fair game: MoE expert count, top-k routing, model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**

* Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
* Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
* Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val\_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 40 minutes. Everything is fair game: change the MoE configuration, the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**MoE-specific levers**: You can tune `N_EXPERTS` (number of experts per layer), `TOP_K` (how many experts each token routes to), `AUX_LOSS_WEIGHT` (load balancing strength), `ASPECT_RATIO` (model width), `HEAD_DIM` (attention head dimension), and expert hidden dimension. The router is a simple linear projection — you can experiment with more complex routing too.

**Suggested exploration order**: Start with hyperparameters (LR, batch size, warmup steps) before touching architecture. Hyperparameter changes are lower-variance and faster to evaluate. Architecture changes (layer count, model width, expert count) are higher-variance and should be backed by a clear hypothesis. Only touch routing strategy and MoE configuration after vanilla hyperparameter tuning is exhausted.

**VRAM budget**: Stay under a **20% increase** over the current baseline peak VRAM. If an experiment requires more, document the tradeoff explicitly in the report and only proceed if the expected val\_bpb improvement is substantial (>0.005). Do not let VRAM blow up dramatically chasing marginal gains.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val\_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val\_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**Diminishing returns**: If 3 consecutive experiments yield val\_bpb improvements of less than 0.002 each, escalate to a more structural change (e.g. a different routing strategy, a different attention variant, or a significant architectural shift) rather than continuing to tune hyperparameters. Incremental tuning has limits — recognize when you've hit them and make a bigger move.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Windows notes

This fork runs natively on Windows with PowerShell. Keep these in mind:

* `uv run train.py > run.log 2>&1` works correctly in PowerShell. Do not use `tee` — it behaves differently on Windows and can pollute your context.
* Use forward slashes in paths where possible to avoid escaping issues.
* Do not rely on Linux-specific shell features (e.g. `&&` chaining in cmd.exe, `$()` subshells). PowerShell syntax differs.
* If a `git` command fails with a path error, check for backslash vs. forward slash issues in the branch name or file path.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 40 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val\_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak\_vram\_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example (drawn from real MoE runs on `ultrafineweb`, RTX 4060 Ti 16GB):

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.320000	3.3	keep	baseline MoE AR=32 top-1 batch=4
b2c3d4e	1.320000	3.4	discard	top-3 routing (tied — no improvement)
c3d4e5f	1.330000	3.3	discard	top-2 routing (worse)
d4e5f6g	1.271000	9.5	discard	dense AR=64 batch=16 (better bpb but 3x VRAM — exceeds budget)
e5f6g7h	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

Each day starts from a fresh daily branch off master (e.g. `autoresearch/mar21`). Individual experiments each get their own branch off that daily branch. **Improvements that beat the current best are always promoted to master** — the daily branch is just a working scratchpad for the day, not the destination for keeper changes.

### Daily setup

At the start of each day:

1. Check out master: `git checkout master && git pull`.
2. Create today's daily branch: `git checkout -b autoresearch/<tag>` (e.g. `autoresearch/mar21`).
3. Push it: `git push -u origin autoresearch/<tag>`.
4. **Run the baseline first**: Before any experiments, run `train.py` as-is to establish today's starting val\_bpb. Log it to `results.tsv` with status `keep` and description `daily baseline`. This is the number every experiment today must beat.

### LOOP FOREVER:

1. **Check git state**: confirm which branch you're on and what the current best val\_bpb is (from `results.tsv`).
2. **Formulate a hypothesis**: Scan `results.tsv` and identify: (a) what has worked and by how much, (b) what failed and why, (c) any near-misses worth revisiting with a twist. Prefer ideas that build on successful patterns rather than random exploration. Articulate clearly why you expect the change to improve val\_bpb before touching any code.
3. **Create an experiment branch** from the daily branch: `git checkout -b autoresearch/<tag>-<exp-name>` (e.g. `autoresearch/mar21-increase-lr`).
4. **Edit `train.py`** with the experimental change.
5. **Commit the change**: `git add train.py && git commit -m "<exp-name>: <one-line description>"`.
6. **Run the experiment** — always saving stdout to the `runs/` folder with the same name as the experiment report (e.g. `runs/<exp-branch-name>.log`):
    ```
    mkdir -p runs
    uv run train.py > runs/<exp-branch-name>.log 2>&1
    ```
    Do NOT use `tee`. Allow up to 45 minutes before killing the process.
7. **Read results**: `grep "^val_bpb:\|^peak_vram_mb:" runs/<exp-branch-name>.log`.
8. **If grep is empty**, the run crashed. Run `tail -n 50 runs/<exp-branch-name>.log` for the stack trace and attempt a fix. **Allow at most 2 fix attempts.** If still broken, log `crash`, revert `train.py` with `git checkout -- train.py`, and move on.
9. **Log to `results.tsv`** — **always**, regardless of whether the experiment improved, tied, or crashed. Every run gets a row.
10. **Write the experiment report** at `reports/<exp-branch-name>.md`:
    * **Hypothesis**: What you were testing and why you expected improvement.
     * **Results**: val\_bpb, memory usage, steps, and other key metrics from `runs/<exp-branch-name>.log`.
    * **Expected**: Did results match your hypothesis? (yes / no / partial)
    * **Outcome**: Improved, tied, worse, or crash.
    * **Next ideas**: 1–2 natural follow-on experiments worth trying. (Log them here for continuity — they don't need to run now.)
11. **Push the experiment branch** — always, regardless of outcome. The `runs/<exp-branch-name>.log` is committed here but will NOT be carried to master:
    ```
    git add results.tsv reports/<exp-branch-name>.md runs/<exp-branch-name>.log
    git commit -m "results and report: <exp-name>"
    git push -u origin autoresearch/<tag>-<exp-name>
    ```
12. **Promote to master if improved**: If val\_bpb is lower than the current best on master, cherry-pick the `train.py` change onto master and push:
    ```
    git checkout master
    git cherry-pick <commit-hash-of-train.py-change>
    git push origin master
    git checkout autoresearch/<tag>
    ```
    Do NOT carry the run log to master. Only `train.py` (and any other source changes) are promoted — not logs.
13. **Always push `results.tsv` and `reports/` to master** after every experiment, win or lose. This keeps master as the permanent lab notebook:
    ```
    git checkout master
    git checkout autoresearch/<tag> -- results.tsv reports/
    git add results.tsv reports/ && git commit -m "log: <exp-name>" && git push
    git checkout autoresearch/<tag>
    ```
14. **Switch back to the daily branch**: `git checkout autoresearch/<tag>` and loop.

### What lives where

| Artifact | Experiment branch | Master |
|---|---|---|
| `train.py` changes | ✅ committed | ✅ only if improved |
| `runs/<exp>.log` | ✅ committed | ❌ never pushed |
| `results.tsv` row | ✅ every run | ✅ every run |
| `reports/<exp>.md` | ✅ every run | ✅ every run |

**Timeout**: Each experiment takes ~40 minutes + a minute or two of overhead. If a run exceeds 60 minutes, kill it and treat it as a crash.

**Crashes**: Allow at most **2 fix attempts**. If the idea is fundamentally broken or you've hit 2 failed fixes, log `crash`, revert `train.py`, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — scan `results.tsv` for near-misses to revisit, re-read the in-scope files for new angles, try combining previous successful changes, try more radical architectural changes. The loop runs until the human interrupts you, period.

