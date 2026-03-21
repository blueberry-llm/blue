# autoresearch-windows

> Turn your Windows gaming PC into an autonomous AI researcher.

This is a Windows-native fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), built to run on desktop consumer NVIDIA GPUs without any unofficial Triton-on-Windows stacks. If you have an RTX card and want to run autonomous LLM research experiments overnight, this is the repo for you.

---

## What is this?

Give an AI agent a real (but small) LLM training setup and let it experiment autonomously while you sleep. The agent modifies `train.py`, runs a 40-minute training job, checks whether the result improved, keeps or discards the change, and loops forever. You wake up in the morning to a log of experiments and (hopefully) a better model.

The key insight: **you never touch the Python files**. Instead, you edit `program.md` — a Markdown file that acts as the agent's instruction manual and research org charter. The agent reads it, follows the experiment loop, and writes up reports. You iterate on `program.md` over time to make the agent smarter and more systematic.

This fork's `program.md` includes improvements over the upstream baseline:
- **Hypothesis discipline** — the agent must review past results before proposing each new idea
- **Exploration order** — hyperparameters first, architecture later, MoE config last
- **Concrete VRAM ceiling** — 20% over baseline; >0.005 gain required to justify exceeding it
- **Diminishing returns trigger** — escalates to structural changes after 3 consecutive small gains
- **Crash fix limit** — hard cap of 2 fix attempts per crash, then revert and move on
- **Richer experiment reports** — each report includes "Next ideas" for continuity
- **Windows-specific notes** — PowerShell quirks, path handling, no `tee`
- **Session stopping criteria** — guidance for shorter human-supervised runs

More context from the original project: [@karpathy's tweet](https://x.com/karpathy/status/2029701092347630069).

---

## Is my GPU supported?

This fork supports desktop consumer NVIDIA GPUs on Windows with a tiered VRAM floor. Laptop GPUs are not officially supported due to wide power and thermal variance.

| Architecture | Min VRAM | Supported GPUs |
|---|---|---|
| Turing | ≥ 8 GB | RTX 2060 12GB, RTX 2060 SUPER, RTX 2070, RTX 2070 SUPER, RTX 2080, RTX 2080 SUPER, RTX 2080 Ti |
| Ampere | ≥ 10 GB | RTX 3060 12GB, RTX 3080 10GB/12GB, RTX 3080 Ti, RTX 3090, RTX 3090 Ti |
| Ada | ≥ 10 GB | RTX 4060 Ti 16GB, RTX 4070, RTX 4070 SUPER, RTX 4070 Ti, RTX 4070 Ti SUPER, RTX 4080, RTX 4080 SUPER, RTX 4090 |
| Blackwell | ≥ 10 GB | RTX 5060 Ti 16GB, RTX 5070, RTX 5070 Ti, RTX 5080, RTX 5090 |

**Not supported:** RTX 2060 6GB (below VRAM floor), all laptop GPUs, AMD/ROCm, Apple Metal, multi-GPU setups.

Tested hardware: RTX 3080 10GB (upstream), RTX 4060 Ti 16GB (this fork). Other listed SKUs are matrix-supported but may be less field-tested.

The runtime path is unified across all supported GPUs: PyTorch SDPA attention + eager execution. No FA3, no `torch.compile`, no unofficial Triton.

---

## How it works

The repo has three files that matter:

| File | Who edits it | What it does |
|---|---|---|
| `prepare.py` | Nobody — read-only | Fixed constants, one-time data prep, BPE tokenizer training, dataloader, evaluation harness |
| `train.py` | The AI agent | MoE GPT model, Muon + AdamW optimizer, training loop. Everything is fair game. |
| `program.md` | You | Agent instructions, experiment loop rules, research org charter |

Training always runs for a **fixed 40-minute wall-clock budget** (excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better. Because the time budget is fixed, the agent automatically finds the best model for *your specific GPU* — not some reference H100.

The current architecture is a Mixture-of-Experts (MoE) GPT with 4 experts per layer and top-1 routing. The agent can change anything: expert count, routing strategy, model width, optimizer, batch size, hyperparameters — whatever it thinks will improve val_bpb.

---

## Quick start (PowerShell)

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), a supported NVIDIA GPU with up-to-date drivers.

```powershell
# 1. Install uv (if you don't already have it)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install dependencies
uv sync

# 3. One-time data download and tokenizer training
#    Default: ultrafineweb dataset. Add --dataset tinystories for a smaller profile.
uv run prepare.py

# 4. Verify your setup with a quick smoke test (~2 min)
uv run train.py --smoke-test

# 5. Run a full training experiment manually (~40 min)
uv run train.py
```

If the smoke test passes, your setup is good. Move on to running the agent.

---

## Running the agent

Open this repo in Claude, Codex, or any agentic AI tool. Point it at `program.md` and kick it off:

```
Have a look at program.md and let's kick off a new experiment — start with the setup.
```

The agent will:
1. Create a dated branch (e.g. `autoresearch/mar21`)
2. Read the repo files for context
3. Establish a baseline by running `train.py` as-is
4. Loop forever: hypothesize → modify → train → evaluate → record → report → repeat

Each experiment takes ~40 minutes. Running overnight gives you roughly 10–12 experiments. You wake up to a full lab notebook in the `reports/` folder and a `results.tsv` tracking every run.

**Tip:** Disable auto-permissions in your agent tool before starting. The agent should only be able to edit `train.py`, run `uv` commands, and use `git` — nothing else.

---

## Autotune

The runtime includes a short eager-mode autotune pass that selects the best batch size and checkpointing strategy for your specific GPU fingerprint (compute capability, VRAM tier, BF16/TF32 support). Results are cached so subsequent runs start immediately.

You can control this with environment variables:

```powershell
# Skip the autotune probe entirely
$env:AUTORESEARCH_DISABLE_AUTOTUNE = "1"; uv run train.py

# Force a fresh autotune (clears cache)
$env:AUTORESEARCH_AUTOTUNE_REFRESH = "1"; uv run train.py
```

---

## Project structure

```
prepare.py       — data prep, tokenizer, dataloader, evaluation (do not modify)
train.py         — model, optimizer, training loop (agent modifies this)
program.md       — agent instructions and experiment loop rules (you modify this)
pyproject.toml   — dependencies
results.tsv      — experiment log (tab-separated)
reports/         — per-experiment markdown reports (the lab notebook)
analysis.ipynb   — optional: explore results interactively
```

---

## Fork scope

This fork exists solely to make `karpathy/autoresearch` work natively on Windows with desktop consumer NVIDIA GPUs. Changes are scoped to compatibility and stability on that target platform.

- **Upstream:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- **Forked via:** [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx)
- **What's changed:** PyTorch SDPA attention instead of FA3, eager execution instead of `torch.compile`, Windows PowerShell compatibility, tiered VRAM policy, autotune for consumer GPU profiles, improved `program.md`
- **What's removed:** The Linux/H100-oriented fast path from upstream is not present in this fork
- **Non-goals:** FA3/H100 paths, Triton-on-Windows, AMD/ROCm, Apple Metal, multi-GPU training

If you need the upstream Linux/H100 path, use [karpathy/autoresearch](https://github.com/karpathy/autoresearch) directly.

---

## License

MIT
