# CPU Scheduler — Distributed Scheduling via Ascending Auctions

Simulation of **distributed CPU time-slot allocation** using the **ascending auction algorithm** from Shoham & Leyton-Brown, *Multiagent Systems* (Ch. 2, §2.3.3). Jobs (agents) bid for contiguous time slots under deadline and length constraints; the auction finds a feasible allocation and slot prices via an iterative, price-increment protocol (ε-step).

## Requirements

- Python 3.10+
- See `requirements.txt`: `numpy`, `matplotlib`, `pandas`

```bash
pip install -r requirements.txt
```

## Quick Start

From the project root:

```bash
python main.py
```

Runs all built-in experiments (book examples 1–3, then epsilon sensitivity over all scenarios). Use `--save` to write plots to `output/`.

## Reproducing Experiments

### Single examples (with optional ε and 2-CPU)

| Example | Description | Command |
|--------|-------------|--------|
| **1** | Book 8-slot, 4 jobs (9am–5pm) | `python main.py --example 1 [--eps 0.25] [--two-cpus] [--save]` |
| **2** | Book 2-slot, 2 jobs | `python main.py --example 2 [--eps 0.25] [--two-cpus] [--save]` |
| **3** | Suboptimal 2-slot example | `python main.py --example 3 [--save]` |
| **4** | Many jobs (8 jobs, 8 slots) | `python main.py --example 4 [--eps 0.25] [--two-cpus] [--save]` |
| **5** | Duplicate of Ex.1 (8 jobs, same 4 types) | `python main.py --example 5 [--eps 0.25] [--two-cpus] [--save]` |
| **6** | Competitive (4 agents, 4 slots) | `python main.py --example 6 [--eps 0.25] [--save]` |
| **7** | 24h night-discount (24 slots, 20 jobs) | `python main.py --example 7 [--eps 0.25] [--two-cpus] [--save]` |

- `--eps` sets the bid increment ε (default `0.25`).
- `--two-cpus` doubles slots (e.g. 8→16, 24→48) for examples 1, 2, 4, 5, 7.
- `--save` writes allocation/price plots under `output/`.
- `--no-plots` suppresses the plot window (useful for batch runs).

### Epsilon sensitivity

- **All experiments:**  
  `python main.py --sensitivity [--save]`  
  Runs sensitivity over ε for every scenario (including 2-CPU variants) and saves plots to `output/` if `--save` is set.

- **Single experiment:**  
  `python main.py --sensitivity --example N [--two-cpus] [--save]`  
  e.g. `python main.py -s -e 1`, `python main.py -s -e 7 --two-cpus --save`.

### Run “all” and save everything

```bash
python main.py --all --save
```

Runs examples 1 (ε=0.25 and ε=1.0), 2, 3, full epsilon sensitivity for all experiments, and saves all plots to `output/`.

## Output files (with `--save`)

Plots are written under `output/` with names like:

- `ex1_allocation_eps0.25.png`, `ex1_allocation_eps0.25_2cpus.png`
- `ex2_allocation_eps0.25.png`, `ex5_duplicate_ex1_eps0.25_2cpus.png`
- `scalability_24h_allocation_eps0.25.png`, `scalability_24h_allocation_eps0.25_2cpus.png`
- `competitive_allocation_eps0.25.png`
- `epsilon_sensitivity_<experiment_name>.png`, `epsilon_sensitivity_all.png`

## Project layout

- `main.py` — CLI and experiment runners
- `src/auction/` — ascending auction implementation
- `src/models/` — market, agents, slots
- `src/experiments/` — scenarios (book examples, 24h, competitive, etc.) and metrics
- `src/visualization/` — allocation and price plots, epsilon-sensitivity figures
