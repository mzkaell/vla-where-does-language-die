# Where Does Language Die? -- targets from CLAUDE.md §10.
#
# Unbuilt targets FAIL LOUDLY (exit 1). A stub that silently succeeds is worse than
# no stub: it lets a phase look done when it is not. See CLAUDE.md §11 (no fabricated
# results) -- the same principle applies to the build surface.

UV      := uv
SUITE   ?= libero_goal
N       ?= 200
MODEL   ?= smolvla
CKPT    ?=
PAIRS   ?= stimuli/libero_goal_pairs_v1.jsonl

# Windows venvs put the interpreter in Scripts/, POSIX in bin/.
ifeq ($(OS),Windows_NT)
  PY     := .venv/Scripts/python.exe
  PY_SIM := .venv-sim/Scripts/python.exe
else
  PY     := .venv/bin/python
  PY_SIM := .venv-sim/bin/python
endif

.PHONY: help setup setup-sim data data-check smoke pairs ifr localization transplant \
        figures test test-fast lint typecheck clean

help:
	@echo "Getting started (new clone):"
	@echo "  make setup        Create .venv (py3.12) and install deps"
	@echo "  make data         Download LIBERO-Goal demos (3.3 GB)"
	@echo "  make smoke        Verify the model loads and predicts"
	@echo "  make test         Run the test suite"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pairs        Contrastive stimulus set        (M1)"
	@echo "  make ifr CKPT=... Instruction-following rate      (M0)"
	@echo "  make localization Causal localization map         (M2) -- NOT BUILT"
	@echo "  make transplant   Binding transplant              (M3) -- NOT BUILT"
	@echo "  make figures      Publication figures             (M4) -- NOT BUILT"
	@echo ""
	@echo "  make setup-sim    LIBERO/MuJoCo env (Linux only, separate venv)"

# ---------------------------------------------------------------- environment
setup:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e ".[vla,interp,dev]"
	@echo "OK: analysis env ready. Next: make data"

# The sim stack cannot coexist with the analysis stack (LIBERO pins
# transformers==4.21.1, numpy==1.22.4). It also cannot run on Windows at all:
# MuJoCo headless osmesa on Windows was closed upstream as "not planned".
setup-sim:
ifeq ($(OS),Windows_NT)
	@echo "ERROR: the LIBERO/MuJoCo sim env does not run on native Windows."
	@echo "       robosuite is Linux/macOS only and MUJOCO_GL=osmesa is unsupported"
	@echo "       on Windows (google-deepmind/mujoco#2164, closed as 'not planned')."
	@echo "       Use WSL2 or a Linux host. See README.md."
	@exit 1
else
	$(UV) venv --python 3.11 .venv-sim
	$(UV) pip install --python $(PY_SIM) -e ".[sim]"
	@echo "OK: sim env ready. Remember: export MUJOCO_GL=osmesa"
endif

# ---------------------------------------------------------------- data
data:
	$(PY) scripts/download_data.py --suite $(SUITE)

data-check:
	$(PY) scripts/download_data.py --suite $(SUITE) --check

# ---------------------------------------------------------------- pipeline
smoke:
	$(PY) scripts/smoke_test.py

pairs:
	$(PY) scripts/build_pairs.py --suite $(SUITE) --n $(N)

# CKPT is required: smolvla_base is not a LIBERO policy and an IFR from it is
# meaningless. The script refuses without it; this makes the reason visible earlier.
ifr:
ifeq ($(strip $(CKPT)),)
	@echo "ERROR: set CKPT to a LIBERO-finetuned SmolVLA, e.g."
	@echo "   make ifr CKPT=k1000dai/smolvla_libero_finetune"
	@echo "lerobot/smolvla_base declares 6-dim state/action vs LIBERO's 8/7 and has"
	@echo "never seen these tasks, so its IFR would measure an incompetent policy."
	@exit 1
else
	$(PY) scripts/reproduce_ifr.py --model $(MODEL) --suite $(SUITE) --checkpoint $(CKPT)
endif

localization:
	@echo "NOT BUILT: M2 localization (Phase 2). See CLAUDE.md §4."; exit 1

transplant:
	@echo "NOT BUILT: M3 binding transplant (Phase 3). See CLAUDE.md §4."; exit 1

figures:
	@echo "NOT BUILT: M4 figures (Phase 4). See CLAUDE.md §4."; exit 1

# ---------------------------------------------------------------- quality
test:
	$(PY) -m pytest

# Skips anything needing real weights or the LIBERO files -- useful in CI or on a
# fresh clone before `make data`.
test-fast:
	$(PY) -m pytest -m "not slow"

lint:
	$(PY) -m ruff check src tests scripts

typecheck:
	$(PY) -m mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
