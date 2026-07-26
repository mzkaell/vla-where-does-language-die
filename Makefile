# Where Does Language Die? -- targets from CLAUDE.md §10.
#
# Unbuilt targets FAIL LOUDLY (exit 1). A stub that silently succeeds is worse than
# no stub: it lets a phase look done when it is not. See CLAUDE.md §11 (no fabricated
# results) -- the same principle applies to the build surface.

PY      := .venv/Scripts/python.exe
UV      := uv
SUITE   ?= libero_goal
N       ?= 200
MODEL   ?= smolvla
PAIRS   ?= stimuli/libero_goal_pairs_v1.jsonl

ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python.exe
else
  PY := .venv/bin/python
endif

.PHONY: help setup setup-sim smoke pairs ifr localization transplant figures test lint typecheck clean

help:
	@echo "Targets:"
	@echo "  setup        Create .venv (py3.12) and install analysis+vla+interp+dev deps"
	@echo "  setup-sim    Create the SEPARATE LIBERO/MuJoCo env (Linux only)"
	@echo "  smoke        Load SmolVLA, run one headless LIBERO-Goal episode"
	@echo "  pairs        Build contrastive pairs        (M1)"
	@echo "  ifr          Instruction-following rate     (M0)"
	@echo "  localization Causal localization map        (M2) -- NOT BUILT"
	@echo "  transplant   Binding transplant             (M3) -- NOT BUILT"
	@echo "  figures      Publication figures            (M4) -- NOT BUILT"
	@echo "  test         pytest"

# ---------------------------------------------------------------- environment
setup:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e ".[vla,interp,dev]"
	@echo "OK: analysis env ready."

# The sim stack cannot coexist with the analysis stack (LIBERO pins
# transformers==4.21.1, numpy==1.22.4). It also cannot run on Windows at all:
# MuJoCo headless osmesa on Windows was closed upstream as "not planned".
setup-sim:
	@if [ "$(OS)" = "Windows_NT" ]; then \
		echo "ERROR: the LIBERO/MuJoCo sim env does not run on native Windows."; \
		echo "       robosuite is Linux/macOS only and MUJOCO_GL=osmesa is unsupported"; \
		echo "       on Windows (google-deepmind/mujoco#2164, closed as 'not planned')."; \
		echo "       Use WSL2 or a Linux host. See README.md 'Simulator environment'."; \
		exit 1; \
	fi
	$(UV) venv --python 3.11 .venv-sim
	$(UV) pip install --python .venv-sim/bin/python -e ".[sim]"
	@echo "OK: sim env ready. Remember: MUJOCO_GL=osmesa for headless."

# ---------------------------------------------------------------- pipeline
smoke:
	$(PY) scripts/smoke_test.py

pairs:
	$(PY) scripts/build_pairs.py --suite $(SUITE) --n $(N)

ifr:
	$(PY) scripts/reproduce_ifr.py --model $(MODEL) --suite $(SUITE)

localization:
	@echo "NOT BUILT: M2 localization (Phase 2). See CLAUDE.md §4."; exit 1

transplant:
	@echo "NOT BUILT: M3 binding transplant (Phase 3). See CLAUDE.md §4."; exit 1

figures:
	@echo "NOT BUILT: M4 figures (Phase 4). See CLAUDE.md §4."; exit 1

# ---------------------------------------------------------------- quality
test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

typecheck:
	$(PY) -m mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
