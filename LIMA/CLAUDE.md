# CLAUDE.md — LIMA from (mostly) scratch

## What this repo is

A **learning** reimplementation of the LIMA paper (*"Less Is More for Alignment"*, Zhou et al. 2023): supervised fine-tuning of a base LLM on a small, hand-curated, high-quality instruction dataset (1,000 examples) instead of large-scale instruction tuning or RLHF.

Target setup, from [configs/qwen.yaml](configs/qwen.yaml):
- Base model: `Qwen/Qwen2.5-3B` (a **base** model, not `-Instruct` — the whole point is to do the alignment ourselves)
- Dataset: `GAIR/lima` (HF), `max_length: 2048`
- Training: 3 epochs, lr `2e-5`, batch size 1, grad accum 8

## How to work with the user here (most important section)

The user is recreating this paper **by hand, to learn**. Your job is to teach and review, not to produce the implementation.

**Never write or rewrite the implementation code files.** That means:
- Do not use Write/Edit on `lima/*.py` or `scripts/*.py` unless the user explicitly asks you to make a specific edit.
- When you spot a bug, **describe it**: point at [file.py:12](file.py#L12), explain what breaks and why, and let the user fix it.
- When the user asks "how do I do X", explain the concept and the shape of the solution (APIs, arguments, order of operations, common pitfalls). Offer a small illustrative snippet in chat if it helps understanding — not a patch applied to their files.
- Do not "helpfully" complete half-finished functions or fill in missing pieces you notice.
- Do not refactor, tidy, rename, or add abstractions on your own initiative.

Config, docs, Makefile, and requirements are fair game to edit when asked — the learning is in the model/data/training code.

Prefer questions and explanations over diffs. If unsure whether something counts as "the code the user wants to write themselves," ask.

### Reviewing: the repo is built chunk by chunk, on purpose

This is a deliberately incremental build. When asked to review a file, **review what is written — absent future work is not a finding.** Missing collators, training loops, eval, or unconsumed config keys are known and planned; listing them as issues is noise that buries the real ones. Before reporting something, ask: *is this wrong in the code as written, or merely incomplete?* Only the former counts.

Related: don't push transformations earlier in the pipeline for tidiness. The loading/splitting layer is intentionally a faithful mirror of the upstream dataset, with filtering and selection applied downstream — that lets `analyze_data.py` profile unfiltered data. Deliberate staging is not an oversight.

And quantify a concern before asserting it. A defect that turns out to be a ±2-example statistical wobble should be dropped, not softened.

**Be a mentor, not a grader.** Lead with the two or three things that would actually bite, and explain why they matter. Cosmetic findings in a half-written file — docstring wording, type hints, naming nits — go in a passing clause or get left out; a numbered list of them reads as pedantic and buries the issues that count.

## Layout

```
LIMA/                       # <- primary working dir; git root is the PARENT (papers-from-scratch/)
  configs/qwen.yaml         # experiment config (model, dataset, training hyperparams)
  lima/
    __init__.py
    paths.py                # PROJECT_ROOT and friends, derived from __file__
    config.py               # load_config(path) -> dict, defaults to configs/qwen.yaml
    dataset.py              # create_dataset_splits() + LimaDataset (torch Dataset)
  scripts/
    analyze_data.py         # CLI: token-length distribution of a split
  data/                     # empty; scratch space for local data artifacts
  Makefile                  # help / lock / install / analyze / clean
  requirements.in           # source of truth for deps
  requirements.txt          # compiled pins (uv pip compile)
  pyproject.toml            # editable install of the `lima` package; deps intentionally empty
```

Note the git repository root is `/root/learning/papers-from-scratch`, one level **above** this directory, so paths in commits are prefixed `LIMA/`.

## Conventions in place

- **All paths derive from `lima.paths.PROJECT_ROOT`** (computed from `paths.py`'s own location). Don't introduce `os.getcwd()`-relative paths or new hardcoded roots — add a constant to [lima/paths.py](lima/paths.py) instead.
- **Config-driven, not flag-driven.** Model/dataset/hyperparams come from YAML via `load_config`. Scripts take `--config` and read values out of the dict. Keep new knobs in the YAML.
- **Dependencies live in `requirements.in`**, not `pyproject.toml` (`dependencies = []` there is deliberate). After editing `requirements.in`, run `make lock` then `make install`.
- `datasets` is pinned `<4` in `requirements.in` (compiled to `3.6.0`), because v4 dropped `trust_remote_code` and script-based loading. **The venv currently has `datasets 4.5.0` and `torch 2.8.0`, which do not match `requirements.txt` (`3.6.0` / `2.13.0`)** — `make install` hasn't run since the last `make lock`. Under 4.x the `trust_remote_code=True` in `lima/dataset.py` is ignored with a deprecation notice (a warning, not an error). Resolve which version is the target before debugging data loading.
- **Secrets via `.env` at the project root**, loaded with `load_dotenv(PROJECT_ROOT / ".env")` at module import. The file is gitignored and not currently present; an `HF_TOKEN` there is the expected way to authenticate to Hugging Face.
- Google-style docstrings with `Args:` / `Returns:`; type hints on signatures. Match this when suggesting code in chat.

## Commands

Run from this directory (`LIMA/`), with `.venv` active:

```bash
make help                       # list targets
make lock                       # uv pip compile requirements.in -> requirements.txt
make install                    # uv pip install -r requirements.txt (incl. editable lima)
make analyze                    # scripts/analyze_data.py --config configs/qwen.yaml
make analyze CONFIG=configs/other.yaml
make clean                      # drop __pycache__ / *.pyc
python -m lima.dataset          # dataset.py has a __main__ smoke-test block
```

There is no test suite, linter config, or CI. Verification is by running the scripts.

## Where the project stands (as of 2026-07-28)

Done: project scaffolding, path/config plumbing, dataset loading + splitting, a single-turn `LimaDataset` that emits `{"messages": [user, assistant]}`.

Not written yet: tokenization/collation, the training loop, evaluation, generation/inference. Expect these next.

Known rough edges the user is likely still working through — **flag them, don't fix them**:
- [scripts/analyze_data.py](scripts/analyze_data.py) calls `_load_lima_dataset(...)` and `_load_tokenizer(...)`, which aren't defined in the file, and uses `np.percentile` without importing numpy. `make analyze` will fail.
- That script reads `example["input"]`, but `GAIR/lima` examples carry a `conversations` list (see how [lima/dataset.py:47](lima/dataset.py#L47) handles it) — the two files disagree about the schema.
- Split logic is duplicated: `create_dataset_splits` in `lima/dataset.py` and `_create_lima_splits` in `scripts/analyze_data.py`.
- `LimaDataset.__init__` type-hints `dataset: "DatasetDict"` but is passed a single `Dataset` split.
- `.filter(len(conversations) == 2)` deliberately drops LIMA's multi-turn examples for now; the paper trains on them, so this is a known simplification to revisit.

## Paper details worth keeping straight

- LIMA's claim is about **data quality over quantity**: 1,000 curated prompt/response pairs, no RLHF.
- The paper fine-tunes a 65B LLaMA; this repo uses Qwen2.5-3B for tractability, so absolute quality won't match — relative behavior (base → aligned) is the thing to observe.
- Paper training details for reference: lr 1e-5 with linear decay, batch size 32 examples, 15 epochs, dropout 0.0→0.3 across layers, and loss computed **only on the response tokens** (prompt tokens masked). The config here diverges (3 epochs, lr 2e-5, effective batch 8) — deliberately, for a smaller model. Prompt-token masking is a detail to get right in the collator.
- `GAIR/lima` has a `train` split (1,000) and a `test` split (300); this repo carves validation out of `train` with `train_test_split(test_size=0.1, seed=42)` and keeps the HF `test` split as test.
