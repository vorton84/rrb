# Half the Rollouts, No Measurable Cost

Code for the ICLR 2027 submission. Built on verl 0.5.0.dev (Apache 2.0, `LICENSE`).

## Layout

```
verl/                framework; one change vs upstream: three fields in verl/workers/config/actor.py used by recipe/fipok
recipe/dapo/         DAPO baseline (seeds 1-3), Dr.GRPO
recipe/fipo/         FIPO (loss_mode=future_kl, implemented in verl/trainer/ppo/core_algos.py)
recipe/fipok/        FIPO-K
recipe/pso/          PSO v1, v2, v5 (A=1.2/1.5/2.0), v6; mean@128 evaluation script
recipe/csr/          CSR
recipe/pso_csr/      PSO+CSR v2/v3/v4
recipe/ktsc/         KL-targeted step control, applied to DAPO and to PSO v5
recipe/irt/          prompt selection with the mixed-group observer; random-selection (eps=1.0) runs at gen 256/128/96 and for the two additional models
data/                builders for the AIME 2025 and AIME 2026 evaluation parquets
eval/                mean@128 summary over evaluation logs
figures/             figure generation for the paper
```

## Setup

```
pip install -e .
mkdir -p models datasets ckpts
```

Place `Qwen2.5-Math-7B`, `Qwen2.5-Math-1.5B`, `Qwen2.5-7B` under `models/` (set `max_position_embeddings` to at least 32768 in each `config.json`).
Place `dapo-math-17k.parquet` and `aime-2024.parquet` from the DAPO release under `datasets/`. Build the other two:

```
python3 data/make_aime2025_parquet.py    # expects datasets/opencompass_AIME2025/aime2025-{I,II}.jsonl
python3 data/make_aime2026_parquet.py    # expects datasets/aime_2026/data/train-00000-of-00001.parquet (MathArena/aime_2026)
```

`export WANDB_MODE=offline` before training. All paths default to this directory; override with `SHARED_BASE`, `MODEL_PATH`, `TRAIN_FILE`, `TEST_FILE`, `CKPTS_DIR`.

## Runs reported in the paper

| paper label | script |
|---|---|
| DAPO seed 1 / 2 / 3 | `recipe/dapo/run_dapo_qwen2.5_7b_math.sh`, `run_dapo_seed2_*.sh`, `run_dapo_seed3_*.sh` |
| Dr.GRPO | `recipe/dapo/run_drgrpo_dapo_qwen2.5_7b_math.sh` |
| FIPO / FIPO-K | `recipe/fipo/run_fipo_qwen2.5_7b_math.sh`, `recipe/fipok/run_fipok_qwen2.5_7b_math.sh` |
| PSO v1, v2, v5 A=1.2/1.5/2.0, v6 | `recipe/pso/run_pso_400steps_*.sh`, `run_pso_v2_*.sh`, `run_pso_v5_rank_amp12_*.sh`, `run_pso_v5_rank_amp15_*.sh`, `run_pso_v5_rank_*.sh`, `run_pso_v6_signed_amp15_*.sh` |
| CSR, PSO+CSR v2/v3/v4 | `recipe/csr/run_csr_*.sh`, `recipe/pso_csr/run_pso_csr_*.sh` |
| DAPO+KTSC, PSO A=1.5/2.0 + KTSC | `recipe/ktsc/run_ktsc_dapo_*.sh`, `run_ktsc2_pso_v5_rank_amp15_*.sh`, `run_ktsc2_pso_v5_rank_amp20_*.sh` |
| tracker selection gen 256 / 128 | `recipe/irt/run_irt_dapo_qwen2.5_7b_math.sh`, `run_irt_dapo_gen128_qwen2.5_7b_math.sh` |
| random selection gen 128, seeds 1-3 | `recipe/irt/run_irt_dapo_gen128_eps1_*.sh` |
| random selection gen 96 | `recipe/irt/run_irt_dapo_gen96_eps1_qwen2.5_7b_math.sh` |
| Qwen2.5-Math-1.5B, Qwen2.5-7B | `recipe/irt/run_irt_dapo_gen256_eps1_qwen2.5_1.5b_math.sh`, `run_irt_dapo_gen256_eps1_qwen2.5_7b_general.sh` |

Runs are capped at 400 steps and stop when the data loader is exhausted. Seed 1 of DAPO was launched with `total_training_steps=200` in its script and continued to 400 by relaunching with `resume_mode=auto`; seeds 2 and 3 set 400 directly.

## Evaluation

Run from the repository root. Checkpoints are read from `ckpts/<project>/<exp_name>/global_step_<N>`.

```
R=$PWD/ckpts/DAPO
CKPT_ROOT=$R EXP_NAME=DAPO-Qwen2.5-Math-7B EVAL_STEP=320 VAL_N=4 bash recipe/pso/run_eval_only_qwen2.5_7b_math.sh > eval_DAPO-Qwen2.5-Math-7B_step320_n4.log
CKPT_ROOT=$R EXP_NAME=DAPO-Qwen2.5-Math-7B EVAL_STEP=320 VAL_N=4 TEST_FILE=$PWD/datasets/aime-2025.parquet bash recipe/pso/run_eval_only_qwen2.5_7b_math.sh > eval25_DAPO-Qwen2.5-Math-7B_step320_n4.log
CKPT_ROOT=$R EXP_NAME=DAPO-Qwen2.5-Math-7B EVAL_STEP=320 VAL_N=4 TEST_FILE=$PWD/datasets/aime-2026.parquet bash recipe/pso/run_eval_only_qwen2.5_7b_math.sh > eval26_DAPO-Qwen2.5-Math-7B_step320_n4.log
python3 eval/summarize_eval128.py
```

`VAL_N=4` gives 128 samples per problem. Checkpoints land under `ckpts/<project_name>/<exp_name>`, where `project_name` is set in each run script (`DAPO`, `PSO`, `FIPO`, `CSR`, `PSO_CSR`). The summary reads every `eval*_n4.log` in the current directory (or `EVAL_LOG_DIR`) and keys on the filename prefix (`eval_`, `eval25_`, `eval26_`).

## Tests

```
python3 -m recipe.ktsc.test_ktsc
python3 -m recipe.irt.test_irt
python3 -m recipe.fipok.test_fipok_scan
```
