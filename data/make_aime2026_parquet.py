

import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PARQUET = os.path.join(BASE, "datasets", "aime-2024.parquet")
IN_PARQUET = os.path.join(BASE, "datasets", "aime_2026", "data", "train-00000-of-00001.parquet")
OUT_PARQUET = os.path.join(BASE, "datasets", "aime-2026.parquet")

EXPECTED = {
    1: 277, 2: 62, 3: 79, 4: 70, 5: 65, 6: 441, 7: 396, 8: 244,
    9: 29, 10: 156, 11: 896, 12: 161, 13: 39, 14: 681, 15: 83,
    16: 178, 17: 243, 18: 503, 19: 279, 20: 190, 21: 50, 22: 754,
    23: 245, 24: 669, 25: 850, 26: 132, 27: 223, 28: 107, 29: 157, 30: 393,
}

def normalise_answer(raw):
    s = str(raw).strip()
    original = s
    for token in ("^{\\circ}", "^\\circ", "\\circ", "^\\text{o}", "\\degree", "°"):
        s = s.replace(token, "")
    s = s.strip().strip("$").strip()
    if s != original:
        print(f"  normalised answer: {original!r} -> {s!r}")
    return s

def load_2026():
    src = pd.read_parquet(IN_PARQUET)
    print(f"source: {IN_PARQUET}  rows={len(src)}  cols={list(src.columns)}")
    for col in ("problem_idx", "answer", "problem"):
        assert col in src.columns, f"source is missing column {col!r}"
    assert len(src) == 30, f"expected 30 AIME 2026 problems, got {len(src)}"

    src = src.sort_values("problem_idx").reset_index(drop=True)
    idxs = [int(i) for i in src["problem_idx"]]
    assert idxs == list(range(1, 31)), f"problem_idx is not 1..30: {idxs}"

    rows = []
    for _, r in src.iterrows():
        idx = int(r["problem_idx"])
        question = str(r["problem"]).strip()
        answer = normalise_answer(r["answer"])
        assert question, f"problem {idx} has empty text"
        assert answer.isdigit() and 0 <= int(answer) <= 999, (
            f"AIME answers are integers 0-999, problem {idx} gave {answer!r}"
        )
        assert int(answer) == EXPECTED[idx], (
            f"ANSWER MISMATCH at problem {idx}: file says {answer}, "
            f"verified key says {EXPECTED[idx]}. Upstream data changed -- "
            f"re-verify against two independent sources before proceeding."
        )
        rows.append((question, answer))
    print(f"all {len(rows)} answers match the independently verified key")
    return rows

def main():
    if os.path.exists(OUT_PARQUET):
        print(f"REFUSING: {OUT_PARQUET} already exists. Delete it yourself if you mean to rebuild.")
        return 1

    df = pd.read_parquet(SRC_PARQUET)
    print(f"template: {SRC_PARQUET}  rows={len(df)}  cols={list(df.columns)}")

    n_unique = len({r["raw_problem"] for r in df["extra_info"]})
    n_repeat, rem = divmod(len(df), n_unique)
    assert rem == 0, f"{len(df)} rows do not divide evenly into {n_unique} problems"
    print(f"template holds {n_unique} problems x {n_repeat} repeats")

    row0 = df.iloc[0]
    content0 = row0["prompt"][0]["content"]
    raw0 = row0["extra_info"]["raw_problem"]
    assert raw0 in content0, "raw_problem is not a substring of the prompt content"
    prefix, suffix = content0.split(raw0, 1)
    print(f"prompt prefix  ({len(prefix)} chars): {prefix[:70]!r} ...")
    print(f"prompt suffix  ({len(suffix)} chars): ... {suffix[-70:]!r}")

    role0 = row0["prompt"][0]["role"]
    data_source0 = row0["data_source"]
    ability0 = row0["ability"]
    style0 = row0["reward_model"]["style"]
    extra_keys = dict(row0["extra_info"])
    assert len(row0["prompt"]) == 1, "prompt is expected to be a single user turn"
    print(f"data_source={data_source0!r} ability={ability0!r} style={style0!r}")
    print(f"extra_info keys={list(extra_keys)}")

    for probe in (1, len(df) // 2, len(df) - 1):
        r = df.iloc[probe]
        assert r["prompt"][0]["content"] == prefix + r["extra_info"]["raw_problem"] + suffix, (
            f"row {probe} does not follow the recovered template"
        )
    print("template verified on 3 sampled rows")

    problems = load_2026()
    assert len({q for q, _ in problems}) == 30, "duplicate problem statements in the source"

    records = []
    for idx, (question, answer) in enumerate(problems):
        extra = dict(extra_keys)
        if "index" in extra:
            extra["index"] = idx
        if "raw_problem" in extra:
            extra["raw_problem"] = question
        for _ in range(n_repeat):
            records.append(
                {
                    "data_source": data_source0,
                    "prompt": [{"role": role0, "content": prefix + question + suffix}],
                    "ability": ability0,
                    "reward_model": {"style": style0, "ground_truth": answer},
                    "extra_info": dict(extra),
                }
            )

    out = pd.DataFrame(records)
    out = out[[c for c in df.columns if c in out.columns]]
    assert len(out) == 30 * n_repeat, f"row count {len(out)} != {30 * n_repeat}"
    assert list(out.columns) == [c for c in df.columns if c in out.columns]

    out.to_parquet(OUT_PARQUET)
    print(f"\nwrote {OUT_PARQUET}  rows={len(out)}  (30 problems x {n_repeat})")

    back = pd.read_parquet(OUT_PARQUET)
    assert len(back) == len(out)
    assert list(back.columns) == list(out.columns)
    b0 = back.iloc[0]
    assert b0["data_source"] == data_source0
    assert b0["prompt"][0]["role"] == role0
    assert b0["reward_model"]["style"] == style0
    truths = [r["ground_truth"] for r in back["reward_model"]][::n_repeat]
    assert [int(t) for t in truths] == [EXPECTED[i] for i in range(1, 31)], (
        "written ground truths do not match the verified key in order"
    )
    print("read-back OK; ground truths match the verified key in order")
    print("\nfirst prompt:\n" + back.iloc[0]["prompt"][0]["content"])
    print("\nground truths:", truths)
    return 0

if __name__ == "__main__":
    sys.exit(main())
