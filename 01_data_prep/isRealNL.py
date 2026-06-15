"""
Hallucination filter for LLM-summarised candidates.

After filter_from_parquest.py produces step3_final_result/{lang}_final.tsv,
this script double-checks every kept summary by verifying it is *literally*
present in the original code content (normalised). Summaries that fail this
"real-anchor" check are treated as hallucinations and removed.

Paper §4.1: corresponds to the strict-deduplication / hallucination-removal
step on top of the spaCy + LanguageTool filtering.
"""

import pandas as pd
import os
import re

# ================= Configuration (overridden in __main__) =================
BASE_DIR = "./data/LowData"
LANGS = ['julia', 'lua', 'ocaml', 'r', 'racket']


def normalize(text):
    """
    Normalise to lowercase alphanumerics only.
    Strips comment markers (#, --, ;;), newlines, and whitespace so the
    check focuses on whether the core text exists verbatim in the code.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z0-9]', '', text.lower())


def isReal():
    """
    For every kept summary in step3_final_result/{lang}_final.tsv,
    verify it occurs (after normalisation) in the original parquet content;
    drop those that do not.
    """
    print("\n>>> Hallucination verification phase <<<")

    for lang in LANGS:
        print(f"\n================ {lang.upper()} ================")

        parquet_file = os.path.join(BASE_DIR, lang, f"{lang}-00000-of-00001.parquet")
        tsv_file = os.path.join(BASE_DIR, "step3_final_result", f"{lang}_final.tsv")

        if not os.path.exists(parquet_file):
            print(f"[skip] Missing parquet: {parquet_file}")
            continue
        if not os.path.exists(tsv_file):
            print(f"[skip] Missing TSV: {tsv_file}")
            continue

        try:
            df_origin = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"Failed to read parquet: {e}")
            continue

        try:
            df_extracted = pd.read_csv(tsv_file, sep='\t', quotechar='"', on_bad_lines='warn')
        except Exception as e:
            print(f"Failed to read TSV: {e}")
            continue

        valid_rows = []
        mismatches = []
        total_checked = 0

        print(f"Verifying {len(df_extracted)} candidates...")

        for _, row in df_extracted.iterrows():
            try:
                ts_idx = int(row['index'])
                summary = str(row['summary'])

                if ts_idx >= len(df_origin):
                    print(f"Error: Index {ts_idx} out of range")
                    continue

                original_content = df_origin.iloc[ts_idx]['content']

                # Core check: normalised summary must be a substring of normalised content
                if normalize(summary) in normalize(original_content):
                    valid_rows.append(row)
                else:
                    mismatches.append({
                        "index": ts_idx,
                        "extracted": summary,
                        "original_snippet": original_content[:500]
                    })

                total_checked += 1

            except Exception as row_e:
                print(f"Row processing error: {row_e}")
                continue

        match_rate = (len(valid_rows) / total_checked * 100) if total_checked > 0 else 0
        print(f"📊 Stats:")
        print(f"   total: {total_checked}")
        print(f"   ✅ verified (kept): {len(valid_rows)}")
        print(f"   ❌ hallucination (dropped): {len(mismatches)}")
        print(f"   pass rate: {match_rate:.2f}%")

        # Overwrite the TSV with only verified rows
        if valid_rows:
            df_final = pd.DataFrame(valid_rows)
            df_final['index'] = df_final['index'].astype(int)
            df_final = df_final.sort_values(by='index')
            df_final[['index', 'summary']].to_csv(
                tsv_file,
                sep='\t',
                index=False,
                encoding='utf-8-sig',
                escapechar='\\'
            )
            print(f"💾 Updated (verified only): {tsv_file}")
        else:
            print(f"⚠️  No samples passed verification; file unchanged.")

        if mismatches:
            print("\n⚠️  [Removed hallucinations - first 3]")
            for m in mismatches[:3]:
                print(f"--- Index: {m['index']} ---")
                print(f"  [LLM extracted]: {m['extracted']}")
                print(f"  [original snippet]: {m['original_snippet'].replace(chr(10), ' ')}...")
                print("")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Hallucination filter for LLM-summarised candidates: "
                    "verify every kept summary has a real anchor in the original code."
    )
    parser.add_argument("--base_dir", default="./data/LowData",
                        help="Working dir containing `<lang>/<lang>-00000-of-00001.parquet` "
                             "and `step3_final_result/<lang>_final.tsv` produced by filter_from_parquest.py "
                             "(default: ./data/LowData)")
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'r', 'racket'])
    args = parser.parse_args()
    BASE_DIR = args.base_dir
    LANGS = args.langs

    isReal()
