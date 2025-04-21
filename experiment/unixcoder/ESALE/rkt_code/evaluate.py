#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score

# bert_score import and calculation might fail, handle gracefully

try:
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: 'bert-score' library not found. BERTScore calculation will be skipped.")
    print("Install it using: pip install bert-score torch (or tensorflow)")
    BERT_SCORE_AVAILABLE = False

# Try importing transformers for manual loading test
try:
    from transformers import AutoTokenizer, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: 'transformers' library not found. Cannot perform manual model loading test for BERTScore.")
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer, AutoModel = None, None  # Define as None

import statistics
import os
import traceback


# --- NLTK Download Check --- (Keep as is)
# ... (previous NLTK check code) ...

# --- File Reading Function --- (Keep modified version from previous response)
def read_files(ref_path, hyp_path, num):
    # ... (previous read_files code with ':' and '\t' splitting) ...
    # Ensure this function works correctly based on the previous iteration
    """读取参考文件和假设（结果）文件，并解析格式"""
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    if not os.path.exists(hyp_path):
        raise FileNotFoundError(f"Hypothesis file not found: {hyp_path}")

    references = []
    hypotheses = []
    malformed_refs = 0
    malformed_hyps = 0

    print(f"Reading references from: {ref_path}")
    with open(ref_path, 'r', encoding='utf-8') as f_ref:
        for i, line in enumerate(f_ref):
            line = line.strip()
            if not line: continue
            parts = line.split(':', 1)
            if len(parts) == 2:
                references.append(parts[1].strip())
            else:
                malformed_refs += 1

    print(f"Reading hypotheses from: {hyp_path}")
    with open(hyp_path, 'r', encoding='utf-8') as f_hyp:
        for i, line in enumerate(f_hyp):
            # line = line.strip()
            if not line: continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                hypotheses.append(parts[1].strip())
            else:
                hypotheses.append('')
                # malformed_hyps += 1

    if malformed_refs > 0:
        print(f"Warning: Found {malformed_refs} potentially malformed lines in reference file (missing ':').")
    if malformed_hyps > 0:
        print(f"Warning: Found {malformed_hyps} potentially malformed lines in hypothesis file (missing '\\t').")

    if num > 0:
        temp = []
        for i in range(num):
            temp.append(references[i])
        references = temp
    if len(references) != len(hypotheses):
        print(f"Error: Number of *successfully parsed* lines mismatch!")
        print(f"  Parsed {len(references)} references from {ref_path}.")
        print(f"  Parsed {len(hypotheses)} hypotheses from {hyp_path}.")
        exit(1)

    if not references:
        print("Warning: No valid content extracted from input files.")
        return [], []

    print(f"Successfully read {len(references)} reference-hypothesis pairs.")
    return references, hypotheses


# --- Evaluation Function ---
def evaluate_summaries(references, hypotheses, local_model_path):
    """计算各种评估指标"""
    bleu4_scores = []
    rougeL_f1_scores = []
    meteor_scores = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth_func = SmoothingFunction().method1

    print(f"\nEvaluating {len(references)} summaries...")

    # --- N-gram Metrics Calculation ---
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if not ref or not hyp: continue

        # BLEU-4
        try:
            ref_tokens_bleu = [word_tokenize(ref.lower())]
            hyp_tokens_bleu = word_tokenize(hyp.lower())
            bleu4 = sentence_bleu(ref_tokens_bleu, hyp_tokens_bleu, weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=smooth_func)
            bleu4_scores.append(bleu4)
        except Exception as e:
            print(f"Warning: BLEU calculation failed for pair {i}: {e}")
            bleu4_scores.append(0.0)

        # ROUGE-L
        try:
            rouge_scores = scorer.score(ref, hyp)
            rougeL_f1_scores.append(rouge_scores['rougeL'].fmeasure)
        except Exception as e:
            print(f"Warning: ROUGE calculation failed for pair {i}: {e}")
            rougeL_f1_scores.append(0.0)

        # METEOR
        try:
            ref_tokens_meteor = word_tokenize(ref.lower())
            hyp_tokens_meteor = word_tokenize(hyp.lower())
            meteor = single_meteor_score(ref_tokens_meteor, hyp_tokens_meteor)
            meteor_scores.append(meteor)
        except Exception as e:
            print(f"Warning: METEOR calculation failed for pair {i}: {e}")
            meteor_scores.append(0.0)

    # --- BERTScore Calculation ---
    bert_f1_scores = []
    bertscore_failed = True  # Default to failed

    if not BERT_SCORE_AVAILABLE:
        print("BERTScore calculation skipped: library not installed.")
    # ... (other checks for model_path, refs/hyps) ...
    elif local_model_path and references and hypotheses:
        print(f"\n--- Starting BERTScore Calculation ---")
        print(f"Using local model path: {local_model_path}")

        # 1. Pre-check path and config.json (Keep this check)
        config_path = os.path.join(local_model_path, 'config.json')
        if not os.path.isdir(local_model_path) or not os.path.exists(config_path):
            print(f"Error: Model directory or config.json check failed for: {local_model_path}")
        # 2. *** Manually Load Model and Tokenizer FOR BERTScore ***
        elif TRANSFORMERS_AVAILABLE:
            loaded_model = None
            loaded_tokenizer = None
            scorer_bert = None  # <--- Initialize scorer object
            try:
                print(f"DEBUG: Manually loading tokenizer and model for BERTScorer...")
                # --- Add device placement ---
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"DEBUG: Using device: {device}")
                # --- ---
                loaded_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                loaded_model = AutoModel.from_pretrained(local_model_path).to(device)  # <-- Move model to device
                print(
                    f"DEBUG: Manual load successful (Tokenizer: {type(loaded_tokenizer)}, Model: {loaded_model.config.model_type}).")

                # ---!!! IMPORTANT CORRECTION STARTS HERE !!!---

                # 4. *** Call the score method of the BERTScorer instance ***
                print(f"Calculating BERTScore using scorer instance...")
                # The scorer's score method takes only candidates and references
                P, R, F1 = score(hypotheses, references, lang="en", batch_size=16,
                                 baseline_path=local_model_path)  # Add batch_size
                # ---!!! IMPORTANT CORRECTION ENDS HERE !!!---

                bert_f1_scores = F1.tolist()
                bertscore_failed = False  # Mark as success!
                print("BERTScore calculation completed successfully.")

            except Exception as e:
                print(f"\n---!!! BERTScore Calculation Failed !!!---")
                if loaded_model is None or loaded_tokenizer is None:
                    print("Failure occurred during manual loading phase.")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {e}")
                print("Traceback:")
                traceback.print_exc()
                print("----------------------------------------")
                # bertscore_failed remains True
            finally:
                # Optional: Clean up memory if model objects are large
                del loaded_model
                del loaded_tokenizer

        else:  # Transformers not available
            print("BERTScore calculation skipped: transformers library needed for manual loading.")

    # If BERTScore failed, fill with zeros (Keep as is)
    if bertscore_failed and references:
        bert_f1_scores = [0.0] * len(references)

    # --- Calculate Averages --- (Keep as is)
    # ... (safe_mean helper and average calculations) ...
    def safe_mean(scores):
        valid_scores = [s for s in scores if isinstance(s, (int, float))]
        return statistics.mean(valid_scores) if valid_scores else 0.0

    avg_bleu4 = safe_mean(bleu4_scores)
    avg_rougeL = safe_mean(rougeL_f1_scores)
    avg_meteor = safe_mean(meteor_scores)
    avg_bert_f1 = safe_mean(bert_f1_scores)

    return {
        "BLEU-4": avg_bleu4,
        "ROUGE-L (F1)": avg_rougeL,
        "METEOR": avg_meteor,
        "BERTScore (F1)": avg_bert_f1,
        "Count": len(references),
        "BERTScore Failed": bertscore_failed
    }


# --- 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate code summaries.")
    parser.add_argument("-r", "--reference", type=str, default="../rkt_ref_40510.txt",
                        help="Path to the reference summaries file (format: index:content).")
    parser.add_argument("-p", "--prediction", type=str, default="../rkt_result/result_rkt_python_NL_4060ti_40510.txt",
                        help="Path to the predicted summaries file (format: index\\tcontent).")
    parser.add_argument("--model_path", type=str, default=" bert-base-uncased",  # 改为 None，明确要求用户提供
                        help="Path to the local directory containing the pre-trained model files for BERTScore (e.g., unixcoder-base). Required for BERTScore.")

    args = parser.parse_args()

    # Explicitly require model_path for BERTScore
    if not args.model_path and BERT_SCORE_AVAILABLE:
        print("Error: '--model_path' is required for BERTScore calculation.")
        exit(1)

    try:
        references, hypotheses = read_files(args.reference, args.prediction,-1)

        if references and hypotheses:
            results = evaluate_summaries(references, hypotheses, args.model_path)

            print("\n--- Evaluation Results ---")
            print(f"Successfully Evaluated Pairs: {results['Count']}")
            # --- Multiply by 100 for reporting ---
            print(f"BLEU-4:          {results['BLEU-4'] * 100:.4f}")  # .2f for 2 decimal places
            print(f"ROUGE-L (F1):    {results['ROUGE-L (F1)'] * 100:.4f}")
            print(f"METEOR:          {results['METEOR'] * 100:.4f}")
            if results['BERTScore Failed']:
                print(f"BERTScore (F1):  Failed / Skipped")
            else:
                print(f"BERTScore (F1):  {results['BERTScore (F1)'] * 100:.4f}")
            print("--------------------------")
        else:
            print("No valid summary pairs found to evaluate.")

    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
