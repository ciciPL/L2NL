import nltk
import sacrebleu
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from . import MyscoreBert

# 下载 NLTK 数据（仅第一次需要）
nltk.download('punkt')
nltk.download('wordnet')

def read_file(path):
    """读取 index:内容 格式的文件，返回按 index 排序后的句子列表"""
    lines = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                idx_str, text = line.strip().split(":", 1)
                idx = int(idx_str)
                lines[idx] = text.strip()
            except ValueError as e:
                print(f"Skipping invalid line: {line.strip()} | Error: {e}")
    return [lines[i] for i in sorted(lines.keys())]

def compute_bleu(refs, hyps):
    scores = []
    for ref, hyp in zip(refs, hyps):
        if not ref or not hyp:
            scores.append(0.0)
            continue
        try:
            score = sacrebleu.corpus_bleu([hyp], [[ref]]).score
            scores.append(score)
        except Exception as e:
            print(f"BLEU error: {e}")
            scores.append(0.0)
    return sum(scores) / len(scores), scores

def compute_rouge(refs, hyps):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for ref, hyp in zip(refs, hyps):
        if not ref or not hyp:
            scores.append(0.0)
            continue
        try:
            score = scorer.score(hyp, ref)
            scores.append(score['rougeL'].fmeasure)
        except Exception as e:
            print(f"ROUGE error: {e}")
            scores.append(0.0)
    return sum(scores) / len(scores), [round(s * 100, 4) for s in scores]

def compute_meteor(refs, hyps):
    scores = []
    for ref, hyp in zip(refs, hyps):
        if not ref or not hyp:
            scores.append(0.0)
            continue
        try:
            ref_toks = word_tokenize(ref.lower())
            hyp_toks = word_tokenize(hyp.lower())
            score = meteor_score([ref_toks], hyp_toks)
            scores.append(score)
        except Exception as e:
            print(f"METEOR error: {e}")
            scores.append(0.0)
    return sum(scores) / len(scores), [round(s * 100, 4) for s in scores]

def compute_bertscore(refs, hyps):
    P, R, F1 = MyscoreBert.score(
        hyps,
        refs,
        model_type='model/microsoft/deberta-xlarge-mnli',  # 不使用预设模型名
        lang="en",
    )
    scores = F1.tolist()
    return float(sum(scores) / len(scores)) * 100, [round(f * 100, 4) for f in scores]

def main():
    #python -m evaluate.aliEval

    HYP_PATH = "experiment/ds-coder-1_3B/rkt_result/rkt_python_NL_4051_2th_c_sft_b.txt"
    REF_PATH = "experiment/ds-coder-1_3B/rkt_result/rkt_ref_4051.txt"

    references = read_file(REF_PATH)
    hypotheses = read_file(HYP_PATH)

    assert len(references) == len(hypotheses), "Reference and hypothesis files must have the same number of lines."

    print("🔍 正在评估模型输出...\n")

    avg_bleu, bleu_list = compute_bleu(references, hypotheses)
    avg_rouge, rouge_list = compute_rouge(references, hypotheses)
    avg_meteor, meteor_list = compute_meteor(references, hypotheses)
    avg_bertscore, bertscore_list = compute_bertscore(references, hypotheses)

    # 可选：打印每句结果
    # for i, (b, r, m, bs) in enumerate(zip(bleu_list, rouge_list, meteor_list, bertscore_list)):
    #     print(f"Sentence {i+1}: BLEU={b:.2f}, ROUGE-L={r:.2f}, METEOR={m:.2f}, BERTScore={bs:.2f}")

    print("\n📊 === 平均评估结果 ===")
    print(f"BLEU:        {avg_bleu:.2f}")
    print(f"ROUGE-L:     {avg_rouge:.2f}")
    print(f"METEOR:      {avg_meteor:.2f}")
    print(f"BERTScore:   {avg_bertscore:.2f}")

if __name__ == "__main__":
    main()