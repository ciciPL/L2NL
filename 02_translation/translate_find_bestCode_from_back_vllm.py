import gc
import json
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from vllm import LLM
import evaluate
import logging
import warnings
from transformers import logging as tr_logging

# ================= 🔇 静音配置 (只留必要的) =================
logging.getLogger("vllm").setLevel(logging.ERROR)
tr_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
# =========================================================

# ================= 配置 =================
MODEL_NAME = "./models/jina-code-embeddings-1.5b"
MAX_MODEL_LEN = 2048
WEIGHT_BLEUS = [0.5]
WEIGHT_JINAS = [0.5]



# =======================================

class JinaVLLMScorer:
    def __init__(self, model_name: str = None, max_model_len: int = None):
        model = model_name or MODEL_NAME
        mlen  = max_model_len or MAX_MODEL_LEN
        print(f"🚀 [Init] Loading Jina Model (vLLM): {model}")
        self.llm = LLM(
            model=model,
            task="embed",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=mlen,
            enforce_eager=False,
            disable_log_stats=True,
        )

    def get_all_embeddings(self, texts):
        """一次性获取所有文本的 Embeddings"""
        if not texts: return None

        # ⭐ 这里我们开启 use_tqdm=True，因为这是全量处理，需要看到 vLLM 的进度
        # vLLM 会自动处理内部的 batching
        outputs = self.llm.embed(texts, use_tqdm=True)

        # 提取 embedding 列表
        embeddings = [out.outputs.embedding for out in outputs]
        return embeddings


def process_file_all_in_one(scorer, bleu_metric, lang, input_file, output_file,WEIGHT_BLEU,WEIGHT_JINA):
    if not os.path.exists(input_file):
        return

    print(f"Processing [{lang}]: {os.path.basename(input_file)}")

    # 1. 读取数据
    data_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data_items.append(json.loads(line))

    # 2. 准备全量列表
    all_indices = []
    all_src_texts = []
    all_bt_texts = []

    for d_idx, item in enumerate(data_items):
        src = item.get('code', item.get('source_code', ''))
        for c_idx, cand in enumerate(item.get('translations', [])):
            bt = cand.get('back_translation', '')
            status = cand.get('status', '')
            # 筛选逻辑：只有 valid/success 的才计算分数
            if bt and status in ['valid', 'repaired_success', 'success']:
                all_indices.append((d_idx, c_idx))
                all_src_texts.append(src)
                all_bt_texts.append(bt)

    total_pairs = len(all_indices)
    if total_pairs == 0:
        print(f"   - No valid pairs found.")
        return

    print(f"   - Total pairs: {total_pairs}. Running vLLM inference...")

    # ================= 阶段一：vLLM 全量推理 (GPU) =================
    # use_tqdm=True 让 vLLM 显示它自己的处理进度
    print("   1. Embedding Source Codes...")
    emb_src_list = scorer.get_all_embeddings(all_src_texts)

    print("   2. Embedding Back-Translations...")
    emb_bt_list = scorer.get_all_embeddings(all_bt_texts)

    # ================= 阶段二：矩阵计算 (GPU) =================
    print("   3. Computing Cosine Similarity...")
    tensor_src = torch.tensor(emb_src_list, device='cuda')
    tensor_bt = torch.tensor(emb_bt_list, device='cuda')

    tensor_src = F.normalize(tensor_src, p=2, dim=1)
    tensor_bt = F.normalize(tensor_bt, p=2, dim=1)

    jina_scores_tensor = (tensor_src * tensor_bt).sum(dim=1)
    jina_scores_list = jina_scores_tensor.cpu().tolist()

    del tensor_src, tensor_bt, emb_src_list, emb_bt_list

    # ================= 阶段三：BLEU 计算与合并 (CPU) =================
    print("   4. Computing BLEU & Merging Data...")

    for i in tqdm(range(total_pairs), desc="   - Finalizing", unit="pair", ncols=80, leave=False):
        # 获取基础数据
        d_idx, c_idx = all_indices[i]
        s_jina = float(jina_scores_list[i])
        src = all_src_texts[i]
        bt = all_bt_texts[i]

        # 计算 BLEU
        s_bleu = 0.0
        try:
            # disable_tqdm=True 防止 evaluate 内部刷屏
            res = bleu_metric.compute(predictions=[bt], references=[[src]])
            s_bleu = res['bleu']
            if s_bleu > 1.0: s_bleu /= 100.0
        except:
            pass

        # 混合分数
        hybrid_score = (s_jina * WEIGHT_JINA) + (s_bleu * WEIGHT_BLEU)

        # ⭐关键：直接把分数写回原始 data_items 的 translations 列表中
        # 这样原始列表里的每一个候选都带有了分数信息
        data_items[d_idx]['translations'][c_idx]['scores'] = {
            'jina': s_jina,
            'bleu': s_bleu,
            'hybrid': hybrid_score
        }
        # 为了方便排序，也可以在平级加一个 key
        data_items[d_idx]['translations'][c_idx]['hybrid_score'] = hybrid_score

    # ================= 阶段四：保存 (丰富字段版) =================
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data_items:
            # 1. 寻找最佳候选
            candidates = item.get('translations', [])
            best_cand = {}
            best_idx = -1

            # 筛选器
            valid_cands_with_idx = []
            for idx, c in enumerate(candidates):
                if c.get('status') in ['valid', 'repaired_success', 'success']:
                    valid_cands_with_idx.append((idx, c))

            # 排序函数
            def get_score_tuple(x):
                # x is (index, candidate_dict)
                return x[1].get('hybrid_score', -1.0)

            if valid_cands_with_idx:
                valid_cands_with_idx.sort(key=get_score_tuple, reverse=True)
                best_idx, best_cand = valid_cands_with_idx[0]
            elif candidates:
                # 实在没有 valid 的，就按某种规则选一个，或者全空
                # 这里为了鲁棒性，选第一个
                best_idx = 0
                best_cand = candidates[0]

            # 2. ⭐ 构建输出对象 (保留全量信息)
            # 使用 copy() 确保 translations 列表被保留
            output_item = item.copy()

            # 3. ⭐ 添加“最佳候选”的快捷字段 (方便后续直接读取，不用再遍历列表)
            output_item['best_candidate_info'] = {
                'index': best_idx,  # 它是列表里的第几个
                'status': best_cand.get('status', 'failed'),
                'python_code': best_cand.get('code', ""),  # 最重要的：最佳 Python 代码
                'back_translation': best_cand.get('back_translation', ""),  # 最重要的：最佳回译（方便人工比对）
                'scores': best_cand.get('scores', {})  # 分数详情
            }

            # 为了兼容你可能的旧代码或方便 CSV 查看，也可以把关键字段“拍平”放在最外层
            output_item['best_python_code'] = best_cand.get('code', "")
            output_item['best_score_hybrid'] = best_cand.get('scores', {}).get('hybrid', 0.0)

            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    print(f"   ✅ Saved: {output_file}")

def _base_subdir(model_path: str) -> str:
    if 'Llama' in model_path: return 'Llama3th'
    if 'Seed'  in model_path: return 'Seed3th'
    if 'deepseek-coder-6.7b-instruct' in model_path: return 'ds_mid3th'
    if '1.3b'  in model_path: return 'ds_small3th'
    if 'Qwen'  in model_path: return 'qwen3th'
    return os.path.basename(model_path.rstrip('/'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Select best Python pivot candidate per (model, lang) using a weighted "
                    "BLEU + SBERT (Jina-Code) score against the original LRPL via back-translation."
    )
    parser.add_argument("--scorer_model", default=MODEL_NAME,
                        help=f"Sentence embedder for SBERT score (default: {MODEL_NAME}).")
    parser.add_argument("--scorer_max_len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--models", nargs="+",
                        default=[
                            "./models/deepseek-coder-1.3b-instruct",
                            "./models/deepseek-coder-6.7b-instruct",
                            "./models/Llama-3.1-8B-Instruct",
                            "./models/Seed-Coder-8B-Instruct",
                            "./models/Qwen2.5-Coder-14B-Instruct",
                        ],
                        help="Translator models whose candidates were back-translated by translate_back_2_lrpl.py.")
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'racket', 'r'],
                        help="Languages to process. Use ['ruby'] for CSN run.")
    parser.add_argument("--data_root", default="./data/LowData",
                        help="Root directory. Default ./data/LowData; use ./data/CSN for CSN run.")
    parser.add_argument("--input_template",
                        default="{data_root}/{lang}/trans_{base}/{lang}_python_multi_trans_vllm_back.jsonl")
    parser.add_argument("--output_template",
                        default="{data_root}/{lang}/trans_{base}/{lang}_python_best_candidate_vllm_B{wb}_S{ws}.jsonl")
    parser.add_argument("--bleu_weights", nargs="+", type=float, default=WEIGHT_BLEUS,
                        help="BLEU weights (paired with --jina_weights). Default: 0.5")
    parser.add_argument("--jina_weights", nargs="+", type=float, default=WEIGHT_JINAS,
                        help="Jina/SBERT weights (paired with --bleu_weights). Default: 0.5")
    args = parser.parse_args()

    if len(args.bleu_weights) != len(args.jina_weights):
        parser.error("--bleu_weights and --jina_weights must have the same length.")

    try:
        global_scorer = JinaVLLMScorer(model_name=args.scorer_model, max_model_len=args.scorer_max_len)
        global_bleu   = evaluate.load("bleu")
    except Exception as e:
        print(e)
        sys.exit(1)

    for wb, ws in zip(args.bleu_weights, args.jina_weights):
        for MODEL_PATH in args.models:
            base_path = _base_subdir(MODEL_PATH)
            print(f"\n{'=' * 10} {base_path}  (BLEU={wb}, SBERT={ws})  {'=' * 10}")
            for lang in args.langs:
                input_path  = args.input_template.format(data_root=args.data_root, lang=lang, base=base_path)
                output_path = args.output_template.format(data_root=args.data_root, lang=lang, base=base_path,
                                                          wb=wb, ws=ws)
                try:
                    process_file_all_in_one(global_scorer, global_bleu, lang, input_path, output_path, wb, ws)
                except Exception as e:
                    import traceback
                    traceback.print_exc()