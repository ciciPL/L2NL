import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import numpy as np

import os
import argparse
import logging
import random
from tqdm import tqdm
import json
import jsonlines
import platform
import sys

# 确保 model.py 存在或提供一个占位符
try:
    from model import SelectorNet
except ImportError:
    print("警告: 无法导入 'model.py'。请确保该文件存在。")


from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

# 根据您的环境调整路径
transformer_path = '../../../model/microsoft/codeBERT-base'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

sysstr = platform.system()


def set_seed(args):
    """设置随机种子以确保结果可复现。"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Example(object):
    """数据样本的表示。"""

    def __init__(self, idx, source, labels):
        self.idx = idx
        self.source = source
        self.labels = labels


class InputFeatures(object):
    """转换为模型输入的特征表示。"""

    def __init__(self, example_id, source_ids, word_masks, stat_masks, labels):
        self.word_masks = word_masks
        self.example_id = example_id
        self.source_ids = source_ids
        self.stat_masks = stat_masks
        self.labels = labels


def read_examples(filename, code_type):
    """
    从 .jsonl 文件读取数据样本。
    修改：同时读取外层代码和 retrieved_candidates 中的代码。
    """
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), desc=f"读取 {os.path.basename(filename)}"):
            line = line.strip()
            try:
                js = json.loads(line)
            except json.JSONDecodeError:
                continue

            if 'idx' not in js:
                js['idx'] = idx

            # --- 1. 处理外层 Main Code ---
            code_main = [i.replace('\n', ' ') for i in js.get(code_type, [])]
            labels_main = js.get('ex_labels', [])

            examples.append(
                Example(
                    idx=f"{idx}_main",
                    source=code_main,
                    labels=labels_main,
                )
            )

            # --- 2. 处理 Retrieved Candidates ---
            candidates = js.get("retrieved_candidates", [])
            for cand_i, cand in enumerate(candidates):
                code_cand = [i.replace('\n', ' ') for i in cand.get(code_type, [])]
                # retrieved code 通常没有 label，给空列表
                labels_cand = cand.get('ex_labels', [])

                examples.append(
                    Example(
                        idx=f"{idx}_cand_{cand_i}",
                        source=code_cand,
                        labels=labels_cand,
                    )
                )

    return examples


def convert_examples_to_features(examples, tokenizer, args, word_length=None, stat_length=None):
    """
    将数据样本转换为模型输入特征。
    (逻辑保持不变，因为 Example 列表已经被打平了)
    """
    features = []
    for example_index, example in tqdm(enumerate(examples), desc="转换特征"):
        source_ids = []
        word_masks = []

        # source
        for i in example.source[:stat_length]:
            stat_tokens = tokenizer.tokenize(i)[:word_length]
            stat_ids = tokenizer.convert_tokens_to_ids(stat_tokens)
            stat_mask = [1] * (len(stat_tokens))
            padding_length = word_length - len(stat_ids)
            stat_ids += [tokenizer.pad_token_id] * padding_length
            stat_mask += [0] * padding_length

            source_ids.append(stat_ids)
            word_masks.append(stat_mask)

        stat_masks = [1] * (len(source_ids))
        padding_length = stat_length - len(source_ids)
        stat_masks += [0] * padding_length
        source_ids += [[tokenizer.pad_token_id] * word_length] * padding_length
        word_masks += [[0] * word_length] * padding_length

        # 如果有标签则处理，没有则创建占位符
        labels = example.labels[:stat_length] if example.labels else [0] * stat_length
        labels += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                word_masks,
                stat_masks,
                labels
            )
        )
    return features


def evaluate(args, model, tokenizer, device, file_path, code_type='cleaned_seqs', prefix=""):
    """
    在给定数据集上评估模型，并返回预测结果。
    """
    eval_examples = read_examples(file_path, code_type)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,
                                                 word_length=args.max_word_length,
                                                 stat_length=args.max_stat_length)

    if len(eval_features) == 0:
        logger.warning("没有读取到有效的特征数据！")
        return []

    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_word_mask = torch.tensor([f.word_masks for f in eval_features], dtype=torch.long)
    all_stat_mask = torch.tensor([f.stat_masks for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_source_ids, all_word_mask, all_stat_mask)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info(f"***** 运行预测 ({prefix}) *****")
    logger.info("  样本数量 (Main + Retrieved) = %d", len(eval_examples))
    logger.info("  批次大小 = %d", args.eval_batch_size)

    model.eval()
    all_preds = []

    for batch in tqdm(eval_dataloader, desc="预测中"):
        batch = tuple(t.to(device) for t in batch)
        source_ids, word_masks, stat_masks = batch

        with torch.no_grad():
            num, active_labels_mask, probs = model(source_ids, word_masks, stat_masks, None)

        prediction = torch.argmax(probs, 1)
        active_preds = prediction[active_labels_mask]

        all_preds.extend(active_preds.cpu().numpy())

    return all_preds


def process_single_entry_predictions(entry_dict, preds, pred_idx, max_stat_len, code_type):
    """
    辅助函数：处理单个字典条目（无论是外层还是 retrieved），填充预测结果。
    返回更新后的 pred_idx。
    """
    # 确定实际用于预测的语句长度（截断逻辑与 convert_features 一致）
    if code_type in entry_dict:
        actual_stat_len = min(len(entry_dict[code_type]), max_stat_len)
    else:
        actual_stat_len = 0  # 如果没有对应的 code_type 字段

    start_idx = pred_idx
    end_idx = pred_idx + actual_stat_len

    # 获取当前 entry 的预测结果片段
    if end_idx > len(preds):
        # 这种情况通常不应该发生，除非数据在 read 和 save 之间发生了变化
        current_preds = preds[start_idx:]
    else:
        current_preds = preds[start_idx: end_idx]

    # 保存由 0/1 组成的列表
    entry_dict['pred_labels'] = [int(p) for p in current_preds]

    # 根据预测结果提取关键语句
    pred_ex_seqs = []
    if code_type in entry_dict:
        seqs = entry_dict[code_type]
        for i, label in enumerate(current_preds):
            if label == 1 and i < len(seqs):
                pred_ex_seqs.append(seqs[i])

    entry_dict['cleaned_seqs_pred'] = pred_ex_seqs

    return end_idx


def save_ex_python_stat(filename, preds, max_stat_len, output_dir, language):
    """
    保存模型的预测结果。
    修改：需要按顺序还原外层和内层的预测。
    """
    base_name = os.path.basename(filename)
    output_filename = os.path.join(output_dir, f"{language}_{base_name.split('.')[0]}_preds.jsonl")

    # 假设 code_type 默认是 'cleaned_seqs'，如果这里有变动需要同步修改
    code_type = 'cleaned_seqs'

    logger.info(f"正在保存预测结果到 {output_filename}")

    with open(filename, encoding="utf-8") as f, jsonlines.open(output_filename, mode='w') as writer:
        pred_idx = 0
        for idx, line in tqdm(enumerate(f), desc="保存预测结果"):
            line = line.strip()
            try:
                js = json.loads(line)
            except json.JSONDecodeError:
                continue

            # --- 1. 填充外层 Main Code 的预测 ---
            pred_idx = process_single_entry_predictions(js, preds, pred_idx, max_stat_len, code_type)

            # --- 2. 填充 Retrieved Candidates 的预测 ---
            candidates = js.get("retrieved_candidates", [])
            for cand in candidates:
                pred_idx = process_single_entry_predictions(cand, preds, pred_idx, max_stat_len, code_type)

            writer.write(js)

        if pred_idx != len(preds):
            logger.warning(f"警告: 预测总数不匹配。剩余 {len(preds) - pred_idx} 个预测未使用。")


def main(language):
    # langs=['julia','lua','ocaml','r','racket']
    langs=['ruby']
    for lang in langs:
        parser = argparse.ArgumentParser()

        ## 基本参数 (与 train.py 保持一致)
        parser.add_argument("--model_type", default='roberta', type=str)
        parser.add_argument("--model_name_or_path", default=transformer_path, type=str)
        parser.add_argument("--output_dir", default=f'../output_structure/{language}', type=str)
        parser.add_argument("--config_name", default="", type=str)
        parser.add_argument("--tokenizer_name", default="", type=str)

        ## 数据文件路径 (与 train.py 保持一致)
        parser.add_argument("--train_filename", default=f'../../../dataset/Clean_PCSD-ast/train/train.jsonl', type=str)
        parser.add_argument("--dev_filename", default=f'../../../dataset/Clean_PCSD-ast/valid/valid.jsonl', type=str)
        parser.add_argument("--test_filename",
                            default=f'../../../dataset/ready_sentences_dataset/Clean_PCSD-ast/test/test.jsonl', type=str)

        ## 模型结构参数 (与 train.py 保持一致)
        parser.add_argument("--max_word_length", default=32, type=int)
        parser.add_argument("--max_stat_length", default=32, type=int)
        parser.add_argument("--word_hidden_size", default=128, type=int)
        parser.add_argument("--stat_hidden_size", default=256, type=int)
        parser.add_argument("--vocab_size", default=50265, type=int)
        parser.add_argument("--embed_size", default=768, type=int)

        ## 预测控制参数

        parser.add_argument("--do_save_ex", default=True, help="是否保存带预测的样本")
        parser.add_argument("--eval_batch_size", default=32, type=int, help="评估/预测批次大小")
        parser.add_argument("--load_model_path", default='../output_structure/python/checkpoint-best-loss/pytorch_model.bin',
                            type=str, help="指定加载模型的路径 (默认为最佳模型)")
        parser.add_argument("--predict_file", default=f"../../../dataset/CSN/ruby/trans_qwen3th/test_sentences.jsonl",
                            type=str, help="要预测的文件")

        ## 其他参数 (与 train.py 保持一致)
        parser.add_argument("--do_lower_case", action='store_true')
        parser.add_argument("--no_cuda", action='store_true')
        parser.add_argument("--local_rank", type=int, default=-1)
        parser.add_argument('--seed', type=int, default=42)

        args = parser.parse_args()
        logger.info(args)

        # 设置设备
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            args.n_gpu = 1
        logger.warning("进程 rank: %s, 设备: %s, n_gpu: %s",
                       args.local_rank, device, args.n_gpu)
        args.device = device
        set_seed(args)

        # 加载分词器和配置
        roberta_path = args.model_name_or_path
        if sysstr == 'Windows' and not os.path.exists(roberta_path):
            logger.warning(f"Windows下找不到路径 {roberta_path}，请检查。")
            roberta_path = 'microsoft/codebert-base'

        config = RobertaConfig.from_pretrained(args.config_name if args.config_name else roberta_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else roberta_path,
                                                     do_lower_case=args.do_lower_case)

        # 加载模型
        encoder = RobertaModel.from_pretrained(roberta_path, config=config)  # 需要 Encoder 来获取词嵌入权重
        word_embeddings_weight = encoder.embeddings.word_embeddings.weight
        model = SelectorNet(batch_size=args.eval_batch_size,  # 使用 eval_batch_size
                            word_embeddings_weight=word_embeddings_weight,
                            word_hidden_size=args.word_hidden_size, stat_hidden_size=args.stat_hidden_size,
                            max_word_len=args.max_word_length, max_stat_len=args.max_stat_length,
                            vocab_size=config.vocab_size, embed_size=config.hidden_size, num_classes=2,
                            imbalance_loss_fct=True)

        # 确定要加载的模型路径
        if args.load_model_path:
            model_path_to_load = args.load_model_path
        else:
            model_path_to_load = os.path.join(args.output_dir, 'checkpoint-best-loss', "pytorch_model.bin")

        if os.path.exists(model_path_to_load):
            logger.info(f"加载模型 {model_path_to_load} 进行预测")
            model.load_state_dict(torch.load(model_path_to_load))
        else:
            logger.error(f"找不到模型文件: {model_path_to_load}。请确保模型已训练或路径正确。")
            sys.exit(1)

        model.to(device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # 确定要预测的文件
        if args.predict_file == "train":
            file_to_predict = args.train_filename
        elif args.predict_file == "valid":
            file_to_predict = args.dev_filename
        elif args.predict_file == "test":
            file_to_predict = args.test_filename
        else:  # 默认为 test
            file_to_predict = args.predict_file

        logger.info(f"***** 开始在 {file_to_predict} 上进行预测 *****")

        # 执行预测
        all_predictions = evaluate(args, model, tokenizer, device, file_to_predict, prefix="Predict")

        # 保存结果
        output_save_dir = os.path.join(args.output_dir, "predictions/trans")
        if not os.path.exists(output_save_dir):
            os.makedirs(output_save_dir)
        save_ex_python_stat(file_to_predict, all_predictions, args.max_stat_length, output_save_dir, language)

        logger.info("***** 预测完成 *****")


if __name__ == "__main__":
    main('python')