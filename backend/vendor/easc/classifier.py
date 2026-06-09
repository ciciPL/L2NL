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
    class SelectorNet(torch.nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.dummy = torch.nn.Linear(10, 2); self.loss_fct = torch.nn.CrossEntropyLoss()
        def forward(self, source_ids, word_masks, stat_masks, labels=None):
            bs, sl, wl = source_ids.shape; logits = self.dummy(torch.randn(bs * sl, 10).to(source_ids.device))
            active = stat_masks.view(-1) == 1; num = active.sum()
            if labels is not None: loss = self.loss_fct(logits, labels.view(-1)); return loss, logits, num, active, torch.softmax(logits, dim=-1)
            else: return None, logits, num, active, torch.softmax(logits, dim=-1)


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
    """从 .jsonl 文件读取数据样本。"""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), desc=f"读取 {os.path.basename(filename)}"):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            code = [i.replace('\n', ' ') for i in js[code_type]]
            # 在预测时，我们可能没有真实标签，或者不需要它们，但为了复用结构，可以传入空列表或真实标签
            labels = js.get('ex_labels', []) # 尝试获取标签，如果没有则为空
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    labels=labels,
                )
            )
    return examples


def convert_examples_to_features(examples, tokenizer, args, word_length=None, stat_length=None):
    """将数据样本转换为模型输入特征。"""
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
    (与 train.py 中的版本基本一致，但更侧重于返回预测)
    """
    eval_examples = read_examples(file_path, code_type)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,
                                                 word_length=args.max_word_length,
                                                 stat_length=args.max_stat_length)
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_word_mask = torch.tensor([f.word_masks for f in eval_features], dtype=torch.long)
    all_stat_mask = torch.tensor([f.stat_masks for f in eval_features], dtype=torch.long)
    # all_labels_tensor = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids, all_word_mask, all_stat_mask)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info(f"***** 运行预测 ({prefix}) *****")
    logger.info("  样本数量 = %d", len(eval_examples))
    logger.info("  批次大小 = %d", args.eval_batch_size)

    model.eval() # 确保模型在评估模式
    all_preds = []
    all_labels = [] # 也可以收集真实标签以计算准确率

    for batch in tqdm(eval_dataloader, desc="预测中"):
        batch = tuple(t.to(device) for t in batch)
        source_ids, word_masks, stat_masks = batch

        with torch.no_grad():
            # 将 labels=None 改为 labels，这样就会把从数据加载器中得到的 labels 传入模型
            # loss, _, num, active_labels_mask, probs = model(source_ids, word_masks, stat_masks,labels)
            num,active_labels_mask, probs = model(source_ids, word_masks, stat_masks, None)
        prediction = torch.argmax(probs, 1)
        active_preds = prediction[active_labels_mask]
        # active_true_labels = labels.view(-1)[active_labels_mask]

        all_preds.extend(active_preds.cpu().numpy())
        # all_labels.extend(active_true_labels.cpu().numpy())

    # 计算整体准确率 (可选)
    all_preds_np = np.array(all_preds)
    # all_labels_np = np.array(all_labels)
    # if len(all_labels_np) > 0:
    #     accuracy = (all_preds_np == all_labels_np).mean()
    #     logger.info(f"***** 预测结果 ({prefix}) *****")
    #     logger.info("  预测准确率 = %s", np.round(accuracy, 5))
    # else:
    #      logger.info("***** 预测完成 ({}) *****".format(prefix))


    return all_preds


def save_ex_python_stat(filename, preds, max_stat_len, output_dir, language):
    """
    保存模型的预测结果以及原始数据。
    (与 train.py 中的版本一致)
    """
    base_name = os.path.basename(filename)
    output_filename = os.path.join(output_dir, f"{language}_{base_name.split('.')[0]}_preds.jsonl")

    logger.info(f"正在保存预测结果到 {output_filename}")

    with open(filename, encoding="utf-8") as f, jsonlines.open(output_filename, mode='w') as writer:
        pred_idx = 0
        for idx, line in tqdm(enumerate(f), desc="保存预测结果"):
            line = line.strip()
            js = json.loads(line)

            if 'cleaned_seqs' in js:
                actual_stat_len = min(len(js['cleaned_seqs']), max_stat_len)
            elif 'ex_labels' in js: # 备选方案
                actual_stat_len = min(len(js['ex_labels']), max_stat_len)
            else:
                logger.warning(f"无法确定 idx {js.get('idx', idx)} 的语句长度。假设为 max_stat_len。")
                actual_stat_len = max_stat_len

            start_idx = pred_idx
            end_idx = pred_idx + actual_stat_len

            if end_idx > len(preds):
                logger.error(f"预测索引超出范围 idx {js.get('idx', idx)}。 "
                             f"需要 {actual_stat_len} 个预测，但只剩下 {len(preds) - start_idx} 个。")
                current_preds = preds[start_idx:]
            else:
                current_preds = preds[start_idx: end_idx]

            js['pred_labels'] = [int(p) for p in current_preds]

            pred_ex_seqs = []
            if 'cleaned_seqs' in js:
                for i, label in enumerate(current_preds):
                    if label == 1 and i < len(js['cleaned_seqs']):
                        pred_ex_seqs.append(js['cleaned_seqs'][i])
            js['cleaned_seqs_pred'] = pred_ex_seqs

            writer.write(js)
            pred_idx = end_idx

        if pred_idx != len(preds):
            logger.warning(f"警告: {len(preds) - pred_idx} 个预测未使用。")

def main(language):
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
    parser.add_argument("--test_filename", default=f'../../../dataset/ready_sentences_dataset/Clean_PCSD-ast/test/test.jsonl', type=str)

    ## 模型结构参数 (与 train.py 保持一致)
    parser.add_argument("--max_word_length", default=32, type=int)
    parser.add_argument("--max_stat_length", default=32, type=int)
    parser.add_argument("--word_hidden_size", default=128, type=int)
    parser.add_argument("--stat_hidden_size", default=256, type=int)
    parser.add_argument("--vocab_size", default=50265, type=int)
    parser.add_argument("--embed_size", default=768, type=int)
    lang = 'ocaml'
    ## 预测控制参数
    parser.add_argument("--do_save_ex", default=True, help="是否保存带预测的样本")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="评估/预测批次大小")
    parser.add_argument("--load_model_path", default='../output_structure/python/checkpoint-best-loss/pytorch_model.bin', type=str, help="指定加载模型的路径 (默认为最佳模型)")
    parser.add_argument("--predict_file", default=f"../../../dataset/LowData/{lang}/{lang}_structure_B0.5_S0.5.jsonl", type=str, help="要预测的文件")
    #修改了output dir路径，加了trans。保存来自LRPL翻译来的python code。然后根据这个python code分割语句，做预测。与LRPL语句预测做区分 //TODO 需要修改dir路径
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
    encoder = RobertaModel.from_pretrained(roberta_path, config=config) # 需要 Encoder 来获取词嵌入权重
    word_embeddings_weight = encoder.embeddings.word_embeddings.weight
    model = SelectorNet(batch_size=args.eval_batch_size, # 使用 eval_batch_size
                        word_embeddings_weight=word_embeddings_weight,
                        word_hidden_size=args.word_hidden_size, stat_hidden_size=args.stat_hidden_size,
                        max_word_len=args.max_word_length, max_stat_len=args.max_stat_length,
                        vocab_size=config.vocab_size, embed_size=config.hidden_size, num_classes=2,
                        imbalance_loss_fct=True) # 此处 imbalance_loss_fct 设置应与训练时一致，但不影响预测

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
        sys.exit(1) # 如果找不到模型则退出

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
    else: # 默认为 test
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