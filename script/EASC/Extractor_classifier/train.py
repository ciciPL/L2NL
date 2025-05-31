import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.utils.tensorboard import SummaryWriter # 导入 TensorBoard
import math # 导入 math 用于计算总步数

import os
import argparse
import logging
import random
from tqdm import tqdm, trange # 导入 trange 用于 epoch 循环
# from itertools import cycle # epoch 模式下不再需要 cycle
import json
import jsonlines
import platform

import sys

# 确保 model.py 存在或提供一个占位符
try:
    from model import SelectorNet
except ImportError:
    print("警告: 无法导入 'model.py'。请确保该文件存在。")
    # 如果需要，可以在此处定义一个占位符 SelectorNet 类以进行基本测试
    class SelectorNet(torch.nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.dummy = torch.nn.Linear(10, 2); self.loss_fct = torch.nn.CrossEntropyLoss()
        def forward(self, source_ids, word_masks, stat_masks, labels=None):
            bs, sl, wl = source_ids.shape; logits = self.dummy(torch.randn(bs * sl, 10).to(source_ids.device))
            active = stat_masks.view(-1) == 1; num = active.sum()
            if labels is not None: loss = self.loss_fct(logits, labels.view(-1)); return loss, logits, num, active, torch.softmax(logits, dim=-1)
            else: return None, logits, num, active, torch.softmax(logits, dim=-1)


from transformers import get_linear_schedule_with_warmup, \
    RobertaConfig, RobertaModel, RobertaTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
            labels = js['ex_labels']
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

        labels = example.labels[:stat_length]
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
    在给定数据集上评估模型。

    Args:
        args: 参数对象。
        model: 要评估的模型。
        tokenizer: 分词器。
        device: CPU 或 GPU。
        file_path: 评估数据文件路径。
        code_type: 要使用的代码字段。
        prefix: 日志和输出的前缀。

    Returns:
        eval_loss (float): 评估损失。
        eval_acc (float): 评估准确率。
        all_preds (list): 所有预测标签。
        all_labels (list): 所有真实标签。
    """
    eval_examples = read_examples(file_path, code_type)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,
                                                 word_length=args.max_word_length,
                                                 stat_length=args.max_stat_length)
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_word_mask = torch.tensor([f.word_masks for f in eval_features], dtype=torch.long)
    all_stat_mask = torch.tensor([f.stat_masks for f in eval_features], dtype=torch.long)
    all_labels_tensor = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids, all_word_mask, all_stat_mask, all_labels_tensor)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info(f"***** 运行评估 ({prefix}) *****")
    logger.info("  样本数量 = %d", len(eval_examples))
    logger.info("  批次大小 = %d", args.eval_batch_size)

    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    nb_eval_steps = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(eval_dataloader, desc="评估中"):
        batch = tuple(t.to(device) for t in batch)
        source_ids, word_masks, stat_masks, labels = batch

        with torch.no_grad():
            loss, _, num, active_labels_mask, probs = model(source_ids, word_masks, stat_masks, labels)

        if args.n_gpu > 1:
            loss = loss.mean()

        eval_loss += loss.item()

        prediction = torch.argmax(probs, 1)
        active_preds = prediction[active_labels_mask]
        active_true_labels = labels.view(-1)[active_labels_mask]

        cur_acc = (active_preds == active_true_labels).sum().float()
        eval_acc += (cur_acc / num.sum()).cpu().detach().data.numpy()

        all_preds.extend(active_preds.cpu().numpy())
        all_labels.extend(active_true_labels.cpu().numpy())
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_acc = eval_acc / nb_eval_steps

    logger.info("***** 评估结果 ({}) *****".format(prefix))
    logger.info("  评估损失 = %s", np.round(eval_loss, 5))
    logger.info("  评估准确率 = %s", np.round(eval_acc, 5))

    return eval_loss, eval_acc, all_preds, all_labels


def main(language):
    parser = argparse.ArgumentParser()

    ## 基本参数
    parser.add_argument("--model_type", default='roberta', type=str, help="模型类型")
    parser.add_argument("--model_name_or_path", default=transformer_path, type=str, help="预训练模型路径或名称")
    parser.add_argument("--output_dir", default=f'../output/{language}', type=str, help="输出目录")
    parser.add_argument("--load_model_path", default=None, type=str, help="加载已训练模型的路径")
    parser.add_argument("--config_name", default="", type=str, help="配置文件名称或路径")
    parser.add_argument("--tokenizer_name", default="", type=str, help="分词器名称或路径")

    ## 数据文件路径
    parser.add_argument("--train_filename", default=f'../../../dataset/Clean_PCSD-ast/train/train.jsonl', type=str)
    parser.add_argument("--dev_filename", default=f'../../../dataset/Clean_PCSD-ast/valid/valid.jsonl', type=str)
    parser.add_argument("--test_filename", default=f'../../../dataset/Clean_PCSD-ast/test/test.jsonl', type=str)

    ## 模型结构参数
    parser.add_argument("--max_word_length", default=32, type=int, help="单个语句的最大词数")
    parser.add_argument("--max_stat_length", default=32, type=int, help="单个样本的最大语句数")
    parser.add_argument("--word_hidden_size", default=128, type=int)
    parser.add_argument("--stat_hidden_size", default=256, type=int)
    parser.add_argument("--vocab_size", default=50265, type=int)
    parser.add_argument("--embed_size", default=768, type=int)

    ## 训练控制参数
    parser.add_argument("--do_train", default=True, help="是否进行训练")
    parser.add_argument("--do_eval", default=True, help="是否在开发集上评估")
    parser.add_argument("--do_test", default=False, help="是否在测试集上测试")
    parser.add_argument("--do_save_ex", default=True, help="是否保存带预测的样本")
    parser.add_argument("--train_batch_size", default=32, type=int, help="训练批次大小")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="评估批次大小")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--weight_decay", default=0, type=float, help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam优化器epsilon")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="预热步数比例") # 改为比例
    parser.add_argument("--num_train_epochs", default=10, type=int, help="训练的总轮数") # 改为 Epoch
    parser.add_argument("--eval_every_epoch", default=1, type=int, help="每隔多少轮评估一次") # 评估频率

    ## 早停与 TensorBoard
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="早停耐心值 (多少轮不提升则停止)")
    parser.add_argument("--tensorboard_log_dir", type=str, default=f'../output/{language}/runs', help="TensorBoard日志目录")

    ## 其他参数
    parser.add_argument("--do_lower_case", action='store_true', help="是否使用小写模型")
    parser.add_argument("--no_cuda", action='store_true', help="是否不使用CUDA")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的 local_rank")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")

    args = parser.parse_args()
    logger.info(args)

    # TensorBoard Writer
    if args.local_rank in [-1, 0]: # 仅在主进程中初始化
        tb_writer = SummaryWriter(args.tensorboard_log_dir)

    # 设置设备
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("进程 rank: %s, 设备: %s, n_gpu: %s, 分布式训练: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    set_seed(args)

    # 创建输出目录
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # 加载模型和分词器
    roberta_path = args.model_name_or_path
    if sysstr == 'Windows' and not os.path.exists(roberta_path):
        logger.warning(f"Windows下找不到路径 {roberta_path}，请检查。")
        # 可以考虑在此处提供一个备用路径或下载逻辑
        roberta_path = 'microsoft/codebert-base' # 例如，使用在线模型

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else roberta_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else roberta_path,
                                                  do_lower_case=args.do_lower_case)
    encoder = RobertaModel.from_pretrained(roberta_path, config=config)

    word_embeddings_weight = encoder.embeddings.word_embeddings.weight
    model = SelectorNet(batch_size=args.train_batch_size, word_embeddings_weight=word_embeddings_weight,
                        word_hidden_size=args.word_hidden_size, stat_hidden_size=args.stat_hidden_size,
                        max_word_len=args.max_word_length, max_stat_len=args.max_stat_length,
                        vocab_size=config.vocab_size, embed_size=config.hidden_size, num_classes=2, # 使用 config 的值
                        imbalance_loss_fct=True)

    if args.load_model_path is not None:
        logger.info("从 {} 加载模型".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    # 分布式/多GPU训练设置
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            model = DDP(model)
        except ImportError:
            raise ImportError("请安装 apex (https://www.github.com/nvidia/apex) 以使用分布式和 fp16 训练。")
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 训练流程
    if args.do_train:
        train_examples = read_examples(args.train_filename, 'cleaned_seqs')
        train_features = convert_examples_to_features(train_examples, tokenizer, args,
                                                      word_length=args.max_word_length,
                                                      stat_length=args.max_stat_length)
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_word_mask = torch.tensor([f.word_masks for f in train_features], dtype=torch.long)
        all_stat_mask = torch.tensor([f.stat_masks for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_word_mask, all_stat_mask, all_labels)

        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # 计算总训练步数
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        num_warmup_steps = int(t_total * args.warmup_ratio) # 使用比例计算预热步数

        # 准备优化器和学习率调度器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=t_total)

        logger.info("***** 开始训练 *****")
        logger.info("  样本数量 = %d", len(train_examples))
        logger.info("  批次大小 = %d", args.train_batch_size)
        logger.info("  训练轮数 = %d", args.num_train_epochs)
        logger.info("  总优化步数 = %d", t_total)
        logger.info("  预热步数 = %d", num_warmup_steps)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_eval_loss = float('inf')
        epochs_no_improve = 0

        model.zero_grad()
        train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

        for epoch_idx, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_steps = 0

            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                source_ids, word_masks, stat_masks, labels = batch
                loss, _, num, active_labels, probs = model(source_ids, word_masks, stat_masks, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                epoch_loss += loss.item()

                # 计算准确率
                prediction = torch.argmax(probs, 1)
                cur_acc = (prediction[active_labels] == labels.view(-1)[active_labels]).sum().float()
                batch_acc = (cur_acc / num.sum()).cpu().detach().data.numpy()
                epoch_acc += batch_acc

                epoch_steps += 1
                epoch_iterator.set_description(f"Epoch {epoch_idx+1} | Loss: {epoch_loss/epoch_steps:.4f} | Acc: {epoch_acc/epoch_steps:.4f}")

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # TensorBoard 日志 (每 N 步记录一次)
                    if args.local_rank in [-1, 0] and global_step % 50 == 0:
                        tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar('Loss/train_step', (tr_loss - logging_loss)/50, global_step)
                        logging_loss = tr_loss

            # --- Epoch 结束 ---
            logger.info(f"Epoch {epoch_idx + 1} 结束。平均训练 Loss: {epoch_loss / epoch_steps:.4f}, 平均训练 Acc: {epoch_acc / epoch_steps:.4f}")

            # TensorBoard 日志 (每轮记录一次)
            if args.local_rank in [-1, 0]:
                tb_writer.add_scalar('Loss/train_epoch', epoch_loss / epoch_steps, epoch_idx + 1)
                tb_writer.add_scalar('Accuracy/train_epoch', epoch_acc / epoch_steps, epoch_idx + 1)

            # 在开发集上评估
            if args.do_eval and args.local_rank in [-1, 0] and (epoch_idx + 1) % args.eval_every_epoch == 0:
                eval_loss, eval_acc, _, _ = evaluate(args, model, tokenizer, device, args.dev_filename, prefix=f"Epoch {epoch_idx + 1}")

                # TensorBoard 日志
                tb_writer.add_scalar('Loss/eval', eval_loss, epoch_idx + 1)
                tb_writer.add_scalar('Accuracy/eval', eval_acc, epoch_idx + 1)

                # 保存最后一个模型
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir): os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info(f"已保存最后一个模型到 {output_model_file}")

                # 检查是否是最佳模型 (基于评估损失)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    epochs_no_improve = 0
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-loss')
                    if not os.path.exists(output_dir): os.makedirs(output_dir)
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info(f"*** 发现新的最佳损失模型 ({best_eval_loss:.5f})，已保存到 {output_model_file} ***")
                else:
                    epochs_no_improve += 1
                    logger.info(f"评估损失未提升。当前最佳损失: {best_eval_loss:.5f}。连续未提升轮数: {epochs_no_improve}")

                # 早停检查
                if epochs_no_improve >= args.early_stopping_patience:
                    logger.info(f"*** 早停触发！连续 {args.early_stopping_patience} 轮评估损失未提升。停止训练。 ***")
                    train_iterator.close()
                    break # 跳出 epoch 循环

            if epochs_no_improve >= args.early_stopping_patience: # 再次检查以确保完全退出
                break

        if args.local_rank in [-1, 0]:
            tb_writer.close() # 关闭 TensorBoard writer

    # 测试流程
    if args.do_test and args.local_rank in [-1, 0]:
        logger.info("***** 开始测试 *****")
        # 加载最佳模型进行测试
        best_model_path = os.path.join(args.output_dir, 'checkpoint-best-loss', "pytorch_model.bin")
        if os.path.exists(best_model_path):
            logger.info(f"加载最佳模型 {best_model_path} 进行测试")
            # 重新加载模型以避免 DataParallel 包装问题
            config = RobertaConfig.from_pretrained(roberta_path)
            encoder = RobertaModel.from_pretrained(roberta_path, config=config)
            word_embeddings_weight = encoder.embeddings.word_embeddings.weight
            model = SelectorNet(batch_size=args.eval_batch_size, word_embeddings_weight=word_embeddings_weight, # 使用 eval batch size
                                word_hidden_size=args.word_hidden_size, stat_hidden_size=args.stat_hidden_size,
                                max_word_len=args.max_word_length, max_stat_len=args.max_stat_length,
                                vocab_size=config.vocab_size, embed_size=config.hidden_size, num_classes=2,
                                imbalance_loss_fct=True)
            model.load_state_dict(torch.load(best_model_path))
            model.to(device)
            if args.n_gpu > 1: # 如果训练时用了 DataParallel，测试时也需要
                model = torch.nn.DataParallel(model)
        else:
            logger.warning("找不到最佳模型，将使用当前模型进行测试 (可能是最后一个模型)。")

        test_loss, test_acc, test_preds, test_labels = evaluate(args, model, tokenizer, device, args.test_filename, prefix="Test")
        logger.info(f"***** 测试结果 *****")
        logger.info(f"  测试损失 = {test_loss:.5f}")
        logger.info(f"  测试准确率 = {test_acc:.5f}")

        # 如果需要，保存预测结果
        if args.do_save_ex:
            save_ex_python_stat(args.test_filename, test_preds, args.max_stat_length, args.output_dir, language)

# ... (保持 save_ex_stat 和 save_ex_python_stat 函数不变) ...
def save_ex_stat(filename, preds, max_stat_len):
    with open(filename, encoding="utf-8") as f,jsonlines.open('output-php.jsonl', mode='a') as writer:
        pred_idx = 0
        for idx, line in tqdm(enumerate(f)):
            line = line.strip()
            js = json.loads(line)

            stat_len = min(len(js['cleaned_seqs']),max_stat_len)

            js['cleaned_seqs_pred'] = preds[pred_idx,pred_idx+stat_len]

            writer.write(js)

def save_ex_python_stat(filename, preds, max_stat_len, output_dir, language):
    """
    保存模型的预测结果以及原始数据。

    Args:
        filename (str): 输入 .jsonl 文件的路径 (例如, test.jsonl)。
        preds (list): 包含所有预测结果 (0 和 1) 的扁平列表。
        max_stat_len (int): 每个样本处理的最大语句数。
        output_dir (str): 输出文件应保存的目录。
        language (str): 正在处理的语言 (例如, 'python', 'java')。
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
            elif 'ex_labels' in js:
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

            js['pred_labels'] = [int(p) for p in current_preds] # 确保是整数

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        language = sys.argv[1]
        main(language)
    else:
        print("请提供语言作为命令行参数，例如: python train.py python")
        # 或者设置一个默认值进行测试:
        main('python')