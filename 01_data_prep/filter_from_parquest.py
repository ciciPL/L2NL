import pandas as pd
import os
import re
import json
from tqdm import tqdm
import spacy
import language_tool_python
from transformers import AutoModel, AutoTokenizer
import torch

# ==========================================
#            第0步：初始化工具
# ==========================================

print("正在初始化工具...")

# spaCy (用于动词检测)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("请先安装: python -m spacy download en_core_web_sm")
    exit(1)

# LanguageTool (用于语法检查)
grammar_tool = language_tool_python.LanguageTool('en-US',language_tool_download_version='6.0')
# grammar_tool = None

# Jina-Code (用于语义相似度)
print("加载 jina-code 嵌入模型...")
JINA_MODEL_PATH = os.environ.get(
    "JINA_MODEL_PATH",
    "./models/jina-code-embeddings-1.5b"   # download from HuggingFace: jinaai/jina-code-embeddings-1.5b
)
jina_model = AutoModel.from_pretrained(JINA_MODEL_PATH)
jina_tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL_PATH)
jina_model = jina_model.cuda()  # 使用GPU
jina_model.eval()

print("✅ 工具初始化完成\n")


# ==========================================
#         工具函数：核心清洗逻辑
# ==========================================

def extract_first_sentence_or_line(raw_comment: str) -> str:
    """
    从原始 docstring 中提取“第一句或第一行”
    - 不做任何清洗
    - 不移除注释符号
    - 只做结构性截断
    """
    if not raw_comment or not raw_comment.strip():
        return None

    text = raw_comment.strip()

    # 统一换行
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return None

    # 合并为一段，便于句号检测
    joined = " ".join(lines)

    # 1. 尝试按“第一句”截断
    sentence_match = re.search(r'(.+?[.!?])(\s|$)', joined)
    if sentence_match:
        return sentence_match.group(1).strip()

    # 2. 退化策略：第一非空行
    return lines[0].strip()


def extract_raw_comment(code, lang):
    """
    根据语言特性，提取原始的文档注释块。
    返回: 未经清洗的原始注释字符串 (包含注释符号 #, --, ;; 等)
    """
    if not isinstance(code, str) or not code.strip():
        return None

    code = code.strip()

    # --- Julia: 优先匹配标准 Docstring """ ... """ ---
    if lang == 'julia':
        match = re.search(r'^"""(.*?)"""', code, re.DOTALL)
        if match: return match.group(1)
        match_single = re.search(r'^"(.*?)"', code, re.DOTALL)
        if match_single: return match_single.group(1)
        return None

    # --- Lua: 锚定提取 (紧贴 function 的 -- 块) ---
    elif lang == 'lua':
        pattern = r'((?:^\s*--.*$\n?)+)(?=\s*(?:local\s+)?function)'
        matches = re.findall(pattern, code, re.MULTILINE)
        if matches:
            return max(matches, key=len)

    # --- OCaml: 标准文档 (** ... *) ---
    elif lang == 'ocaml':
        match = re.search(r'\(\*\*(.*?)\*\)', code, re.DOTALL)
        if match: return match.group(1)

    # --- R: 锚定提取 (紧贴 function 的 # 块) ---
    elif lang == 'r':
        pattern = r'((?:^\s*#.*$\n?)+)(?=\s*[\w\.]+\s*(?:<-|=)\s*function)'
        matches = re.findall(pattern, code, re.MULTILINE)
        if matches:
            return max(matches, key=len)

    # --- Racket: 锚定提取 (紧贴 define 的 ;; 块) ---
    elif lang == 'racket':
        pattern = r'((?:^\s*;;.*$\n?)+)(?=\s*\(define)'
        matches = re.findall(pattern, code, re.MULTILINE)
        if matches:
            return max(matches, key=len)

    return None


def extract_code_only(content, lang):
    """
    从content中提取纯代码部分（去除docstring）
    用于Layer 4的语义相似度计算
    """
    if not isinstance(content, str) or not content.strip():
        return None

    content = content.strip()

    # --- Julia ---
    if lang == 'julia':
        # 移除 """ ... """
        content = re.sub(r'^""".*?"""', '', content, flags=re.DOTALL)
        # 移除单引号docstring
        content = re.sub(r'^".*?"', '', content, flags=re.DOTALL)

    # --- Lua ---
    elif lang == 'lua':
        # 移除紧贴function前的 -- 注释块
        content = re.sub(r'^\s*--.*$', '', content, flags=re.MULTILINE)

    # --- OCaml ---
    elif lang == 'ocaml':
        # 移除 (** ... *)
        content = re.sub(r'\(\*\*.*?\*\)', '', content, flags=re.DOTALL)

    # --- R ---
    elif lang == 'r':
        # 移除紧贴function前的 # 注释块
        content = re.sub(r'^\s*#.*$', '', content, flags=re.MULTILINE)

    # --- Racket ---
    elif lang == 'racket':
        # 移除紧贴define前的 ;; 注释块
        content = re.sub(r'^\s*;;.*$', '', content, flags=re.MULTILINE)

    # 清理空行
    lines = [l for l in content.split('\n') if l.strip()]
    return '\n'.join(lines)


def remove_comment_markers(text, lang):
    """去除各语言的注释符号"""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        s = line.strip()
        if not s: continue

        # 去除前缀
        if lang == 'lua':
            s = s.lstrip('-').strip()
        elif lang in ['r', 'julia']:
            s = s.lstrip('#').lstrip("'").strip()
        elif lang == 'racket':
            s = s.lstrip(';').strip()
        elif lang == 'ocaml':
            s = s.lstrip('*').strip()

        if s:
            cleaned_lines.append(s)

    return "\n".join(cleaned_lines)


def clean_summary(text, lang):
    """
    清洗summary，但不修改内容（不移除URL等）
    只做格式化和截断
    """
    if not text: return None

    # 1. 去除注释符号
    text = remove_comment_markers(text, lang)
    if not text: return None

    # 2. 垃圾数据熔断 (Fast Fail)
    if re.match(r'^:type', text) or re.match(r'^:rtype', text) or re.match(r'^>>>', text):
        return None

    # 3. 强力截断 (Cutoff)
    cutoff_keywords = [
        # --- Python / Sphinx 风格 ---
        r'Args:', r'Arguments:',
        r'Parameters', r'----------',
        r'Returns:', r'Returns', r'Yields:', r'Raises:',
        r':param', r':return', r':rtype', r':raise',
        r'\.\. code::', r'\.\. math::',

        # --- Javadoc / Doxygen / LuaDoc 风格 (新增) ---
        r'@param', r'@return', r'@arg', r'@field',
        r'@brief', r'@see', r'@usage',

        # --- 示例标记 (新增 E.g.) ---
        r'Examples:', r'Example:', r'>>>',
        r'E\.g\.', r'e\.g\.', r'i\.e\.',  # 增加常见缩写

        # --- 其他常见分割线 ---
        r'###', r'==='
    ]

    for pat in cutoff_keywords:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]

    # 4. 移除残留的 Sphinx 标记 (行内)
    text = re.sub(r':[a-z]+ .*?:', '', text)

    # 5. 最终格式化
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. 长度检查（下限和上限）
    word_count = len(text.split())
    if word_count < 3: return None  # 下限：3个词
    if word_count > 50: return None  # 上限：50个词

    if len(text) < 10: return None  # 字符下限

    # 7. 关键词过滤
    if "TODO" in text or "FIXME" in text:
        return None

    return text


def normalize(s):
    """归一化：仅保留小写字母和数字"""
    return re.sub(r'[^a-z0-9]', '', s.lower())


def verify_real(summary, original_content):
    """验证摘要是否在原始content中（数据完整性验证）"""
    if not summary: return False
    norm_sum = normalize(summary)
    norm_content = normalize(original_content)
    return norm_sum in norm_content


# ==========================================
#           Layer 2: 代码质量过滤
# ==========================================

def filter_code_quality(code, lang):
    """
    检查代码质量（在提取docstring之前）
    """
    if not code or not code.strip():
        return False, "empty_code"

    # 1. 行数检查
    lines = [l for l in code.split('\n') if l.strip()]
    line_count = len(lines)

    if line_count < 3:
        return False, "too_short"
    if line_count > 150:
        return False, "too_long"

    # 2. 测试函数检查（简单版）
    func_def_line = lines[0] if lines else ""
    if 'test' in func_def_line.lower() or 'Test' in func_def_line:
        return False, "test_function"

    # 3. 空函数检查（简单版：检查是否只有return/pass）
    code_lower = code.lower()
    content_lines = [l for l in lines if l.strip() and not l.strip().startswith(('#', '--', ';;', '(*'))]

    if len(content_lines) <= 2:
        # 只有函数定义+一行return/pass
        if 'pass' in code_lower or (content_lines and 'return' in content_lines[-1].lower()):
            return False, "stub_function"

    return True, "ok"


# ==========================================
#        Layer 2: 文档基础过滤（补充）
# ==========================================

def check_identifier_repetition(summary, code, lang):
    """
    检查summary是否只是重复代码的标识符
    阈值：如果标识符占比 > 70%，拒绝
    """
    # 提取函数名和参数名（简单版）
    identifiers = set()

    # 提取函数名
    if lang == 'julia':
        match = re.search(r'function\s+(\w+)', code)
        if match: identifiers.add(match.group(1).lower())
    elif lang == 'lua':
        match = re.search(r'function\s+(\w+)', code)
        if match: identifiers.add(match.group(1).lower())
    elif lang == 'r':
        match = re.search(r'(\w+)\s*(?:<-|=)\s*function', code)
        if match: identifiers.add(match.group(1).lower())
    elif lang == 'racket':
        match = re.search(r'\(define\s+\((\w+)', code)
        if match: identifiers.add(match.group(1).lower())
    elif lang == 'ocaml':
        match = re.search(r'let\s+(\w+)', code)
        if match: identifiers.add(match.group(1).lower())

    # 提取参数名（简单版：括号内的标识符）
    param_match = re.findall(r'\(([^)]+)\)', code)
    if param_match:
        params = param_match[0].split(',')
        for p in params:
            p = p.strip().split()[0]  # 取第一个词（去掉类型）
            if p and p.isidentifier():
                identifiers.add(p.lower())

    # 计算summary中标识符占比
    summary_words = [w.lower() for w in re.findall(r'\w+', summary)]
    if not summary_words:
        return False

    identifier_count = sum(1 for w in summary_words if w in identifiers)
    ratio = identifier_count / len(summary_words)

    return ratio <= 0.7  # 允许70%以下


def check_license_copyright(summary):
    """检查是否包含License/Copyright"""
    patterns = [
        r'copyright', r'license', r'all rights reserved',
        r'MIT License', r'GPL', r'Apache'
    ]
    summary_lower = summary.lower()
    return not any(re.search(pat, summary_lower) for pat in patterns)


# ==========================================
#         Layer 3: 内容质量检查
# ==========================================

def check_token_overlap(summary, code, lang):
    """
    检查summary与code的token重复度
    返回overlap比例
    """
    # 提取code中的标识符
    code_tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', code.lower()))

    # 提取summary中的词（去除停用词）
    stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'it', 'for', 'on', 'with', 'as', 'by'}
    summary_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z_]\w*\b', summary)]

    # 去除停用词
    summary_clean = [w for w in summary_tokens if w not in stopwords]

    if not summary_clean:
        return 0.0

    # 计算overlap
    overlap = code_tokens & set(summary_clean)
    ratio = len(overlap) / len(summary_clean)

    return ratio


def has_verb_spacy(summary):
    """用spaCy检测是否包含动词"""
    doc = nlp(summary)
    return any(token.pos_ == "VERB" for token in doc)


def check_informative_content(summary):
    """检查summary是否有信息量"""
    # 1. 检查是否包含动词
    if not has_verb_spacy(summary):
        return False, "no_verb"

    # 2. 检查是否是纯类型声明
    if re.match(r'^\w+\s*->\s*\w+$', summary):
        return False, "type_declaration"

    # 3. 检查是否是问题/TODO
    if summary.startswith('How to') or summary.startswith('TODO'):
        return False, "question_or_todo"

    # 4. 检查实际内容词数量
    stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'it', 'for', 'on', 'with', 'as', 'by'}
    content_words = [w for w in summary.split() if w.lower() not in stopwords]

    if len(content_words) < 2:
        return False, "insufficient_content"

    return True, "ok"


def check_grammar_quality(summary):
    """用LanguageTool检查语法质量"""
    matches = grammar_tool.check(summary)

    # 只计算严重的语法错误
    serious_errors = [m for m in matches if m.ruleIssueType in ['grammar', 'misspelling']]

    return len(serious_errors) < 1, len(serious_errors)


# ==========================================
#         Layer 4: 语义质量（Jina-Code）
# ==========================================

def compute_semantic_similarity(code, summary):
    """
    用jina-code计算code和summary的语义相似度
    """
    # Encode code
    code_inputs = jina_tokenizer(
        code,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    ).to('cuda')

    # Encode summary
    sum_inputs = jina_tokenizer(
        summary,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    ).to('cuda')

    with torch.no_grad():
        # Mean pooling
        code_outputs = jina_model(**code_inputs)
        code_emb = code_outputs.last_hidden_state.mean(dim=1)

        sum_outputs = jina_model(**sum_inputs)
        sum_emb = sum_outputs.last_hidden_state.mean(dim=1)

    # Cosine similarity
    similarity = torch.cosine_similarity(code_emb, sum_emb, dim=1).item()

    return similarity


# ==========================================
#         Step 1: 采样原始数据
# ==========================================

def step_1_sample_raw_data(config):
    """步骤一：读取 Parquet -> 取前N% -> 保存为 JSONL"""
    print("\n>>> [步骤一] 开始采样原始数据 (Parquet -> JSONL) <<<")
    os.makedirs(config['raw_jsonl_dir'], exist_ok=True)

    for lang in config['langs']:
        input_path = os.path.join(config['base_dir'], lang, f"{lang}-00000-of-00001.parquet")
        output_path = os.path.join(config['raw_jsonl_dir'], f"{lang}_raw_sample.jsonl")

        if not os.path.exists(input_path):
            print(f"[跳过] 源文件缺失: {input_path}")
            continue

        try:
            df = pd.read_parquet(input_path)
            total_len = len(df)
            sample_size = int(total_len * config['sample_ratio'])
            df_sample = df.head(sample_size).copy()

            if 'index' not in df_sample.columns:
                df_sample['index'] = range(len(df_sample))

            target_cols = ['index', 'content']
            final_df = df_sample[target_cols]

            final_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
            print(f"✅ [{lang}] 采样完成: {sample_size}/{total_len} -> {output_path}")

        except Exception as e:
            print(f"❌ [{lang}] 处理失败: {e}")


# ==========================================
#      Step 2: 多层过滤清洗（完整版）
# ==========================================

def step_2_multilayer_filter(config):
    """
    步骤二：5层过滤清洗
    Layer 1: 结构完整性
    Layer 2: 基础质量
    Layer 3: 内容质量
    Layer 4: 语义质量
    Layer 5: 去重采样
    """
    print("\n>>> [步骤二] 开始多层过滤清洗 <<<")

    os.makedirs(config['intermediate_dir'], exist_ok=True)

    for lang in config['langs']:
        input_path = os.path.join(config['raw_jsonl_dir'], f"{lang}_raw_sample.jsonl")

        if not os.path.exists(input_path):
            print(f"⚠️ [跳过] 找不到: {input_path}")
            continue

        print(f"\n{'=' * 50}")
        print(f"🔵 正在处理 [{lang.upper()}]")
        print(f"{'=' * 50}")

        # 统计信息
        stats = {
            'total': 0,
            'layer1_pass': 0,  # 结构完整性
            'layer2_pass': 0,  # 基础质量
            'layer3_pass': 0,  # 内容质量
            'layer4_pass': 0,  # 语义质量
        }

        reject_reasons = {}

        candidates = []

        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"{lang} - Layer 1-4"):
            stats['total'] += 1

            try:
                row = json.loads(line)
                idx = row.get('index')
                content = row.get('content')

                # ============ Layer 1: 结构完整性 ============
                if not content or not content.strip():
                    reject_reasons['empty_content'] = reject_reasons.get('empty_content', 0) + 1
                    continue

                # 提取docstring
                raw_comment = extract_raw_comment(content, lang)
                if not raw_comment:
                    reject_reasons['no_docstring'] = reject_reasons.get('no_docstring', 0) + 1
                    continue
                # TODO 这个可以修改逻辑，用于提取doc第一句或第一行
                # ============ 新增：第一句 / 第一行提取（CodeSearchNet 风格） ============
                first_unit = extract_first_sentence_or_line(raw_comment)
                if not first_unit:
                    reject_reasons['first_sentence_failed'] = reject_reasons.get('first_sentence_failed', 0) + 1
                    continue

                # 清洗docstring
                clean_doc = clean_summary(raw_comment, lang)
                if not clean_doc:
                    reject_reasons['clean_failed'] = reject_reasons.get('clean_failed', 0) + 1
                    continue

                # 验证完整性
                if not verify_real(clean_doc, content):
                    reject_reasons['verify_failed'] = reject_reasons.get('verify_failed', 0) + 1
                    continue

                stats['layer1_pass'] += 1

                # ============ Layer 2: 基础质量 ============
                # 2a. 代码质量
                code_ok, code_reason = filter_code_quality(content, lang)
                if not code_ok:
                    reject_reasons[f'code_{code_reason}'] = reject_reasons.get(f'code_{code_reason}', 0) + 1
                    continue

                # 2b. 标识符重复检查
                if not check_identifier_repetition(clean_doc, content, lang):
                    reject_reasons['identifier_repetition'] = reject_reasons.get('identifier_repetition', 0) + 1
                    continue

                # 2c. License/Copyright
                if not check_license_copyright(clean_doc):
                    reject_reasons['license_copyright'] = reject_reasons.get('license_copyright', 0) + 1
                    continue

                stats['layer2_pass'] += 1

                # ============ Layer 3: 内容质量 ============
                # 3a. Token overlap
                overlap = check_token_overlap(clean_doc, content, lang)
                if overlap > 1:
                    reject_reasons['bad_overlap'] = reject_reasons.get('bad_overlap', 0) + 1
                    continue

                # 3b. 信息量检查
                informative, info_reason = check_informative_content(clean_doc)
                if not informative:
                    reject_reasons[f'info_{info_reason}'] = reject_reasons.get(f'info_{info_reason}', 0) + 1
                    continue

                # 3c. 语法检查
                grammar_ok, error_count = check_grammar_quality(clean_doc)
                if not grammar_ok:
                    reject_reasons['grammar_errors'] = reject_reasons.get('grammar_errors', 0) + 1
                    continue

                stats['layer3_pass'] += 1

                # ============ Layer 4: 语义质量 ============
                # 提取纯代码
                code_only = extract_code_only(content, lang)
                if not code_only:
                    reject_reasons['code_extraction_failed'] = reject_reasons.get('code_extraction_failed', 0) + 1
                    continue

                # 计算语义相似度
                try:
                    similarity = compute_semantic_similarity(code_only, clean_doc)

                    if not (0.4 < similarity < 0.9):
                        reject_reasons['bad_similarity'] = reject_reasons.get('bad_similarity', 0) + 1
                        continue
                except Exception as e:
                    reject_reasons['similarity_error'] = reject_reasons.get('similarity_error', 0) + 1
                    continue

                stats['layer4_pass'] += 1

                # ============ 通过所有层，加入候选 ============
                candidates.append({
                    'index': idx,
                    'summary': clean_doc,
                    'code': code_only,
                    'overlap': overlap,
                    'similarity': similarity,
                    'length': len(clean_doc.split())
                })

            except Exception as e:
                reject_reasons['processing_error'] = reject_reasons.get('processing_error', 0) + 1
                continue

        # 保存Layer 1-4的候选结果
        intermediate_path = os.path.join(config['intermediate_dir'], f"{lang}_candidates.json")
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)

        # 打印统计
        print(f"\n📊 [{lang}] Layer 1-4 过滤统计:")
        print(f"  原始样本: {stats['total']}")
        print(f"  Layer 1 (结构完整性): {stats['layer1_pass']} ({stats['layer1_pass'] / stats['total'] * 100:.1f}%)")
        print(f"  Layer 2 (基础质量): {stats['layer2_pass']} ({stats['layer2_pass'] / stats['total'] * 100:.1f}%)")
        print(f"  Layer 3 (内容质量): {stats['layer3_pass']} ({stats['layer3_pass'] / stats['total'] * 100:.1f}%)")
        print(f"  Layer 4 (语义质量): {stats['layer4_pass']} ({stats['layer4_pass'] / stats['total'] * 100:.1f}%)")

        print(f"\n❌ 拒绝原因分布:")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {reason}: {count}")


# ==========================================
#       Step 3: 去重与最终采样
# ==========================================

def step_3_dedup_and_sample(config):
    """
    步骤三：Layer 5 - 去重与分层采样
    """
    print("\n>>> [步骤三] Layer 5: 去重与最终采样 <<<")

    os.makedirs(config['final_output_dir'], exist_ok=True)

    for lang in config['langs']:
        intermediate_path = os.path.join(config['intermediate_dir'], f"{lang}_candidates.json")

        if not os.path.exists(intermediate_path):
            print(f"⚠️ [跳过] 找不到候选文件: {intermediate_path}")
            continue

        with open(intermediate_path, 'r', encoding='utf-8') as f:
            candidates = json.load(f)

        print(f"\n🔵 [{lang}] Layer 5 处理: {len(candidates)} 个候选")

        # ============ 5.1 完全重复去重 ============
        seen_summaries = set()
        unique_candidates = []

        for cand in candidates:
            if cand['summary'] not in seen_summaries:
                seen_summaries.add(cand['summary'])
                unique_candidates.append(cand)

        print(f"  完全重复去重: {len(candidates)} -> {len(unique_candidates)}")

        # ============ 5.2 近似重复去重（简单版：编辑距离）============
        from difflib import SequenceMatcher

        def is_similar(s1, s2, threshold=0.85):
            return SequenceMatcher(None, s1, s2).ratio() > threshold

        final_candidates = []
        for cand in unique_candidates:
            is_dup = False
            for existing in final_candidates:
                if is_similar(cand['summary'], existing['summary']):
                    is_dup = True
                    break
            if not is_dup:
                final_candidates.append(cand)

        print(f"  近似重复去重: {len(unique_candidates)} -> {len(final_candidates)}")

        # ============ 5.3 分层采样（按长度和复杂度）============
        # 按summary长度分层
        short = [c for c in final_candidates if 3 <= c['length'] <= 15]
        medium = [c for c in final_candidates if 15 < c['length'] <= 30]
        long_sum = [c for c in final_candidates if 30 < c['length'] <= 50]

        print(f"  长度分布: 短={len(short)}, 中={len(medium)}, 长={len(long_sum)}")

        # 采样目标（最终保留的样本数）
        target_size = min(len(final_candidates), config.get('target_size_per_lang', 3000))

        # 按比例采样（短:中:长 = 4:4:2）
        import random
        random.seed(42)

        n_short  = min(len(short),    int(target_size * 0.4))
        n_medium = min(len(medium),   int(target_size * 0.4))
        n_long   = min(len(long_sum), int(target_size * 0.2))

        sampled = []
        if n_short  > 0: sampled.extend(random.sample(short,    n_short))
        if n_medium > 0: sampled.extend(random.sample(medium,   n_medium))
        if n_long   > 0: sampled.extend(random.sample(long_sum, n_long))

        # 如果还不够，从剩余的随机补充
        if len(sampled) < target_size:
            remaining = [c for c in final_candidates if c not in sampled]
            additional = min(len(remaining), target_size - len(sampled))
            sampled.extend(random.sample(remaining, additional))

        print(f"  最终采样: {len(sampled)} 个")

        # ============ 保存最终结果（TSV格式）============
        output_path = os.path.join(config['final_output_dir'], f"{lang}_final.tsv")

        df_result = pd.DataFrame([
            {'index': c['index'], 'summary': c['summary']}
            for c in sampled
        ])

        df_result.to_csv(
            output_path,
            sep='\t',
            index=False,
            encoding='utf-8-sig',
            escapechar='\\'
        )

        print(f"✅ [{lang}] 最终结果保存: {output_path}")
        print(f"  样本数: {len(df_result)}")
        print(f"  平均长度: {sum(c['length'] for c in sampled) / len(sampled):.1f} 词")


# ==========================================
#              MAIN 主函数
# ==========================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MultiPL-T low-resource cleaning pipeline (5-layer filter)."
    )
    parser.add_argument("--base_dir", default="./data/LowData",
                        help="Root containing per-language MultiPL-T parquet files (default: ./data/LowData)")
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'r', 'racket'])
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                        help="Top-fraction of each parquet kept in Step 1 (paper §4.1 uses 0.1)")
    parser.add_argument("--raw_jsonl_dir",    default="./data/LowData/step1_raw_jsonl")
    parser.add_argument("--intermediate_dir", default="./data/LowData/step2_intermediate")
    parser.add_argument("--final_output_dir", default="./data/LowData/step3_final_result")
    parser.add_argument("--target_size_per_lang", type=int, default=3000)
    parser.add_argument("--skip_step1", action="store_true",
                        help="Skip the parquet → jsonl sampling step")
    parser.add_argument("--skip_step2", action="store_true",
                        help="Skip Layer 1-4 filtering")
    parser.add_argument("--skip_step3", action="store_true",
                        help="Skip Layer 5 dedup + final sampling")
    args = parser.parse_args()

    config = {
        "base_dir":              args.base_dir,
        "langs":                 args.langs,
        "sample_ratio":          args.sample_ratio,
        "raw_jsonl_dir":         args.raw_jsonl_dir,
        "intermediate_dir":      args.intermediate_dir,
        "final_output_dir":      args.final_output_dir,
        "target_size_per_lang":  args.target_size_per_lang,
    }

    print("\n" + "=" * 60)
    print("  MultiPL-T data cleaning — 5-layer filter pipeline")
    print("=" * 60)

    if not args.skip_step1: step_1_sample_raw_data(config)
    if not args.skip_step2: step_2_multilayer_filter(config)
    if not args.skip_step3: step_3_dedup_and_sample(config)

    print("\n" + "=" * 60)
    print("✅ All steps complete.")
    print("=" * 60)
    print(f"\nFinal output: {config['final_output_dir']}/")
    print("Format: <lang>_final.tsv  (columns: index, summary)")


if __name__ == "__main__":
    main()
