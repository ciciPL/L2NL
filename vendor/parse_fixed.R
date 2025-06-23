library(treesitter)
library(treesitter.r)  # 显式加载

# 检查 treesitter.r 是否可用
if (!requireNamespace("treesitter.r", quietly = TRUE)) {
  stop("需要安装 treesitter.r：devtools::install_github('r-lib/treesitter.r')")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("请传入 R 代码作为参数")
r_code <- args[1]

# 初始化 parser
r_lang <- treesitter.r::language()
parser <- parser(r_lang)
tree <- parser_parse(parser, r_code)
root <- tree_root_node(tree)

# 定义需要提取的「语句节点类型」
STATEMENT_TYPES <- c(
  "expression_statement", "if_statement", "for_statement",
  "while_statement", "function_definition", "assignment_expression",
  "return_statement", "break_statement", "next_statement"
)
SEPARATOR <- " R_STATEMENT_SEPARATOR_MAGIC_STRING_0123456789 "

# 递归遍历语法树，提取所有符合条件的语句节点
extract_statements <- function(node) {
  statements <- character(0)  # 初始化空字符向量

  # 跳过 NULL 节点
  if (is.null(node)) return(statements)

  # 当前节点是否是「语句节点」
  if (node_type(node) %in% STATEMENT_TYPES &&
      node_is_named(node) &&
      !node_is_missing(node) &&
      nzchar(trimws(node_text(node)))) {
    statements <- c(statements, node_text(node))
  }

  # 获取子节点数量并检查是否大于 0
  child_count <- node_child_count(node)
  if (child_count > 0) {
    # 遍历子节点（注意索引从 1 开始）
    for (i in 1:child_count) {
      child <- node_child(node, i)
      # 递归提取子节点中的语句
      statements <- c(statements, extract_statements(child))
    }
  }

  return(statements)
}

# 执行递归提取
all_statements <- extract_statements(root)

# 输出结果（用分隔符拼接）
if (length(all_statements) > 0) {
  cat(paste(all_statements, collapse = SEPARATOR))
} else {
  # 如果没有提取到语句，输出空字符串
  cat("")
}