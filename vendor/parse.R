# parse.R
# 加载必要的库
library(treesitter)

# 确保 treesitter.r 包（R语言的 tree-sitter 文法）已安装并可用
if (!requireNamespace("treesitter.r", quietly = TRUE)) {
  stop("Package 'treesitter.r' is not installed. Please install it to parse R code with R-specific grammar. You can typically install it using: devtools::install_github('r-lib/treesitter.r') or the relevant method if it's on CRAN.")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Please provide R code as a command line argument.", call. = FALSE)
}
r_code_to_parse <- args[1]

r_language <- NULL
if (exists("language", envir = asNamespace("treesitter.r"))) {
    r_language <- treesitter.r::language()
} else {
    stop("Could not retrieve the R language grammar from the 'treesitter.r' package. Ensure it is installed and loaded correctly.")
}

parser <- parser(r_language)
tree <- parser_parse(parser, r_code_to_parse)
root_node <- tree_root_node(tree)

STATEMENT_SEPARATOR <- " R_STATEMENT_SEPARATOR_MAGIC_STRING_0123456789 "

child_count <- node_child_count(root_node)
output_count <- 0

if (child_count > 0) {
  for (i in 1:child_count) {
    child <- node_child(root_node, i)

    # 修正节点检查逻辑:
    # 1. 首先检查 child 对象本身是否为 R 的 NULL (虽然 node_child 通常会返回一个节点对象或错误)
    # 2. 使用 node_is_named() 来判断是否为具名节点 (通常我们关心的是具名语句节点)
    # 3. 使用 node_is_missing() 检查节点是否是语法中预期但源码中缺失的
    # 4. 检查 node_type() 是否为 "comment" 或 "ERROR"

    # 一个有效的、我们想要提取的节点，通常是:
    # - 不是 R 的 NULL
    # - 是一个具名节点 (node_is_named(child) == TRUE)
    # - 不是缺失节点 (node_is_missing(child) == FALSE)
    # - 类型不是 "comment" 或 "ERROR"
    # - 文本内容在去除空白后不为空

    # is.null() 检查 R 对象是否为 NULL
    # node_is_named() 检查是否为具名节点 (匿名节点如 '(' 通常不是我们想要的独立语句)
    # node_is_missing() 检查节点是否是语法层面期望但源码中缺失的
    if (!is.null(child) &&        # 确保 child 不是 R 的 NULL
        node_is_named(child) &&   # 通常我们关心的是具名节点
        !node_is_missing(child) && # 确保节点不是缺失的
        node_type(child) != "comment" &&
        node_type(child) != "ERROR" &&
        nzchar(trimws(node_text(child)))) {

      if (output_count > 0) {
          cat(STATEMENT_SEPARATOR)
      }
      cat(node_text(child))
      output_count <- output_count + 1
    }
  }
} else if (!is.null(root_node) &&
           !node_is_missing(root_node) && # 也为根节点添加此检查
           node_type(root_node) == "program" &&
           nzchar(trimws(node_text(root_node)))) {
   # 此分支逻辑基本不变，但触发条件更严格
}

if (output_count > 0) {
    cat("\n")
}