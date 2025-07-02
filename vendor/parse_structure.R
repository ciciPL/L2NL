library(treesitter)
library(treesitter.r)
library(jsonlite)

node_text <- function(node, code) {
  if (is.null(node)) return("")
  substr(code, node_start_byte(node) + 1, node_end_byte(node))
}

merge_results <- function(base, new) {
  list(
    function_def = c(base$function_def, new$function_def),
    loops = c(base$loops, new$loops),
    conditionals = c(base$conditionals, new$conditionals),
    assignments = c(base$assignments, new$assignments),
    others = c(base$others, new$others)
  )
}

process_function_body <- function(body_node, code) {
  result <- list(
    loops = character(0),
    conditionals = character(0),
    assignments = character(0),
    others = character(0)
  )

  if (is.null(body_node)) {
    message("[process_function_body] body_node is NULL")
    return(result)
  }

  message("[process_function_body] body_node type: ", node_type(body_node))
  message("[process_function_body] body_node child count: ", node_child_count(body_node))

  children <- Filter(node_is_named, lapply(seq_len(node_child_count(body_node)), function(i) node_child(body_node, i)))

  message("[process_function_body] filtered named children count: ", length(children))

  for (child in children) {
    type <- node_type(child)
    message("[process_function_body] child node type: ", type)
    message("[process_function_body] child text: ", node_text(child, code))

    if (type %in% c("body", "block", "braced_expression")) {
      nested <- process_function_body(child, code)
      result <- merge_results(result, nested)
    } else if (type %in% c("for_statement", "while_statement", "repeat_statement")) {
      result$loops <- c(result$loops, node_text(child, code))
    } else if (type == "if_statement") {
      result$conditionals <- c(result$conditionals, node_text(child, code))
    } else if (type == "binary_operator") {
      result$assignments <- c(result$assignments, node_text(child, code))
    } else {
      result$others <- c(result$others, node_text(child, code))
    }
  }

  return(result)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("No R code provided")
r_code <- args[1]

r_lang <- treesitter.r::language()
parser <- parser(r_lang)
tree <- parser_parse(parser, r_code)
root <- tree_root_node(tree)

if (is.null(root) || node_type(root) == "ERROR") stop("Parse error")

message("[Main] Root node type: ", node_type(root))
message("[Main] Children count: ", node_child_count(root))

final_result <- list(
  function_def = character(0),
  loops = character(0),
  conditionals = character(0),
  assignments = character(0),
  others = character(0)
)

for (i in seq_len(node_child_count(root))) {
  stmt <- node_child(root, i)
  if (is.null(stmt) || !node_is_named(stmt)) next

  message("[Main] Child ", i, " node type: ", node_type(stmt))
  message("[Main] Child ", i, " code: ", node_text(stmt, r_code))

  is_func <- FALSE

  if (node_type(stmt) == "binary_operator" && node_child_count(stmt) >= 2) {
    # 打印所有子节点类型和文本，确认结构
    for (idx in 1:node_child_count(stmt)) {
      child <- node_child(stmt, idx)
      message(sprintf("[Main] stmt child %d type: %s text: %s", idx, node_type(child), node_text(child, r_code)))
    }

    # 找出右侧的函数定义节点
    rhs <- NULL
    for (idx in 1:node_child_count(stmt)) {
      child <- node_child(stmt, idx)
      if (node_type(child) == "function_definition") {
        rhs <- child
        break
      }
    }

    if (is.null(rhs)) {
      message("[Main] no function_definition found among stmt children")
    } else {
      message("[Main] rhs node type: ", node_type(rhs))
      message("[Main] rhs text: ", node_text(rhs, r_code))
      is_func <- TRUE

      lhs <- node_child(stmt, 1)

      params <- NULL
      func_body <- NULL
      for (j in 1:node_child_count(rhs)) {
        child <- node_child(rhs, j)
        ctype <- node_type(child)
        message(sprintf("[Main] function_definition child %d type: %s text: %s", j, ctype, node_text(child, r_code)))
        if (ctype == "parameters") {
          params <- child
        } else if (ctype %in% c("body", "block", "braced_expression")) {
          func_body <- child
        }
      }

      message("[Main] Found function: ", node_text(lhs, r_code))
      message("[Main] Parameters: ", node_text(params, r_code))
      message("[Main] func_body node type: ", ifelse(is.null(func_body), "NULL", node_type(func_body)))
      if (!is.null(func_body)) {
        message("[Main] func_body child count: ", node_child_count(func_body))
        for (k in 1:node_child_count(func_body)) {
          fb_child <- node_child(func_body, k)
          message("[Main] func_body child ", k, " type: ", node_type(fb_child))
          message("[Main] func_body child ", k, " text: ", node_text(fb_child, r_code))
        }
        message("[Main] func_body text: ", node_text(func_body, r_code))
      }

      final_result$function_def <- c(final_result$function_def, paste(node_text(lhs, r_code), "<- function", node_text(params, r_code), "{}"))

      body_parts <- process_function_body(func_body, r_code)
      final_result <- merge_results(final_result, body_parts)
    }
  }

  if (!is_func) {
    final_result$others <- c(final_result$others, node_text(stmt, r_code))
  }
}

cat(toJSON(final_result, pretty = TRUE))
