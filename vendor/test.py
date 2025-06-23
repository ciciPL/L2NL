import tree_sitter_julia
import tree_sitter_lua
import tree_sitter_ocaml
import tree_sitter_python
from tree_sitter import Language, Parser

LANG_python = Language(tree_sitter_python.language())

Parser_python = Parser(LANG_python)

code = b"""def add(a,b):\treturn a+b\n"""
tree_python = Parser_python.parse(code)
print(tree_python.root_node)

language_ocaml = Language(tree_sitter_ocaml.language_ocaml())
parser = Parser(language_ocaml)
tree = parser.parse(
    b"""
    module M : sig
      val x : int
    end
    """
)

print(tree.root_node)
language_julia = Language(tree_sitter_julia.language())
parser = Parser(language_julia)
tree_julia = parser.parse(
    b"""
    function job_tasks(conf)     
    return conf["tasks"] 
    end
    """
)
print(tree_julia.root_node)

language_lua = Language(tree_sitter_lua.language())
parser = Parser(language_lua)
tree_lua = parser.parse(
    b"""
local function transform(text)     local result = ""     for char in text:gmatch(".") do         if char == ' ' then             result = result .. ' '         else             result = result .. char:upper()         end     end     return result end
    """
)
print(tree_lua.root_node)