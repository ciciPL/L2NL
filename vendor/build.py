from tree_sitter import Language

Language.build_library(
    'build/my_language.so',
    [
        'tree-sitter-python',
        'tree-sitter-ocaml',
        'tree-sitter-julia',
        'tree-sitter-lua',
    ]
)
