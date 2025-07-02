using TreeSitter
using TreeSitter.Julia
using JSON

function parse_julia_code(code::String)
    lang = TreeSitter.Julia.get_language()
    parser = Parser()
    set_language!(parser, lang)
    tree = parse(parser, code)
    return tree
end

function node_to_dict(node, code)
    children = [node_to_dict(child, code) for child in node.children]
    Dict(
        "type" => string(node.type),
        "start_byte" => node.start,
        "end_byte" => node.stop,
        "text" => code[node.start+1:node.stop],
        "children" => children
    )
end

function main()
    code = read(stdin, String)
    tree = parse_julia_code(code)
    root_node = root(tree)
    ast_dict = node_to_dict(root_node, code)
    println(JSON.json(ast_dict))
end

main()
