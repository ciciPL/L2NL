import json
import os
import subprocess
import tempfile

# --- Lua ---
try:
    from luaparser import ast as lua_ast
    from luaparser.parser import Parser as LuaParser

    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False
    # print("Warning: luaparser library not found. Lua parsing will not be available.")

# --- R (via rpy2) ---
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.rinterface import RRuntimeError

    R_AVAILABLE = True
    # Define R's deparse function for converting R objects back to string
    r_deparse = robjects.r['deparse']
    r_parse = robjects.r['parse']
except ImportError:
    R_AVAILABLE = False
    # print("Warning: rpy2 library not found or R not configured. R parsing will not be available.")
except Exception as e:  # Catches errors if R itself is not installed/found by rpy2
    R_AVAILABLE = False
    # print(f"Warning: rpy2 available but R environment issue: {e}. R parsing will not be available.")

# --- Julia (via juliacall) ---
try:
    from juliacall import Main as jlmain

    # jlmain.seval("using Meta") # Meta is usually available by default for parse
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False
    # print("Warning: juliacall library not found. Julia parsing will not be available.")
except Exception as e:  # Catches errors if Julia itself is not installed/found
    JULIA_AVAILABLE = False
    # print(f"Warning: juliacall available but Julia environment issue: {e}. Julia parsing will not be available.")


class LanguageASTParser:
    def __init__(self):
        """
        Initializes the parser utility.
        Checks for availability of required libraries/tools.
        """
        if not LUAPARSER_AVAILABLE:
            print("Info: Lua parsing functionality will be limited as 'luaparser' is not installed.")
        if not R_AVAILABLE:
            print("Info: R parsing functionality will be limited as 'rpy2' is not installed or R is not configured.")
        if not JULIA_AVAILABLE:
            print(
                "Info: Julia parsing functionality will be limited as 'juliacall' is not installed or Julia is not configured.")
        # Racket and OCaml checks would depend on finding their command-line tools
        # For example, check if 'racket' or 'ocamlc' are in PATH.
        # This is omitted for brevity here but important for robust implementation.

    def _node_to_lua_string(self, node, original_code):
        """
        Helper to convert a Lua AST node to its string representation.
        This is a simplified version; luaparser itself might not have a perfect
        pretty-printer for arbitrary sub-nodes that exactly matches original formatting.
        A more robust way is to use line/column information if available from the AST node
        to extract the original substring.
        """
        try:
            # Attempt to use luaparser's to_lua_source if applicable for the node type
            # This typically works on the whole tree or major structures.
            # For individual statements, it might be more complex.
            # As a fallback, we can get line numbers.
            if hasattr(node, 'first_line') and hasattr(node, 'last_line'):
                lines = original_code.splitlines()
                start_line = node.first_line - 1
                end_line = node.last_line
                # This might include too much if statements are on the same line originally
                # or if indentation/leading/trailing parts are complex.
                # This is a common challenge in AST-to-source for sub-elements.
                return "\n".join(lines[start_line:end_line]).strip()
            else:
                # Fallback if line info is not directly on the node or to_lua_source is not ideal
                # This will require a more sophisticated way to reconstruct source from node type and attributes
                return f"[[Lua AST Node: {type(node).__name__} - content unavailable for simple extraction]]"

        except Exception:
            return f"[[Error converting Lua node {type(node).__name__} to string]]"

    def parse_lua(self, code_string):
        """
        Parses Lua code into statements.
        Returns: (list_of_statement_strings, success_boolean)
        """
        if not LUAPARSER_AVAILABLE:
            return ["Lua parser (luaparser) not available."], False

        statements = []
        try:
            parser = LuaParser()
            tree = parser.parse(code_string)

            if tree and hasattr(tree, 'body') and hasattr(tree.body, 'body'):  # tree.body is a Block node
                for stmt_node in tree.body.body:
                    # Convert each statement node back to string
                    # Using a simple representation here. A full pretty-printer or
                    # source-code slicer based on node extents would be more accurate.
                    # For now, let's try a simple reconstruction or use line numbers.
                    stmt_str = lua_ast.to_lua_source(stmt_node)  # Try this first
                    statements.append(stmt_str.strip())
                return statements, True
            else:
                return ["Failed to parse Lua code into a valid AST structure."], False
        except Exception as e:
            return [f"Lua parsing error: {str(e)}"], False

    def parse_r(self, code_string):
        """
        Parses R code into statements using rpy2.
        Returns: (list_of_statement_strings, success_boolean)
        """
        if not R_AVAILABLE:
            return ["R parser (rpy2 or R environment) not available."], False

        statements = []
        try:
            # Parse the code string. This can result in multiple expressions.
            parsed_exprs = r_parse(text=code_string)
            for i in range(len(parsed_exprs)):
                expr = parsed_exprs.ro_getItem(i + 1)  # R vectors are 1-indexed from rpy2
                # Deparse the R expression object back to a string
                stmt_str_vector = r_deparse(expr)
                statements.append("\n".join(stmt_str_vector))
            return statements, True
        except RRuntimeError as e:  # Specific rpy2 error for R issues
            return [f"R parsing error (RRuntimeError): {str(e)}"], False
        except Exception as e:
            return [f"R parsing error: {str(e)}"], False

    def parse_julia(self, code_string):
        """
        Parses Julia code into statements using juliacall.
        Returns: (list_of_statement_strings, success_boolean)
        """
        if not JULIA_AVAILABLE:
            return ["Julia parser (juliacall or Julia environment) not available."], False

        statements = []
        try:
            # Meta.parseall can parse a string containing multiple expressions
            # It returns a :toplevel expression if multiple, or a single expression.
            # We might need to handle this structure.
            # For simplicity, let's parse one by one if Meta.parse is preferred,
            # or handle the result of Meta.parseall.
            # jlmain.eval(f"using Meta") # Ensure Meta is loaded

            # Using Meta.parseall for robustness with multiple top-level expressions
            # The input string to Julia needs to be properly quoted/escaped.
            # Using repr() for Python strings is a good heuristic for simple cases.
            julia_code_str_literal = repr(code_string)
            parsed_obj = jlmain.seval(f"Meta.parseall({julia_code_str_literal})")

            # The result could be a single Expr or an Expr with head :toplevel
            # whose args are the actual statements.
            if hasattr(parsed_obj, 'head') and str(parsed_obj.head) == ':toplevel':
                # If it's a :toplevel block, its arguments are the statements
                expr_list = parsed_obj.args
            elif hasattr(parsed_obj, 'head'):  # Single expression
                expr_list = [parsed_obj]
            else:  # Should not happen with parseall if code is valid
                expr_list = []

            for expr in expr_list:
                if expr is None: continue  # Julia can have `nothing` as a placeholder
                # Convert Julia Expr object back to string
                # str(expr) via juliacall usually gives a good representation.
                statements.append(str(expr))

            # Check if parsing actually produced anything for non-empty code
            if not statements and code_string.strip():
                # This might indicate an issue or a piece of code that parses to nothing (e.g. comments only)
                # If Meta.parseall itself errors, it would raise an exception caught below.
                # If it returns an empty :toplevel or a single "nothing" like expression, this check is useful.
                # For now, we assume if no exception and list is empty, it's valid (e.g. comments-only code)
                pass

            return statements, True
        except Exception as e:  # juliacall typically wraps Julia errors into Python exceptions
            return [f"Julia parsing error: {str(e)}"], False

    def _run_external_parser(self, command_parts, code_string, temp_suffix):
        """Helper to run an external command-line parser."""
        statements = []
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=temp_suffix, encoding='utf-8') as tmp_in:
                tmp_in.write(code_string)
                input_filepath = tmp_in.name

            full_command = command_parts + [input_filepath]

            process = subprocess.run(full_command, capture_output=True, text=True, encoding='utf-8', timeout=10)

            os.remove(input_filepath)  # Clean up temp file

            if process.returncode == 0:
                # Assuming the external tool prints one statement per line
                # Or outputs JSON that needs to be parsed.
                # This part needs to be adapted to the specific tool's output format.
                output_lines = process.stdout.strip().splitlines()
                # Example: if tool outputs JSON list of strings
                # try:
                # statements = json.loads(process.stdout)
                # success = True
                # except json.JSONDecodeError:
                # return [f"Failed to decode JSON output from external parser: {process.stdout}"], False

                # For now, assume one statement per line in stdout
                statements = [line for line in output_lines if line.strip()]
                return statements, True
            else:
                error_msg = f"External parser error (return code {process.returncode}):\n"
                error_msg += f"STDERR: {process.stderr.strip()}\n"
                error_msg += f"STDOUT: {process.stdout.strip()}"
                return [error_msg], False

        except FileNotFoundError:  # If the command itself (e.g. 'racket') is not found
            return [f"External parser command '{command_parts[0]}' not found. Is it installed and in PATH?"], False
        except subprocess.TimeoutExpired:
            return ["External parser timed out."], False
        except Exception as e:
            if 'input_filepath' in locals() and os.path.exists(input_filepath):
                os.remove(input_filepath)
            return [f"Error running external parser: {str(e)}"], False
        finally:
            if 'input_filepath' in locals() and os.path.exists(input_filepath):
                try:
                    os.remove(input_filepath)
                except OSError:
                    pass  # Ignore if already removed or other issues

    def parse_racket(self, code_string):
        """
        Parses Racket code into statements (top-level S-expressions).
        This is a conceptual implementation relying on an external Racket script.
        You'll need a Racket script (e.g., 'parse_racket_helper.rkt') that
        reads code, processes it, and prints each top-level form as a string.
        """
        # TODO: Create a 'parse_racket_helper.rkt' script.
        # Example parse_racket_helper.rkt:
        # #lang racket/base
        # (require racket/pretty json)
        # (define (main code-str)
        #   (with-input-from-string code-str
        #     (lambda ()
        #       (let loop ([forms '()])
        #         (let ([form (read-syntax #f (current-input-port))])
        #           (if (eof-object? form)
        #               (reverse forms) ; Output as a JSON list of strings
        #               (loop (cons (syntax->string form) forms)))))))
        # ; Read from command line argument (path to file containing code)
        # (let ([args (current-command-line-arguments)])
        #  (if (vector-empty? args)
        #      (eprintf "Usage: racket parse_racket_helper.rkt <filepath>\n")
        #      (let* ([filepath (vector-ref args 0)]
        #             [code (file->string filepath)])
        #        (displayln (jsexpr->string (main code))))))

        # The command would be ['racket', 'path/to/parse_racket_helper.rkt']
        # The helper script should take a filepath as input, read it, parse,
        # and print a JSON list of strings (each string being a top-level form).

        # For this placeholder, we assume such a script exists and outputs JSON list of strings
        helper_script_path = "parse_racket_helper.rkt"  # Needs to be in PATH or provide full path
        if not os.path.exists(helper_script_path):
            return [f"Racket helper script '{helper_script_path}' not found. "
                    "Racket parsing requires an external Racket script. See comments in code."], False

        command = ['racket', helper_script_path]
        statements, success = self._run_external_parser(command, code_string, ".rkt")
        if success:
            try:
                # If the Racket script outputs a single JSON string which is a list of strings
                if len(statements) == 1:
                    parsed_statements = json.loads(statements[0])
                    if isinstance(parsed_statements, list) and all(isinstance(s, str) for s in parsed_statements):
                        return parsed_statements, True
                return ["Failed to parse JSON output from Racket script correctly."], False
            except json.JSONDecodeError as e:
                return [f"Racket script output was not valid JSON: {e}\nOutput: {' '.join(statements)}"], False
        else:
            return statements, False  # statements list here contains the error message from _run_external_parser

    def parse_ocaml(self, code_string):
        """
        Parses OCaml code into statements (top-level phrases).
        Conceptual: Relies on an OCaml tool or custom OCaml script.
        Example: A script using OCaml's compiler-libs to parse and pretty-print phrases.
        """
        # TODO: Create an OCaml helper script or use ocamlc options if suitable.
        # An OCaml script might look like this (highly simplified concept):
        # (* parse_ocaml_helper.ml *)
        # open Lexing
        # open Parsetree
        # open Pprintast
        #
        # let () =
        #   let filepath = Sys.argv.(1) in
        #   let ic = open_in filepath in
        #   let lb = from_channel ic in
        #   try
        #     let phrases = Parse.interface lb (* or Parse.implementation for .ml *) in
        #     (* Convert each phrase to string; Pprintast.signature_item for .mli or structure_item for .ml *)
        #     List.iter (fun phrase ->
        #       Format.printf "%s\n" (Format.asprintf "%a" signature_item phrase) (* Adjust for .ml *)
        #     ) phrases;
        #     close_in ic
        #   with
        #   | End_of_file -> close_in_noerr ic
        #   | exn ->
        #     Location.report_exception Format.err_formatter exn;
        #     close_in_noerr ic;
        #     exit 1
        #
        # To compile: ocamlc -I +compiler-libs ocamlcommon.cma ocamlbytecomp.cma ocamllex.cma parse_ocaml_helper.ml -o parse_ocaml_helper
        # Command: ['./parse_ocaml_helper']

        # This placeholder assumes a compiled OCaml helper 'parse_ocaml_helper'
        # that takes a filepath, and prints each top-level phrase on a new line.
        helper_executable_path = "./parse_ocaml_helper"  # Needs to exist and be executable
        if not os.path.exists(helper_executable_path) or not os.access(helper_executable_path, os.X_OK):
            return [f"OCaml helper executable '{helper_executable_path}' not found or not executable. "
                    "OCaml parsing requires a compiled OCaml helper. See comments in code."], False

        command = [helper_executable_path]
        return self._run_external_parser(command, code_string, ".ml")  # or .mli


if __name__ == '__main__':
    parser = LanguageASTParser()

    print("\n--- Testing Lua ---")
    lua_code = """
    function greet(name)
        local message = "Hello, " .. name .. "!"
        print(message)
        return message
    end
    local x = 10; local y = 20
    greet("Lua User")
    """
    statements, success = parser.parse_lua(lua_code)
    print(f"Success: {success}")
    if success:
        for i, stmt in enumerate(statements):
            print(f"  Stmt {i + 1}: {stmt}")
    else:
        print(f"  Error: {statements[0]}")

    print("\n--- Testing R ---")
    r_code = """
    greeting <- function(name) {
        message <- paste("Hello,", name, "!")
        print(message)
        return(message)
    }
    x_val = 100 * 2; y_val = x_val / 5
    greeting("R User")
    """
    statements, success = parser.parse_r(r_code)
    print(f"Success: {success}")
    if success:
        for i, stmt in enumerate(statements):
            print(f"  Stmt {i + 1}: {stmt}")
    else:
        print(f"  Error: {statements[0]}")

    print("\n--- Testing Julia ---")
    julia_code = """
    function say_hello(person_name)
        msg = "Hello, " * person_name * "!"
        println(msg)
        return msg
    end
    a = 1; b = 2; c = a + b
    say_hello("Julia User")
    # This is a comment
    """
    statements, success = parser.parse_julia(julia_code)
    print(f"Success: {success}")
    if success:
        for i, stmt in enumerate(statements):
            print(f"  Stmt {i + 1}: {stmt}")
    else:
        print(f"  Error: {statements[0]}")

    print("\n--- Testing Racket (Conceptual - Requires external script) ---")
    racket_code = """
    #lang racket
    (define (welcome name)
      (string-append "Welcome, " name "!"))
    (define x 30)
    (welcome "Racket User")
    """
    # You would need to create 'parse_racket_helper.rkt' as described in the method.
    # For this example, it will likely fail if the script is not present.
    if os.path.exists("parse_racket_helper.rkt"):  # Simple check
        statements, success = parser.parse_racket(racket_code)
        print(f"Success: {success}")
        if success:
            for i, stmt in enumerate(statements):
                print(f"  Stmt {i + 1}: {stmt}")
        else:
            print(f"  Error: {statements[0]}")
    else:
        print("  Skipping Racket test: parse_racket_helper.rkt not found.")

    print("\n--- Testing OCaml (Conceptual - Requires external executable) ---")
    ocaml_code = """
    let greet subject =
      let msg = "Hello, " ^ subject ^ "!" in
      print_endline msg;
      msg
    ;;
    let num = 42 + 1;;
    greet "OCaml User";;
    """
    # You would need to create and compile 'parse_ocaml_helper.ml' as described.
    # For this example, it will likely fail if the executable is not present.
    if os.path.exists("./parse_ocaml_helper") and os.access("./parse_ocaml_helper", os.X_OK):
        statements, success = parser.parse_ocaml(ocaml_code)
        print(f"Success: {success}")
        if success:
            for i, stmt in enumerate(statements):
                print(f"  Stmt {i + 1}: {stmt}")
        else:
            print(f"  Error: {statements[0]}")
    else:
        print("  Skipping OCaml test: ./parse_ocaml_helper not found or not executable.")