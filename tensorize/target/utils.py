import ast

def get_last_function(py_str):
    tree = ast.parse(py_str)
    last_func = tree.body[-1]
    return ast.unparse(last_func)


def replace_return_value(py_str):
    tree = ast.parse(py_str)
    last_func = tree.body[-1]

    # Get the value of the last function call
    assert isinstance(last_func.body[-1], ast.Return)

    last_call = None
    for stmt in last_func.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            last_call = stmt.value
    assert last_call

    # Remove the return statement
    last_func.body = last_func.body[:-2]

    # Add a return statement returning the value of the last function call
    last_func.body.append(ast.Return(value=last_call))

    return ast.unparse(last_func)


def get_fn(fn_str, fn_name):
    fn_ast = ast.parse(fn_str)
    fn_body = fn_ast.body
    for stmt in fn_body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == fn_name:
            return stmt
    return None


def get_stmts_without_return(fn):
    return [stmt for stmt in fn.body if not isinstance(stmt, ast.Return)]


def inline_ast_functions(prog_str):
    prog_ast = ast.parse(prog_str)
    fn = prog_ast.body[-1]

    for stmt in fn.body:
        if isinstance(stmt.value, ast.Call):
            # Get the called function
            called_fn_name = stmt.value.func.id
            called_fn = get_fn(prog_str, called_fn_name)

            # Map the arguments of the called function to the arguments of the caller
            mapping = {}
            for arg_idx, arg in enumerate(called_fn.args.args):
                mapping[arg.arg] = stmt.value.args[arg_idx].id

            # Build new body
            new_body = []
            for stmt_idx, called_fn_stmt in enumerate(called_fn.body):
                if isinstance(called_fn_stmt, ast.Assign) and type(
                    called_fn_stmt.value
                ) in [ast.Call, ast.Name, ast.Constant]:
                    # Rename the arguments of the called function with the names of arguments of the caller
                    # LHS
                    if stmt_idx < len(get_stmts_without_return(called_fn)) - 1:
                        # Here we're at statements before the last statement of the function
                        var_id = called_fn_name + "_var" + str(stmt_idx)
                        old_var_id = called_fn_stmt.targets[0].id
                        called_fn_stmt.targets[0].id = var_id
                        mapping[old_var_id] = var_id
                    else:
                        # Here we're at the last statement of the function
                        if isinstance(stmt, ast.Return):
                            called_fn_stmt = ast.Return(called_fn_stmt.value)
                        elif isinstance(stmt, ast.Assign):
                            called_fn_return = stmt.targets[0].id
                            called_fn_stmt.targets[0].id = called_fn_return
                        else:
                            print("ERROR: Unsupported statement type %s" % type(stmt))
                            assert False

                    # RHS
                    rhs = called_fn_stmt.value
                    if isinstance(rhs, ast.Call):
                        for arg in rhs.args:
                            if isinstance(arg, ast.Name):
                                if arg.id not in mapping:
                                    print(
                                        "ERROR: arg.id: %s not in mapping %s "
                                        % (arg.id, mapping)
                                    )
                                    assert False

                                arg.id = mapping[arg.id]
                    elif isinstance(rhs, ast.Name):
                        assert rhs.id in mapping
                        rhs.id = mapping[rhs.id]
                    elif isinstance(rhs, ast.Constant):
                        pass
                    else:
                        print("ERROR: Unsupported statement type %s" % type(rhs))
                        assert False

                    new_body.append(called_fn_stmt)

            # Add the statements of the called function to the caller
            stmt_idx = fn.body.index(stmt)
            fn.body = fn.body[:stmt_idx] + new_body + fn.body[stmt_idx + 1 :]

    return ast.unparse(fn)


def inline_stmts(prog_str):
    prog_ast = ast.parse(prog_str)
    fn = prog_ast.body[-1]

    # if len(fn.body) > 5:
    #    return prog_str

    defs = {arg.arg: arg for arg in fn.args.args}
    to_remove = []

    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            assign_stmt = stmt

            call_stmt = assign_stmt.value
            if isinstance(call_stmt, ast.Call):
                for arg_idx, arg in enumerate(call_stmt.args):
                    if isinstance(arg, ast.Name):
                        call_stmt.args[arg_idx] = defs[arg.id]

                for target in assign_stmt.targets:
                    defs[target.id] = call_stmt
            elif isinstance(call_stmt, ast.Name):
                defs[assign_stmt.targets[0].id] = defs[call_stmt.id]
            elif isinstance(call_stmt, ast.Constant):
                defs[assign_stmt.targets[0].id] = call_stmt
            elif isinstance(call_stmt, ast.BinOp):
                if isinstance(call_stmt.left, ast.Name):
                    call_stmt.left = defs[call_stmt.left.id]
                if isinstance(call_stmt.right, ast.Name):
                    call_stmt.right = defs[call_stmt.right.id]
                defs[assign_stmt.targets[0].id] = call_stmt
            else:
                print("ERROR: Unsupported statement type %s" % type(call_stmt))
                assert False
            to_remove.append(stmt)

        elif isinstance(stmt, ast.Return):
            return_stmt = stmt

            call_stmt = return_stmt.value
            if isinstance(call_stmt, ast.Call):
                for arg_idx, arg in enumerate(call_stmt.args):
                    if isinstance(arg, ast.Name):
                        call_stmt.args[arg_idx] = defs[arg.id]
                return_stmt.value = call_stmt
            elif isinstance(call_stmt, ast.Name):
                return_stmt.value = defs[call_stmt.id]
            else:
                raise NotImplementedError

    for stmt in to_remove:
        fn.body.remove(stmt)

    return ast.unparse(fn)



class JnpFullTransformer(ast.NodeTransformer):
    """
    An AST transformer that visits every 'Call' node.
    If the call is to `jnp.full(..., x)`, it replaces the entire
    call node with just its second argument, `x`.
    """
    def visit_Call(self, node):
        # First, ensure we visit any nested calls within the arguments
        # This allows the transformer to work on nested jnp.full calls
        self.generic_visit(node)

        # Check if this node is the function call we're looking for
        # We need to check:
        # 1. It's an Attribute call (like obj.method)
        # 2. The object 'obj' is a Name (like 'jnp')
        # 3. The name 'id' is 'jnp'
        # 4. The attribute 'attr' is 'full'
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'jnp' and
            node.func.attr == 'full'):

            # It's a match!
            # The user's example `jnp.full([3, 3], 5.0)` implies we want
            # the second argument (index 1).
            if len(node.args) >= 2:
                # Replace this entire ast.Call node with its second argument
                return node.args[1]
            else:
                # This is a jnp.full call with fewer than 2 args.
                # It's safer to just leave it as-is than to crash.
                pass

        # If it wasn't our target function, return the node unchanged
        return node

def replace_jnp_full(py_str: str) -> str:
    """
    Parses a Python string, finds all calls to `jnp.full(..., x)`,
    and replaces them with `x`.
    """
    try:
        # 1. Parse the string into an AST
        tree = ast.parse(py_str)

        # 2. Create and run the transformer
        transformer = JnpFullTransformer()
        transformed_tree = transformer.visit(tree)

        # 3. Fix line numbers and other location info
        ast.fix_missing_locations(transformed_tree)

        # 4. Unparse the transformed tree back into a string
        return ast.unparse(transformed_tree)

    except Exception as e:
        print(f"Error during AST transformation: {e}")
        return py_str # Return original string on error
