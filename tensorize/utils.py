import threading
import queue
import sympy
import time
from sympy.abc import x, y, z # Example symbols
import ast
import jax.numpy as jnp

def arg_ranks_to_args(arg_ranks):
    args = []
    for rank in arg_ranks:
        shape = tuple([3 for _ in range(rank)])
        args.append(jnp.ones(shape))
    return tuple(args)


def get_kernel_function(source_code: str) -> str:
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "kernel":
            return ast.get_source_segment(source_code, node)
    raise ValueError("No kernel function found")


def count_nops(source_function: str) -> int:
    tree = ast.parse(source_function)
    op_count = 0
    
    def is_jnp_call(node):
        return (isinstance(node.func, ast.Attribute) and 
                isinstance(node.func.value, ast.Name) and 
                node.func.value.id == 'jnp')

    for node in ast.walk(tree):
        # Count 1: Binary Operations (e.g., A * B, A + B, A @ C)
        if isinstance(node, ast.BinOp):
            op_count += 1
            
        # Count 2: Function Calls for operations (e.g., jnp.sum, jnp.power)
        elif isinstance(node, ast.Call):
            if is_jnp_call(node):
                op_count += 1
                
        # Count 3: Attribute Access for operations (e.g., A.T)
        elif isinstance(node, ast.Attribute) and node.attr == 'T':
            op_count += 1
                
    return op_count

def _solve_target_thread(equation, symbols, result_queue):
    """
    Target function to run sympy.solve in a separate thread.
    Puts the result in the queue.
    """
    try:
        # This will run in the same process, just a different thread.
        # So, if sympy.solve has a C-level crash, it can still affect the main interpreter.
        solution = sympy.solve(equation, symbols, dict=True)
        result_queue.put(solution)
    except Exception as e:
        result_queue.put(e)
    finally:
        # Always put a sentinel value to indicate completion or termination attempt
        result_queue.put(None) # Sentinel value for thread completion

def solve_with_timeout(equation, symbols, timeout_seconds):
    """
    Wrapper for sympy.solve that runs it in a separate thread with a timeout.
    This avoids the "daemonic processes are not allowed to have children" error
    but does not provide full process isolation.

    Args:
        equation: The equation(s) to solve (e.g., sympy.Eq, or a list of equations).
        symbols: The symbol(s) to solve for (e.g., x, [x, y]).
        timeout_seconds: The maximum time in seconds to wait for a solution.

    Returns:
        The solution from sympy.solve, or None if a timeout occurs.

    Raises:
        Exception: If an exception occurs within sympy.solve in the child thread.
        TimeoutError: If the thread times out.
    """
    result_queue = queue.Queue()
    thread = threading.Thread(
        target=_solve_target_thread,
        args=(equation, symbols, result_queue)
    )

    thread.daemon = True # Allow main program to exit even if thread is stuck
    thread.start()

    try:
        # Wait for the result or a sentinel value (None)
        # We need to loop because the sentinel could be put immediately on error
        # or after the solution is found.
        start_time = time.monotonic()
        result = None
        while True:
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"sympy.solve timed out after {timeout_seconds} seconds.")

            # Calculate remaining time for the queue.get timeout
            remaining_time = timeout_seconds - elapsed_time
            if remaining_time <= 0:
                 raise TimeoutError(f"sympy.solve timed out after {timeout_seconds} seconds.")

            try:
                # Use a small timeout for queue.get to periodically check elapsed time
                item = result_queue.get(timeout=min(remaining_time, 0.1)) # Small timeout to allow checks
                if item is None: # Sentinel received, thread is done (or terminating)
                    break
                result = item # Store the actual result
                # Continue if we got a result, but the thread might still be putting sentinel
            except queue.Empty:
                # Queue was empty, continue loop to check timeout
                continue
        
        # After loop, if we have a result, check if it's an exception
        if isinstance(result, Exception):
            raise result # Re-raise any exception from the child thread
        return result

    finally:
        # Ensure the thread is joined if possible, clean up
        # A daemon thread might not always be joinable if it's truly stuck,
        # but it will be terminated with the main process.
        if thread.is_alive():
            # In a thread, we can't 'terminate' it forcefully like a process
            # unless we add explicit checking within sympy.solve (which we can't)
            # or rely on it being a daemon thread that exits with the main program.
            pass # We rely on daemon=True for cleanup if stuck
        else:
            thread.join() # Join if it completed cleanly