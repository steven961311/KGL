import logging
import os
import time
from multiprocessing import Pool, cpu_count
from BenchParser import BenchParser
from CreateEncryption import get_locked_circuit
from utils.logic import convert_str_to_expr

# Path: src/BenchParser.py

"""This module will get a Expression from parser_bench_file and create the encrypted circuit using the CreateEncryption module"""

from typing import Callable, Dict, List, Tuple
from pyeda.boolalg.expr import Expression, Variable
from pyeda.inter import exprvar

"""    def Y( A_: int, B_: int, k1_: int, k2_: int) -> Literal[1, 0]:
        A, B, k1, k2 = map(exprvar, ["A", "B", "k1", "k2"])
        point = {
            A: A_,
            B: B_,
            k1: k1_,
            k2: k2_,
        }
        expr = Or(And(~k1, k2, A, ~B), And(k1, k2, B))
        return int(expr.restrict(point))  # type: ignore
"""


def convert_exp_to_fn(
    exp: Expression, var_names_for_args_order: List[Variable]
) -> Callable:
    def fn(
        *args,
    ):  # note that the args are the inputs to the function and need to be in the same order as the expression
        # logging.debug(f"var_names_for_args_order = {var_names_for_args_order}")
        point = {}
        for i, arg in enumerate(args):
            point[exprvar(var_names_for_args_order[i].name)] = arg
        return int(exp.restrict(point))

    return fn


def _get_inputs_in_order(exp: Expression) -> tuple[List[str], List[Variable]]:
    """Helper function to get inputs in order (needs to be at module level for pickling)"""
    variables = exp.support
    # Create a list to store the variables in order
    ordered_variables = []
    ordered_variables_str = []
    # Iterate over the tokens in the expression
    for token in variables:
        # If the token is a variable and it is not already in the list, add it to the list
        if token and token not in ordered_variables:
            ordered_variables.append(token)
            ordered_variables_str.append(token.name)
    return ordered_variables_str, ordered_variables


def _encrypt_single_gate(args: Tuple[str, str, List[int], int, int]) -> Tuple[str, str]:
    """
    Helper function to encrypt a single gate (needs to be at module level for multiprocessing)
    
    Parameters:
    args: Tuple containing (gate_name, expr_str, key_values, max_key_bit_size, starting_key)
    
    Returns:
    Tuple[str, str]: (gate_name, encrypted_expression_str)
    """
    gate, expr_str, key_values, max_key_bit_size, starting_key = args
    
    # Convert string back to expression
    from pyeda.inter import expr as pyeda_expr
    expr = pyeda_expr(expr_str)
    
    inps_str, inps_var = _get_inputs_in_order(expr)
    key_names = [
        "keyinput" + str(i + starting_key) for i in range(max_key_bit_size)
    ]

    fn = convert_exp_to_fn(expr, inps_var)

    # Create the encrypted circuit
    encrypted_circuit = get_locked_circuit(
        inputs=inps_str,
        outputs=[gate],
        output_fns=[fn],
        keys=key_names,
        key_combinations=key_values,
    )
    
    # Convert expression to string for return
    encrypted_expr = list(encrypted_circuit.values())[0]
    return (gate, str(encrypted_expr))


class BenchEncryptor:
    def __init__(self, bench_file: str):
        self.bench_file = bench_file
        self.parsed_bench = BenchParser(bench_file)
        # self.output_functions = self.parsed_bench.get_output_functions()

    def __get_inputs_in_order(
        self, exp: Expression
    ) -> tuple[List[str], List[Variable]]:
        return _get_inputs_in_order(exp)

    def get_encrypted_expression_for_bench(
        self, max_key_bit_size: int
    ) -> Tuple[Dict[str, Expression], Dict[str, Expression]]:
        """
        This function reads a bench file, parses the boolean functions, and creates the encrypted circuit.

        Parameters:
        bench_file (str): The path to the bench file.

        Returns:
        Tuple[Dict[str, Expression], Dict[str, Expression]]: A tuple containing two dictionaries:
                - The original boolean functions of the circuit. Expressions pyEda
                - The second dictionary is the encrypted boolean functions of the encrypted circuit. Expressions pyEda
        """
        # Parse the bench file to get the simplified boolean functions
        simplified_functions = self.parsed_bench.get_output_functions()

        return simplified_functions, self.encrypt_gates(
            simplified_functions, [1, 3], max_key_bit_size
        )

    def encrypt_up_to_k_inputs(
        self,
        k: int,
        key_values: List[int] = [1, 3],
        max_key_bit_size: int = 2,
        max_gates_to_lock: int = 0,
    ) -> Dict[str, Expression]:
        """
        This function encrypts the circuit up to k inputs.

        Parameters:
        bench_file (str): The path to the bench file.

        Returns:
        Tuple[Dict[str, Expression], Dict[str, Expression]]: A tuple containing two dictionaries:
                - The original boolean functions of the circuit. Expressions pyEda
                - The second dictionary is the encrypted boolean functions of the encrypted circuit. Expressions pyEda
        """
        logging.debug("encrypting up to k inputs")
        start = time.time()
        gates_with_k_inputs = self.parsed_bench.get_gates_with_k_inputs(k)
        # { "gate_name": ['x1', 'x2' ...'xk' ]}
        original_k = k
        while len(gates_with_k_inputs) == 0:
            logging.info(f"Could not find k to encrypt keys, trying k={k+1}")
            k += 1
            gates_with_k_inputs = self.parsed_bench.get_gates_with_k_inputs(k)
            if (k - original_k) > 100:
                raise ValueError("Could not find k to encrypt keys")

        # cut the gates to encrypt to the max_gates_to_lock
        if max_gates_to_lock < len(gates_with_k_inputs):
            logging.info(
                f"Cutting gates to encrypt to {max_gates_to_lock}, from {len(gates_with_k_inputs)}"
            )
            gates_with_k_inputs = {
                key: val
                for key, val in list(gates_with_k_inputs.items())[:max_gates_to_lock]
            }
        expr_with_k_inputs: dict[str, Expression] = {
            gate: expr
            for gate in gates_with_k_inputs
            if (
                expr := convert_str_to_expr(
                    gate, self.parsed_bench.parsed_bench["gates"]
                )
            )
            is not None
        }
        logging.info(f"Number of gates to be processed: {len(expr_with_k_inputs)}")
        logging.info(f"Number of key_values to be processed: {len(key_values)}")
        r = self.encrypt_gates(expr_with_k_inputs, key_values, max_key_bit_size)
        logging.info(
            f"Time taken to encrypt up to k inputs: {time.time() - start} seconds"
        )
        return r

    def encrypt_gates(
        self,
        gates: dict[str, Expression],
        key_values: List[int],
        max_key_bit_size: int,
        num_workers: int = 2,
    ) -> Dict[str, Expression]:
        """
        This function encrypts the gates in the circuit using multiprocessing.

        Parameters:
        gates: Dictionary of gate names and their expressions
        key_values: List of key values to use for encryption
        max_key_bit_size: Maximum key bit size
        num_workers: Number of worker processes (default: 2)

        Returns:
        Dict[str, Expression]: Dictionary of encrypted expressions
        """
        if len(gates) == 0:
            return {}
        
        # Prepare arguments for each gate (convert Expression to string)
        starting_key = 0
        gate_args = []
        for gate, expr in gates.items():
            gate_args.append((gate, str(expr), key_values, max_key_bit_size, starting_key))
            starting_key += max_key_bit_size
        
        # Split work between workers
        logging.info(f"Using {num_workers} workers to process {len(gates)} gates")
        
        encrypted_circuits: Dict[str, Expression] = {}
        
        # Use multiprocessing Pool
        with Pool(processes=num_workers) as pool:
            results = pool.map(_encrypt_single_gate, gate_args)
        
        # Collect results and convert string back to Expression
        from pyeda.inter import expr as pyeda_expr
        for gate_name, encrypted_expr_str in results:
            encrypted_circuits[gate_name] = pyeda_expr(encrypted_expr_str)
        
        return encrypted_circuits


if __name__ == "__main__":
    logging.debug("start \n\n")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_encryptor = BenchEncryptor(cur_dir + "/benchmarks/c17.bench")
    #     simplified_functions, encrypted_circuit = (
    #         file_encryptor.get_encrypted_expression_for_bench()
    #     )
    #
    #     logging.debug("Simplified Functions:")
    #     for output, exp in simplified_functions.items():
    #         logging.debug(f"{output}: {exp}")
    #
    #     logging.debug("\nEncrypted Circuit:")
    #     for output, exp in encrypted_circuit.items():
    #         logging.debug(f"{output}: {exp}")

    logging.debug(file_encryptor.encrypt_up_to_k_inputs(3))
    logging.debug("end \n\n")