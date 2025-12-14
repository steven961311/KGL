import logging
import os
import time
from typing import Dict, List, Tuple

import multiprocessing

from pyeda.boolalg.expr import Expression, Variable
from pyeda.inter import exprvar, And, Or, Not, Nand, Nor, Xor

from BenchParser import BenchParser
from CreateEncryption import get_locked_circuit_bitslice
from utils.logic import convert_str_to_expr

_GATE_REGISTRY: Dict[str, Expression] = {}
_GET_LOCKED = None 


def _worker_init(get_locked_ref):
    global _GET_LOCKED
    _GET_LOCKED = get_locked_ref


def _get_inputs_in_order(exp: Expression) -> tuple[List[str], List[Variable]]:
    # IMPORTANT: deterministic order
    ordered_variables = sorted(list(exp.support), key=lambda v: v.name)
    return [v.name for v in ordered_variables], ordered_variables


def parse_pyeda_funcstyle(expr_str: str) -> Expression:
    allowed_fns = {
        "And": And,
        "Or": Or,
        "Not": Not,
        "Nand": Nand,
        "Nor": Nor,
        "Xor": Xor,
        "Buf": (lambda x: x),  # if appears
    }

    class Env(dict):
        def __missing__(self, k: str):
            if k in allowed_fns:
                return allowed_fns[k]
            v = exprvar(k)
            self[k] = v
            return v

    s = expr_str.strip()
    if s == "0":
        # constant false
        return And(0)
    if s == "1":
        # constant true
        return Or(1)

    return eval(s, {"__builtins__": {}}, Env())


def _encrypt_single_gate(args) -> Tuple[str, str]:
    gate_name, key_values, max_key_bit_size, starting_key = args

    expr = _GATE_REGISTRY.get(gate_name)
    if expr is None:
        raise KeyError(f"Expression for gate '{gate_name}' not found in _GATE_REGISTRY")

    inps_str, _ = _get_inputs_in_order(expr)
    key_names = [f"keyinput{starting_key + i}" for i in range(max_key_bit_size)]

    # Use initializer-provided function reference if available
    if _GET_LOCKED is None:
        locked = get_locked_circuit_bitslice(
            inputs=inps_str,
            outputs=[gate_name],
            expr_strs=[str(expr)],
            keys=key_names,
            key_combinations=key_values,
        )
    else:
        locked = _GET_LOCKED(
            inputs=inps_str,
            outputs=[gate_name],
            expr_strs=[str(expr)],
            keys=key_names,
            key_combinations=key_values,
        )

    return gate_name, locked[gate_name]


class BenchEncryptor:
    def __init__(self, bench_file: str):
        self.bench_file = bench_file
        self.bench_parser = BenchParser(bench_file)
        self.parsed_bench = self.bench_parser

    def get_encrypted_expression_for_bench(self, max_key_bit_size: int):
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
        num_workers: int = 0,
    ) -> Dict[str, Expression]:
        logging.debug("encrypting up to k inputs")
        start = time.time()

        gates_with_k_inputs = self.parsed_bench.get_gates_with_k_inputs(k)
        original_k = k
        while len(gates_with_k_inputs) == 0:
            logging.info(f"Could not find k to encrypt keys, trying k={k+1}")
            k += 1
            gates_with_k_inputs = self.parsed_bench.get_gates_with_k_inputs(k)
            if (k - original_k) > 100:
                raise ValueError("Could not find k to encrypt keys")

        if max_gates_to_lock > 0 and max_gates_to_lock < len(gates_with_k_inputs):
            logging.info(
                f"Cutting gates to encrypt to {max_gates_to_lock}, from {len(gates_with_k_inputs)}"
            )
            gates_with_k_inputs = {
                key: val
                for key, val in list(gates_with_k_inputs.items())[:max_gates_to_lock]
            }

        expr_with_k_inputs: Dict[str, Expression] = {}
        for gate in gates_with_k_inputs:
            expr = convert_str_to_expr(gate, self.parsed_bench.parsed_bench["gates"])
            if expr is not None:
                expr_with_k_inputs[gate] = expr

        logging.info(f"Number of gates to be processed: {len(expr_with_k_inputs)}")
        logging.info(f"Number of key_values to be processed: {len(key_values)}")

        r = self.encrypt_gates(
            expr_with_k_inputs, key_values, max_key_bit_size, num_workers=num_workers
        )

        logging.info(f"Time taken to encrypt up to k inputs: {time.time() - start} seconds")
        return r

    def encrypt_gates(
        self,
        gates: Dict[str, Expression],
        key_values: List[int],
        max_key_bit_size: int,
        num_workers: int = 0,
    ) -> Dict[str, Expression]:
        if len(gates) == 0:
            return {}

        if not num_workers or num_workers <= 0:
            cpu = multiprocessing.cpu_count()
            num_workers = min(max(1, min(cpu, 16)), len(gates))

        global _GATE_REGISTRY
        _GATE_REGISTRY = {}
        for gate, expr in gates.items():
            _GATE_REGISTRY[gate] = expr

        starting_key = 0
        task_args = []
        for gate in sorted(gates.keys()):
            task_args.append((gate, key_values, max_key_bit_size, starting_key))
            starting_key += max_key_bit_size

        logging.info(f"Using {num_workers} worker processes to process {len(gates)} gates")

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(get_locked_circuit_bitslice,),
        ) as pool:
            results = pool.map(_encrypt_single_gate, task_args)

        encrypted_circuits: Dict[str, Expression] = {}
        for gate_name, encrypted_expr_str in results:
            try:
                encrypted_circuits[gate_name] = parse_pyeda_funcstyle(encrypted_expr_str)
            except Exception as e:
                logging.error(f"Failed to parse encrypted expression for {gate_name}: {e}")
                logging.error(f"Expression string was: {encrypted_expr_str}")
                raise

        return encrypted_circuits


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("start \n\n")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_encryptor = BenchEncryptor(cur_dir + "/benchmarks/c17.bench")
    logging.debug(file_encryptor.encrypt_up_to_k_inputs(3))
    logging.debug("end \n\n")