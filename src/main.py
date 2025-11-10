""" File to run to create a locked version of a circuit

THe following arguments needs to be passed when running the file:
--file  - the .bench file to Encrypt
--key_values - the values of each different key to encrypt. Example "1 2 3 4" 
--k - Number of inputs of the gate to encrypt, it needs to be more than the number of keys
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Union
from pyeda.inter import exprvar, And, Or, Not, Nand, Nor, Xor, espresso_exprs, Xnor
from pyeda.boolalg.expr import Expression
from pprint import pprint, pformat


from BenchKeyGen import BenchEncryptor
from BenchFileWriter import BenchFileWriter
from utils.logic import (
    convert_s_to_map,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(asctime)s -- %(message)s",
    datefmt="%h-%d-%Y   %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)

sys.setrecursionlimit(30000)  # Increase the recursion depth limit to 30000


def all_expr_start_with_and(exprs: List[Expression]) -> bool:
    return all([str(expr).strip().lower().startswith("and") for expr in exprs])


if __name__ == "__main__":
    start = time.time()
    # fmt: off
    parser = argparse.ArgumentParser(description="Encrypt a circuit")
    parser.add_argument("--file", dest="file",  type=str, help="The .bench file to Encrypt", default="./benchmarks/iscas85/c17.bench",)
    parser.add_argument( "--key_values", dest="key_values", type=str, help='The values of each different key to encrypt. Example "1 2 3 4"',  default="1 3 2 3",)
    parser.add_argument( "--k", dest="k", type=int, help="Number of inputs of the gate to encrypt, it needs to be more than the number of keys",  default=2,)
    parser.add_argument('--max_gates_to_lock', dest='number_of_gates_to_lock', type=int, help='Number of gates to lock', default=9999999,)
    args = parser.parse_args()
    # fmt: on
    file = args.file
    key_values = [int(i) for i in args.key_values.split(" ") if i != ""]
    k = int(args.k)
    max_gates_to_lock = args.number_of_gates_to_lock
    number_of_keys = len(key_values)
    if (2**k) < number_of_keys:
        raise ValueError("k needs to be more than the number of keys")
    max_key_bit_size = max([len(bin(n)) - 2 for n in key_values])
    locking_info: list[str] = []
    locking_info.append("# locked version of " + file.split("\\")[-1])
    locking_info.append("# number of keys: " + str(number_of_keys))
    locking_info.append("# keys: " + str(key_values))
    locking_info.append(
        "# number of bits per key (keyinputs#): " + str(max_key_bit_size)
    )
    locking_info.extend(
        [
            f"# key{i} = {'0'*(max_key_bit_size-(len(bin(key))-2)) + bin(key).replace('0b', '')}"
            for i, key in enumerate(key_values)
        ]
    )
    locking_info.append("# k (gate is locked at k bits): " + str(k))
    for info in locking_info:
        logging.info(info[1:])
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    encryptor = BenchEncryptor(os.path.join(cur_dir, file))
    inps = encryptor.parsed_bench.inputs

    outputs = encryptor.parsed_bench.outputs
    gates = encryptor.parsed_bench.gates
    count = 0

    encrypted_gates = encryptor.encrypt_up_to_k_inputs(
        k, key_values, max_key_bit_size, max_gates_to_lock=max_gates_to_lock
    )
    original_k = k
    while all_expr_start_with_and(list(encrypted_gates.values())):
        logging.info(f"all gates start with AND, increasing k to {k + 1} ")
        k += 1
        encrypted_gates = encryptor.encrypt_up_to_k_inputs(
            k, key_values, max_key_bit_size, max_gates_to_lock=max_gates_to_lock
        )
        if k - original_k > 100:
            raise ValueError(
                "The number of gates to lock is too high, try lowering the number of gates to lock"
            )

    inps.extend(
        ["keyinput" + str(i) for i in range(max_key_bit_size * len(encrypted_gates))]
    )

    for gate, expr in encrypted_gates.items():
        m = convert_s_to_map(str(expr), count)
        # print("gate:", gate, "expr:", expr, "\n\n")
        # print("m:\n", pformat(m), "\n")
        count += len(m)
        locking_info.append(f"  {gate} - {expr}")
        # logging.info(f"  {gate} - {expr}")
        if len(m) > 0:
            keys = list(m.keys())
            cur_gate_keys_except_last = {key: m[key] for key in keys[:-1]}
            gates.update(cur_gate_keys_except_last)
            gates[gate] = m[keys[-1]]

    writer = BenchFileWriter(
        path=os.path.join(cur_dir, file.replace(".bench", "") + "_encrypted.bench"),
        inputs=inps,
        outputs=outputs,
        gates=gates,
        flip_flops={},
        locking_info=locking_info,
    )
    writer.write()
    logging.info(f"Time taken to encrypt file: {time.time() - start} seconds\n")
