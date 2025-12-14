# Path: src/CreateEncryption.py

import os
import re
import ast
import tempfile
import subprocess
import logging
from typing import Dict, List
from functools import reduce

import numpy as np

# --- Paths / temp dir (must exist at module top-level for multiprocessing workers) ---
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ABC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../Tools/abc/abc"))
TEMP_DIR = "/dev/shm" if os.path.exists("/dev/shm") else None

_UINT64_ALL1 = np.uint64(0xFFFFFFFFFFFFFFFF)

# ---- Small per-process caches (help a lot when many gates share same fanin) ----
_BITSLICED_INPUTS_CACHE: Dict[int, List[np.ndarray]] = {}


def _parse_bench_to_pyeda_string(bench_content: str) -> Dict[str, str]:
    """
    Parse ABC .bench into {output_wire_name: expanded_expr_str}.
    Supports multiple OUTPUTs.
    Supports basic gates + 2-input LUTs.
    """
    lines = bench_content.splitlines()
    definitions: Dict[str, str] = {}
    output_wires: List[str] = []

    output_pattern = re.compile(r"^\s*OUTPUT\((.+?)\)\s*$")
    gate_pattern = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\((.+?)\)\s*$")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = output_pattern.match(line)
        if m:
            output_wires.append(m.group(1).strip())
            continue

        if line.startswith("INPUT"):
            continue

        m = gate_pattern.match(line)
        if not m:
            continue

        wire_name, gate_type_raw, inputs_str = m.groups()
        gt = gate_type_raw.strip().upper()
        inputs = [x.strip() for x in inputs_str.split(",")]

        if "NOT" in gt:
            expr_str = f"Not({inputs[0]})"
        elif "BUF" in gt:
            expr_str = f"Buf({inputs[0]})"
        elif "AND" in gt and "NAND" not in gt:
            expr_str = f"And({', '.join(inputs)})"
        elif "NAND" in gt:
            expr_str = f"Nand({', '.join(inputs)})"
        elif "OR" in gt and "NOR" not in gt and "XOR" not in gt:
            expr_str = f"Or({', '.join(inputs)})"
        elif "NOR" in gt:
            expr_str = f"Nor({', '.join(inputs)})"
        elif "XOR" in gt:
            expr_str = f"Xor({', '.join(inputs)})"
        elif "LUT" in gt:
            # 2-input LUT only
            try:
                hex_str = gt.split()[-1]
                val = int(hex_str, 16)
                mt = []
                if (val >> 0) & 1:
                    mt.append(f"And(Not({inputs[0]}), Not({inputs[1]}))")
                if (val >> 1) & 1:
                    mt.append(f"And({inputs[0]}, Not({inputs[1]}))")
                if (val >> 2) & 1:
                    mt.append(f"And(Not({inputs[0]}), {inputs[1]})")
                if (val >> 3) & 1:
                    # FIX: removed extra ')'
                    mt.append(f"And({inputs[0]}, {inputs[1]})")

                if not mt:
                    expr_str = "0"
                elif len(mt) == 1:
                    expr_str = mt[0]
                elif len(mt) == 4:
                    expr_str = "1"
                else:
                    expr_str = f"Or({', '.join(mt)})"
            except Exception as e:
                logging.error(f"Failed to parse LUT '{gt}': {e}")
                expr_str = f"And({', '.join(inputs)})"
        else:
            logging.error(f"Unknown gate type '{gate_type_raw}'. Defaulting to And.")
            expr_str = f"And({', '.join(inputs)})"

        definitions[wire_name] = expr_str

    # expand DAG into expressions (text substitution)
    temp_wires = sorted(definitions.keys(), key=len, reverse=True)
    out_exprs: Dict[str, str] = {}

    for ow in output_wires:
        if ow not in definitions:
            out_exprs[ow] = "0"
            continue

        final_str = definitions[ow]
        for _ in range(30):
            changed = False
            for w in temp_wires:
                if w == ow:
                    continue
                pat = r"\b" + re.escape(w) + r"\b"
                if re.search(pat, final_str):
                    final_str = re.sub(pat, definitions[w], final_str)
                    changed = True
            if not changed:
                break

        out_exprs[ow] = final_str

    return out_exprs


def _make_bitsliced_inputs(num_inputs: int) -> List[np.ndarray]:
    """
    Cached bitsliced patterns with the same order as:
      row i -> [(i>>(n-1-j))&1 for j in 0..n-1]
    Returns: list of uint64 arrays, each array shape (W,)
    """
    cached = _BITSLICED_INPUTS_CACHE.get(num_inputs)
    if cached is not None:
        return cached

    if num_inputs <= 0:
        _BITSLICED_INPUTS_CACHE[num_inputs] = []
        return []

    R = 1 << num_inputs
    W = (R + 63) // 64

    idx = np.arange(W * 64, dtype=np.uint64).reshape(W, 64)
    bitpos = np.arange(64, dtype=np.uint64)
    mask = (np.uint64(1) << bitpos).reshape(1, 64)

    words = []
    for j in range(num_inputs):
        shift = np.uint64(num_inputs - 1 - j)
        bits = (idx >> shift) & np.uint64(1)        # (W,64)
        w = (bits * mask).sum(axis=1).astype(np.uint64)
        words.append(w)

    _BITSLICED_INPUTS_CACHE[num_inputs] = words
    return words


class _BitsliceExprEvaluator:
    __slots__ = ("env", "W")

    def __init__(self, env: Dict[str, np.ndarray], W: int):
        self.env = env
        self.W = W

    def _const(self, v):
        if v is True or v == 1:
            return np.full(self.W, _UINT64_ALL1, dtype=np.uint64)
        if v is False or v == 0:
            return np.zeros(self.W, dtype=np.uint64)
        raise TypeError(f"Unsupported constant: {v!r}")

    def eval(self, expr_str: str) -> np.ndarray:
        node = ast.parse(expr_str, mode="eval").body
        return self._eval(node)

    def _eval(self, node) -> np.ndarray:
        if isinstance(node, ast.Name):
            return self.env[node.id]

        if isinstance(node, ast.Constant):
            return self._const(node.value)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return ~self._eval(node.operand)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            fn = node.func.id
            args = [self._eval(a) for a in node.args]

            if fn == "Not":
                return ~args[0]
            if fn == "Buf":
                return args[0]
            if fn == "And":
                return reduce(lambda x, y: x & y, args, np.full(self.W, _UINT64_ALL1, dtype=np.uint64))
            if fn == "Or":
                return reduce(lambda x, y: x | y, args, np.zeros(self.W, dtype=np.uint64))
            if fn == "Xor":
                return reduce(lambda x, y: x ^ y, args, np.zeros(self.W, dtype=np.uint64))
            if fn == "Nand":
                a = reduce(lambda x, y: x & y, args, np.full(self.W, _UINT64_ALL1, dtype=np.uint64))
                return ~a
            if fn == "Nor":
                a = reduce(lambda x, y: x | y, args, np.zeros(self.W, dtype=np.uint64))
                return ~a

            raise ValueError(f"Unsupported function: {fn}")

        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def get_locked_circuit_bitslice(
    inputs: List[str],
    outputs: List[str],
    expr_strs: List[str],
    keys: List[str],
    key_combinations: List[int],
) -> Dict[str, str]:
    """
    Fast: bitslicing evaluate outputs (no restrict), then PLA->ABC->bench->expr-string.
    Keeps original semantics: key assigned per-row using i % len(key_combinations).
    """
    if not os.path.exists(ABC_PATH):
        raise FileNotFoundError(f"ABC binary not found at: {ABC_PATH}")
    if len(outputs) != len(expr_strs):
        raise ValueError("outputs and expr_strs must have the same length")
    if not key_combinations:
        raise ValueError("key_combinations cannot be empty")

    n_in = len(inputs)
    R = 1 << n_in
    W = (R + 63) // 64

    # bitslice inputs (cached by n_in)
    in_words = _make_bitsliced_inputs(n_in)
    env = {name: w for name, w in zip(inputs, in_words)}

    # eval outputs
    ev = _BitsliceExprEvaluator(env, W)
    out_words_list = []
    for s in expr_strs:
        s = s.strip()
        if s == "0":
            out_words_list.append(np.zeros(W, dtype=np.uint64))
        elif s == "1":
            out_words_list.append(np.full(W, _UINT64_ALL1, dtype=np.uint64))
        else:
            out_words_list.append(ev.eval(s))

    # precompute key strings
    num_keys = len(keys)
    total_inputs_count = num_keys + n_in
    key_strs = [format(k, f"0{num_keys}b") for k in key_combinations]
    K = len(key_combinations)

    # stream-write PLA (no giant truth table)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pla", delete=False, dir=TEMP_DIR) as tmp_pla:
        tmp_pla_path = tmp_pla.name
        tmp_pla.write(f".i {total_inputs_count}\n")
        tmp_pla.write(f".o {len(outputs)}\n")
        tmp_pla.write(f".ilb {' '.join(keys + inputs)}\n")
        tmp_pla.write(f".ob {' '.join(outputs)}\n")

        buf = []
        BUF_FLUSH = 8192
        has_any_true = False

        for w in range(W):
            ow = [out_words_list[j][w] for j in range(len(outputs))]
            if all(x == 0 for x in ow):
                continue

            base = w * 64
            for b in range(64):
                i = base + b
                if i >= R:
                    break

                out_bits = []
                any1 = False
                shiftb = np.uint64(b)
                for j in range(len(outputs)):
                    bit = (ow[j] >> shiftb) & np.uint64(1)
                    if bit:
                        any1 = True
                        out_bits.append("1")
                    else:
                        out_bits.append("0")

                if not any1:
                    continue

                has_any_true = True
                kstr = key_strs[i % K]
                istr = format(i, f"0{n_in}b")
                buf.append(f"{kstr}{istr} {''.join(out_bits)}\n")
                if len(buf) >= BUF_FLUSH:
                    tmp_pla.writelines(buf)
                    buf.clear()

        if buf:
            tmp_pla.writelines(buf)

        tmp_pla.write(".e\n")

    if not has_any_true:
        try:
            os.remove(tmp_pla_path)
        except OSError:
            pass
        return {out: "0" for out in outputs}

    tmp_bench_path = tmp_pla_path.replace(".pla", ".bench")

    try:
        abc_cmd = f"read_pla {tmp_pla_path}; strash; write_bench {tmp_bench_path}"
        subprocess.run([ABC_PATH, "-c", abc_cmd], check=True, capture_output=True, text=True)

        if not os.path.exists(tmp_bench_path):
            logging.error(f"ABC output missing: {tmp_bench_path}")
            return {out: "0" for out in outputs}

        with open(tmp_bench_path, "r") as f:
            bench_data = f.read()

        raw_out_exprs_by_wire = _parse_bench_to_pyeda_string(bench_data)

        # map by name first, fallback by OUTPUT order
        out_order = []
        for line in bench_data.splitlines():
            m = re.match(r"^\s*OUTPUT\((.+?)\)\s*$", line.strip())
            if m:
                out_order.append(m.group(1).strip())

        result = {}
        for idx, out in enumerate(outputs):
            if out in raw_out_exprs_by_wire:
                result[out] = raw_out_exprs_by_wire[out]
            elif idx < len(out_order):
                result[out] = raw_out_exprs_by_wire.get(out_order[idx], "0")
            else:
                result[out] = "0"

        return result

    finally:
        try:
            os.remove(tmp_pla_path)
        except OSError:
            pass
        try:
            os.remove(tmp_bench_path)
        except OSError:
            pass