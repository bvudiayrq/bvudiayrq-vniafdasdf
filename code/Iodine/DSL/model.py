# core/model.py

import json
from typing import Dict, List, Tuple, Set, Optional, Callable, TypeVar, Any, Generic
from collections import deque
import itertools
import argparse
import pickle
import sys

T = TypeVar('T')

class Worklist:
    def __init__(self, items: Optional[Iterable[T]] = None):
        self.items = deque(items) if items else deque()
        self.item_set = set(items) if items else set()

    def add(self, item: T):
        if item not in self.item_set:
            self.item_set.add(item)
            self.items.append(item)

    def pop(self) -> T:
        item = self.items.popleft()
        self.item_set.remove(item)
        return item

    def is_empty(self) -> bool:
        return not self.items

    def __len__(self) -> int:
        return len(self.items)


class DataFlowAnalysis(Generic[T]):
    def __init__(self, func: 'IRFunction', forward: bool):
        self.func = func
        self.is_forward = forward
        self.in_facts: Dict[int, T] = {}
        self.out_facts: Dict[int, T] = {}

    def initialize(self, boundary_val: T, top_val: T):
        for uid in self.func.blocks:
            self.in_facts[uid] = top_val
            self.out_facts[uid] = top_val

        if self.func.entry_block:
            entry_uid = self.func.entry_block.uid
            if self.is_forward:
                self.in_facts[entry_uid] = boundary_val
            else:
                exit_blocks = [uid for uid, block in self.func.blocks.items() if not block.successors]
                for uid in exit_blocks:
                    self.out_facts[uid] = boundary_val

    def transfer(self, block: 'BasicBlock', fact: T) -> T:
        raise NotImplementedError

    def meet(self, facts: List[T]) -> T:
        raise NotImplementedError

    def run(self, boundary_val: T, top_val: T):
        self.initialize(boundary_val, top_val)

        worklist = Worklist(self.func.blocks.keys())

        iteration_count = 0
        max_iterations = len(self.func.blocks) * 100

        while not worklist.is_empty():
            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"Warning: Data-flow analysis exceeded {max_iterations} iterations.")
                break

            block_uid = worklist.pop()
            block = self.func.blocks[block_uid]

            if self.is_forward:
                pred_uids = block.predecessors
                in_fact = self.meet([self.out_facts[p] for p in pred_uids]) if pred_uids else self.in_facts[block_uid]
                self.in_facts[block_uid] = in_fact

                old_out_fact = self.out_facts[block_uid]
                new_out_fact = self.transfer(block, in_fact)

                if new_out_fact != old_out_fact:
                    self.out_facts[block_uid] = new_out_fact
                    for s_uid in block.successors:
                        worklist.add(s_uid)
            else:  # Backward
                succ_uids = block.successors
                out_fact = self.meet([self.in_facts[s] for s in succ_uids]) if succ_uids else self.out_facts[block_uid]
                self.out_facts[block_uid] = out_fact

                old_in_fact = self.in_facts[block_uid]
                new_in_fact = self.transfer(block, out_fact)

                if new_in_fact != old_in_fact:
                    self.in_facts[block_uid] = new_in_facts
                    for p_uid in block.predecessors:
                        worklist.add(p_uid)


class LiveVariablesAnalysis(DataFlowAnalysis[Set[str]]):
    def __init__(self, func: 'IRFunction'):
        super().__init__(func, forward=False)
        self.all_vars = self._collect_all_vars()

    def _collect_all_vars(self) -> Set[str]:
        all_vars = set()
        for block in self.func.blocks.values():
            for node in block.nodes:
                for var in node.get_uses():
                    all_vars.add(var.name)
                defined_var = node.get_def()
                if defined_var:
                    all_vars.add(defined_var.name)
        return all_vars

    def meet(self, facts: List[Set[str]]) -> Set[str]:
        result = set()
        for f in facts:
            result.update(f)
        return result

    def transfer(self, block: BasicBlock, fact: Set[str]) -> Set[str]:
        live_out = fact.copy()
        for node in reversed(block.nodes):
            defined_var = node.get_def()
            if defined_var:
                live_out.discard(defined_var.name)
            for var in node.get_uses():
                live_out.add(var.name)
        return live_out


class FSM:
    def __init__(self, name: str):
        self.name: str = name
        self.functions: Dict[str, 'IRFunction'] = {}
        self._analysis_cache: Dict[str, Any] = {}

    def build_from_ir(self, ir_nodes: List['ExecutionNode']):
        nodes_by_func: Dict[str, List['ExecutionNode']] = {}
        for node in ir_nodes:
            nodes_by_func.setdefault(node.function_name, []).append(node)

        for func_name, func_nodes in nodes_by_func.items():
            if func_nodes:
                self.functions[func_name] = self._build_function_cfg(func_name, func_nodes)

    def _build_function_cfg(self, name: str, nodes: List['ExecutionNode']) -> 'IRFunction':
        func = IRFunction(name=name)
        if not nodes:
            return func

        offset_to_node = {node.offset: node for node in nodes}
        leaders = {nodes[0].uid}

        for node in nodes:
            if node.opcode in [IROpcodeType.JUMP, IROpcodeType.JUMPI]:
                if node.offset + 1 in offset_to_node:
                    leaders.add(offset_to_node[node.offset + 1].uid)
                target_var = node.symbol_vars[1] if node.opcode == IROpcodeType.JUMPI else node.symbol_vars[0]
                try:
                    target_offset = int(target_var.name)
                    if target_offset in offset_to_node:
                        leaders.add(offset_to_node[target_offset].uid)
                except (ValueError, TypeError):
                    pass

        current_block = None
        sorted_nodes = sorted(nodes, key=lambda n: n.offset)
        for node in sorted_nodes:
            if node.uid in leaders:
                if current_block:
                    func.blocks[current_block.uid] = current_block
                current_block = BasicBlock(uid=node.uid)
                if func.entry_block is None:
                    func.entry_block = current_block
            if current_block:
                current_block.nodes.append(node)
        if current_block:
            func.blocks[current_block.uid] = current_block

        uid_to_block_id = {node.uid: block.uid for block in func.blocks.values() for node in block.nodes}

        for uid, block in func.blocks.items():
            last_node = block.nodes[-1]
            if last_node.opcode in [IROpcodeType.JUMP, IROpcodeType.JUMPI]:
                target_var = last_node.symbol_vars[1] if last_node.opcode == IROpcodeType.JUMPI else last_node.symbol_vars[0]
                try:
                    target_offset = int(target_var.name)
                    if target_offset in offset_to_node:
                        target_leader_uid = uid_to_block_id[offset_to_node[target_offset].uid]
                        block.successors.append(target_leader_uid)
                        if target_leader_uid in func.blocks:
                            func.blocks[target_leader_uid].predecessors.append(uid)
                except (ValueError, TypeError):
                    pass

            if last_node.opcode not in [IROpcodeType.JUMP, IROpcodeType.RETURN, IROpcodeType.STOP, IROpcodeType.REVERT]:
                next_node_offset = last_node.offset + 1
                if next_node_offset in offset_to_node:
                    fallthrough_leader_uid = uid_to_block_id[offset_to_node[next_node_offset].uid]
                    if fallthrough_leader_uid in func.blocks:
                        block.successors.append(fallthrough_leader_uid)
                        func.blocks[fallthrough_leader_uid].predecessors.append(uid)
        return func

    def run_live_variables_analysis(self, func_name: str) -> Tuple[Dict[int, Set[str]], Dict[int, Set[str]]]:
        func = self.functions.get(func_name)
        if not func:
            return {}, {}
        analysis = LiveVariablesAnalysis(func)
        analysis.run(boundary_val=set(), top_val=set())
        return analysis.in_facts, analysis.out_facts

    def get_dominators(self, func_name: str, post_dom: bool = False) -> Dict[int, Set[int]]:
        cache_key = f"post_dom_{func_name}" if post_dom else f"dom_{func_name}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        func = self.functions.get(func_name)
        if not func or not func.entry_block:
            return {}

        all_blocks = set(func.blocks.keys())
        entry_node = func.entry_block.uid
        if post_dom:
            exit_nodes = [uid for uid, block in func.blocks.items() if not block.successors]
            if not exit_nodes:
                return {}
            entry_node = exit_nodes[0]

        dom = {uid: all_blocks.copy() for uid in all_blocks}
        dom[entry_node] = {entry_node}

        adj = func.blocks
        relatives_attr = 'predecessors' if not post_dom else 'successors'

        changed = True
        while changed:
            changed = False
            order = sorted(list(all_blocks)) if not post_dom else reversed(sorted(list(all_blocks)))
            for uid in order:
                if uid == entry_node:
                    continue
                relatives = getattr(adj[uid], relatives_attr)
                if not relatives:
                    continue
                relative_doms = [dom[p] for p in relatives if p in dom]
                if not relative_doms:
                    continue
                new_dom_set = set.intersection(*relative_doms)
                new_dom_set.add(uid)
                if new_dom_set != dom[uid]:
                    dom[uid] = new_dom_set
                    changed = True

        self._analysis_cache[cache_key] = dom
        return dom

    def get_immediate_dominator(self, func_name: str, doms: Dict[int, Set[int]]) -> Dict[int, Optional[int]]:
        idom = {}
        for n in doms:
            s_doms = doms[n] - {n}
            idom[n] = None
            for d in s_doms:
                is_immediate = True
                for other_d in s_doms:
                    if d != other_d and d in doms.get(other_d, set()):
                        is_immediate = False
                        break
                if is_immediate:
                    idom[n] = d
                    break
        return idom

    def get_dominance_frontier(self, func_name: str) -> Dict[int, Set[int]]:
        cache_key = f"df_{func_name}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        func = self.functions.get(func_name)
        if not func:
            return {}
        doms = self.get_dominators(func_name)
        idoms = self.get_immediate_dominator(func_name, doms)

        df = {uid: set() for uid in func.blocks}
        for b_uid in func.blocks:
            preds = func.blocks[b_uid].predecessors
            if len(preds) >= 2:
                for p_uid in preds:
                    runner = p_uid
                    while runner != idoms.get(b_uid):
                        if runner in df:
                            df[runner].add(b_uid)
                        runner = idoms.get(runner)
                        if runner is None:
                            break
        self._analysis_cache[cache_key] = df
        return df

    def find_loops_in_function(self, func_name: str) -> List[Set[int]]:
        cache_key = f"loops_{func_name}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        doms = self.get_dominators(func_name)
        func = self.functions.get(func_name)
        if not func:
            return []

        loops = []
        for u_uid, block in func.blocks.items():
            for v_uid in block.successors:
                if v_uid in doms.get(u_uid, set()):
                    loop_nodes = {u_uid, v_uid}
                    worklist = Worklist([u_uid])
                    visited_in_loop = {u_uid, v_uid}
                    while not worklist.is_empty():
                        curr = worklist.pop()
                        for pred_uid in func.blocks[curr].predecessors:
                            if pred_uid not in visited_in_loop:
                                loop_nodes.add(pred_uid)
                                worklist.add(pred_uid)
                                visited_in_loop.add(pred_uid)
                    loops.append(loop_nodes)

        self._analysis_cache[cache_key] = loops
        return loops

    def transform_to_ssa(self, func_name: str):
        func = self.functions.get(func_name)
        if not func:
            return

        df = self.get_dominance_frontier(func_name)
        all_vars = self._collect_all_vars_in_func(func)

        for var_name in all_vars:
            defs = self._find_def_blocks(func, var_name)
            worklist = Worklist(defs)
            has_phi = set()

            while not worklist.is_empty():
                d_uid = worklist.pop()
                for f_uid in df.get(d_uid, set()):
                    if f_uid not in has_phi:
                        self._insert_phi_function(func.blocks[f_uid], var_name)
                        has_phi.add(f_uid)
                        if f_uid not in defs:
                            worklist.add(f_uid)

        self._rename_variables(func)

    def _collect_all_vars_in_func(self, func: 'IRFunction') -> Set[str]:
        all_vars = set()
        for block in func.blocks.values():
            for node in block.nodes:
                if node.get_def():
                    all_vars.add(node.get_def().name)
                for use in node.get_uses():
                    all_vars.add(use.name)
        return all_vars

    def _find_def_blocks(self, func: 'IRFunction', var_name: str) -> Set[int]:
        def_blocks = set()
        for uid, block in func.blocks.items():
            for node in block.nodes:
                defined_var = node.get_def()
                if defined_var and defined_var.name == var_name:
                    def_blocks.add(uid)
                    break
        return def_blocks

    def _insert_phi_function(self, block: 'BasicBlock', var_name: str):
        num_preds = len(block.predecessors)
        phi_vars = [SymbolicVariableIR(name=var_name, ssa_version=0) for _ in range(num_preds)]
        phi_def = SymbolicVariableIR(name=var_name, ssa_version=0)
        phi_node = ExecutionNode(
            uid=self._new_uid_for_ir(), opcode=IROpcodeType.PHI,
            symbol_vars=phi_vars + [phi_def],
            predecessors_list=[[] for _ in range(num_preds + 1)],
            successors=[], function_name=block.nodes[0].function_name, tx_id="",
            node_type=IRNodeType.PHI, offset=block.nodes[0].offset - 1
        )
        block.nodes.insert(0, phi_node)

    def _rename_variables(self, func: 'IRFunction'):
        idom = self.get_immediate_dominator(func.name, self.get_dominators(func.name))

        counters = {var: 0 for var in self._collect_all_vars_in_func(func)}
        stacks = {var: [0] for var in self._collect_all_vars_in_func(func)}

        def rename_recursive(block_uid):
            block = func.blocks[block_uid]
            pushed_counts = {var: 0 for var in counters}

            for node in block.nodes:
                if node.opcode != IROpcodeType.PHI:
                    for i, use_var in enumerate(node.get_uses()):
                        if use_var.name in stacks and stacks[use_var.name]:
                            node.symbol_vars[i].ssa_version = stacks[use_var.name][-1]

                defined_var = node.get_def()
                if defined_var:
                    counters[defined_var.name] += 1
                    stacks[defined_var.name].append(counters[defined_var.name])
                    pushed_counts[defined_var.name] += 1
                    defined_var.ssa_version = counters[defined_var.name]

            for succ_uid in block.successors:
                succ_block = func.blocks[succ_uid]
                pred_index = succ_block.predecessors.index(block_uid)
                for node in succ_block.nodes:
                    if node.opcode == IROpcodeType.PHI:
                        use_var = node.symbol_vars[pred_index]
                        if use_var.name in stacks and stacks[use_var.name]:
                            use_var.ssa_version = stacks[use_var.name][-1]

            children = [uid for uid, parent_uid in idom.items() if parent_uid == block_uid]
            for child_uid in children:
                rename_recursive(child_uid)

            for var, count in pushed_counts.items():
                for _ in range(count):
                    stacks[var].pop()

        if func.entry_block:
            rename_recursive(func.entry_block.uid)

    def _new_uid_for_ir(self) -> int:
        max_uid = 0
        for func in self.functions.values():
            for block in func.blocks.values():
                for node in block.nodes:
                    if node.uid > max_uid:
                        max_uid = node.uid
        return max_uid + 1

    def to_dot(self, func_name: str, detailed: bool = True) -> str:
        func = self.functions.get(func_name)
        if not func:
            return "digraph G {}"

        dot = f"digraph {func_name} {{\n"
        dot += "    rankdir=TB; fontname=\"Helvetica\";\n"
        dot += "    node [shape=plaintext];\n"

        for uid, block in func.blocks.items():
            label = f"<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\">"
            label += f"<tr><td bgcolor=\"lightblue\"><b>Block {uid}</b></td></tr>"
            for node in block.nodes:
                op = node.opcode.value.replace('<', '&lt;').replace('>', '&gt;')
                uses = ", ".join([v.get_ssa_name() for v in node.get_uses()])
                defs = node.get_def().get_ssa_name() if node.get_def() else ""

                if node.opcode == IROpcodeType.PHI:
                    inst_str = f"{defs} = &#934;({uses})"
                elif defs:
                    inst_str = f"{defs} = {op}({uses})"
                else:
                    inst_str = f"{op}({uses})"
                label += f"<tr><td align=\"left\">{node.offset}: {inst_str}</td></tr>"
            label += "</table>>"
            dot += f"    {uid} [label={label}];\n"

        for uid, block in func.blocks.items():
            for succ_uid in block.successors:
                dot += f"    {uid} -> {succ_uid};\n"

        dot += "}\n"
        return dot

    def to_json(self, func_name: str) -> str:
        func = self.functions.get(func_name)
        if not func:
            return json.dumps({})
        result = {"name": func.name, "entry_block": func.entry_block.uid if func.entry_block else None, "blocks": {}}
        for uid, block in func.blocks.items():
            result["blocks"][str(uid)] = {"uid": uid, "predecessors": block.predecessors, "successors": block.successors,
                                         "nodes": [node.to_dict() for node in block.nodes]}
        return json.dumps(result, indent=2)


# === 新增：命令行接口支持 ===

class SymbolicVariableIR:
    """简化定义，防止反向引用错误"""
    def __init__(self, name: str, ssa_version: int = 0):
        self.name = name
        self.ssa_version = ssa_version

    def get_ssa_name(self) -> str:
        return f"{self.name}_{self.ssa_version}" if self.ssa_version > 0 else self.name


class IROpcodeType:
    JUMP = type('op', (), {'value': 'JUMP'})()
    JUMPI = type('op', (), {'value': 'JUMPI'})()
    RETURN = type('op', (), {'value': 'RETURN'})()
    STOP = type('op', (), {'value': 'STOP'})()
    REVERT = type('op', (), {'value': 'REVERT'})()
    PHI = type('op', (), {'value': 'PHI'})()


class IRNodeType:
    PHI = "PHI"


class ExecutionNode:
    def __init__(self, uid: int, opcode, symbol_vars: List[SymbolicVariableIR], predecessors_list, successors,
                 function_name: str, tx_id: str, node_type: str, offset: int):
        self.uid = uid
        self.opcode = opcode
        self.symbol_vars = symbol_vars
        self.predecessors_list = predecessors_list
        self.successors = successors
        self.function_name = function_name
        self.tx_id = tx_id
        self.node_type = node_type
        self.offset = offset

    def get_def(self) -> Optional[SymbolicVariableIR]:
        if self.opcode == IROpcodeType.PHI:
            return self.symbol_vars[-1]
        return None

    def get_uses(self) -> List[SymbolicVariableIR]:
        if self.opcode == IROpcodeType.PHI:
            return self.symbol_vars[:-1]
        return self.symbol_vars.copy()

    def to_dict(self):
        return {
            "uid": self.uid,
            "opcode": self.opcode.value,
            "symbol_vars": [{"name": v.name, "ssa_version": v.ssa_version} for v in self.symbol_vars],
            "function_name": self.function_name,
            "offset": self.offset
        }


class IRFunction:
    def __init__(self, name: str):
        self.name = name
        self.blocks: Dict[int, 'BasicBlock'] = {}
        self.entry_block: Optional['BasicBlock'] = None


class BasicBlock:
    def __init__(self, uid: int):
        self.uid = uid
        self.nodes: List[ExecutionNode] = []
        self.predecessors: List[int] = []
        self.successors: List[int] = []


def load_model(pkl_path: str) -> Any:
    """加载.pkl模型文件"""
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model from {pkl_path}: {e}", file=sys.stderr)
        sys.exit(1)


def read_ltl_file(ltl_path: str) -> List[str]:
    """读取LTL属性文件，每行一个属性"""
    try:
        with open(ltl_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return lines
    except Exception as e:
        print(f"Error reading LTL file {ltl_path}: {e}", file=sys.stderr)
        sys.exit(1)


def fsm_to_smv(fsm: FSM, ltl_props: List[str]) -> str:
    """
    将 FSM 转换为 NuSMV 模型代码
    """
    func_name = next(iter(fsm.functions)) if fsm.functions else ""
    func = fsm.functions.get(func_name)
    if not func or not func.entry_block:
        return "-- ERROR: No valid function found in model\n"

    states = ', '.join(str(uid) for uid in sorted(func.blocks.keys()))
    entry = func.entry_block.uid

    smv = f"""MODULE main
VAR
    pc : {{{states}}};

ASSIGN
    init(pc) := {entry};
    next(pc) :=
        case\n"""

    for uid, block in func.blocks.items():
        for succ in block.successors:
            smv += f"            (pc = {uid}) : {succ};\n"

    smv += """            TRUE : pc;
        esac;

-- LTL Properties
"""
    for i, prop in enumerate(ltl_props, 1):
        smv += f"LTLSPEC {prop}; -- Property {i}\n"

    return smv


def main():
    parser = argparse.ArgumentParser(description="Convert a serialized FSM model to NuSMV (.smv) format with LTL properties.")
    parser.add_argument('model_pkl', help='Path to the serialized model (.pkl file)')
    parser.add_argument('--ltl', required=True, help='Path to the LTL properties file (.ltl)')
    args = parser.parse_args()

    # 加载模型
    fsm = load_model(args.model_pkl)

    # 读取 LTL 属性
    ltl_props = read_ltl_file(args.ltl)

    # 转换为 SMV
    smv_code = fsm_to_smv(fsm, ltl_props)

    # 输出到 stdout
    print(smv_code)


if __name__ == "__main__":
    main()