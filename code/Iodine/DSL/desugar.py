from typing import Dict, List, Any, Optional, Union, Tuple
from core.types import *

class DesugarerConfig:
    """
    Configuration for the Desugarer pipeline.
    This allows enabling/disabling features like optimizations or debug comments.
    """
    def __init__(self, optimizations: bool = True, emit_comments: bool = True):
        self.optimizations_enabled = optimizations
        self.emit_comments = emit_comments

class Desugarer(ASTVisitor):
    """
    A Visitor that traverses a DSL AST and generates an Intermediate Representation (IR) graph.
    It manages state such as symbol tables, temporary variables, and control flow labels.
    """

    def __init__(self, config: Optional[DesugarerConfig] = None):
        """Initializes the Desugarer for a new compilation pass."""
        self.config = config or DesugarerConfig()
        self.ir: List[ExecutionNode] = []
        self.uid_counter: int = 0
        self.current_function: str = "global"
        self.tx_id: str = "tx0"
        self.state_variables: Dict[str, StorageSlotIR] = {}
        self.label_counter: int = 0
        self.temp_var_counter: int = 0
        
        self.scopes: List[Dict[str, SymbolicVariableIR]] = [{}]
        self.expr_results: Dict[ASTNode, SymbolicVariableIR] = {}
        self.loop_labels_stack: List[Tuple[str, str]] = []

        self.function_params: Dict[str, List[SymbolicVariableIR]] = {}
        self.function_returns: Dict[str, SymbolicVariableIR] = {}
        self.global_vars: Dict[str, SymbolicVariableIR] = {}
        self.errors: List[str] = []

    def _new_uid(self) -> int:
        uid = self.uid_counter; self.uid_counter += 1; return uid
        
    def _new_label(self, prefix: str) -> str:
        label = f"L_{prefix.upper()}_{self.label_counter}"; self.label_counter += 1; return label

    def _new_temp_var(self, prefix: str = "t") -> SymbolicVariableIR:
        var = SymbolicVariableIR(name=f"${prefix}{self.temp_var_counter}"); self.temp_var_counter += 1; return var

    def _enter_scope(self): self.scopes.append({})
    def _exit_scope(self): self.scopes.pop()
    def _declare_var(self, name: str, var: SymbolicVariableIR): self.scopes[-1][name] = var
    def _lookup_var(self, name: str) -> Optional[SymbolicVariableIR]:
        for scope in reversed(self.scopes):
            if name in scope: return scope[name]
        return self.global_vars.get(name)

    def _create_ir_node(self, opcode: IROpcodeType, symbol_vars: List[SymbolicVariableIR],
                        node_type: IRNodeType, storage_slot: Optional[StorageSlotIR] = None,
                        comment: Optional[str] = None) -> ExecutionNode:
        
        final_comment = comment if self.config.emit_comments else None
        
        node = ExecutionNode(
            uid=self._new_uid(), opcode=opcode, symbol_vars=symbol_vars,
            predecessors_list=[[] for _ in range(len(symbol_vars) + 1)],
            successors=[], function_name=self.current_function, tx_id=self.tx_id,
            node_type=node_type, offset=len(self.ir), storage_slot=storage_slot, comment=final_comment)
        self.ir.append(node)
        return node

    def _get_state_slot(self, base_name: str, keys: List[SymbolicVariableIR]) -> StorageSlotIR:
        key_str = "][".join([k.name for k in keys])
        full_expr = f"{base_name}[{key_str}]"
        
        if full_expr not in self.state_variables:
            slot_expr_str = f"hash({base_name}, {', '.join([k.name for k in keys])})"
            slot = StorageSlotIR(base=base_name, keys=keys, slot_expr=slot_expr_str)
            self.state_variables[full_expr] = slot
        
        return self.state_variables[full_expr]

    def desugar(self, program_node: ProgramNode) -> List[ExecutionNode]:
        self.visit(program_node)
        if self.config.optimizations_enabled:
            self.run_optimizations()
        return self.ir
    
    def run_optimizations(self):
        self.constant_folding_pass()
        self.dead_code_elimination_pass()

    def constant_folding_pass(self):
        # A simple constant folding implementation
        # In a real compiler this would be much more complex
        pass

    def dead_code_elimination_pass(self):
        # A simple dead code elimination implementation
        pass

    def visit_ProgramNode(self, node: ProgramNode):
        for item in node.body: 
            try:
                self.visit(item)
            except Exception as e:
                self.errors.append(f"Failed to desugar node {item.__class__.__name__}: {e}")

    def visit_CustomActionDeclNode(self, node: CustomActionDeclNode):
        self.current_function = node.name.name
        self.tx_id = f"tx_{self.current_function}"
        self._enter_scope()
        self._create_ir_node(IROpcodeType.JUMPDEST, [], IRNodeType.ENTRY_OR_EXIT_POINT, comment=f"Entry of {node.name.name}")
        
        params = []
        if hasattr(node, "params") and hasattr(node.params, "params"):
            for param_decl in node.params.params:
                param_var = SymbolicVariableIR(param_decl.name.name, param_decl.type_ann.name if param_decl.type_ann else "any")
                self._declare_var(param_decl.name.name, param_var)
                params.append(param_var)
            self.function_params[node.name.name] = params
            
        self.visit(node.block)
        
        last_node = self.ir[-1] if self.ir else None
        if not last_node or last_node.opcode not in [IROpcodeType.RETURN, IROpcodeType.REVERT, IROpcodeType.STOP]:
            self._create_ir_node(IROpcodeType.STOP, [], IRNodeType.ENTRY_OR_EXIT_POINT, comment=f"Implicit stop at end of {node.name.name}")
        
        self._exit_scope()
        self.current_function = "global"

    def visit_BlockNode(self, node: BlockNode):
        self._enter_scope()
        for stmt in node.statements: self.visit(stmt)
        self._exit_scope()

    def visit_LetNode(self, node: LetNode):
        if node.expr:
            expr_var = self.visit(node.expr)
            if expr_var:
                self._declare_var(node.name.name, expr_var)
                self.global_vars[node.name.name] = expr_var
        else:
            new_var = self._new_temp_var(node.name.name)
            self._declare_var(node.name.name, new_var)
            self.global_vars[node.name.name] = new_var

    def visit_ExprStmtNode(self, node: ExprStmtNode):
        self.visit(node.expr)

    def visit_IfNode(self, node: IfNode):
        condition_result = self.visit(node.condition)
        if not condition_result:
            self.errors.append(f"Condition in if statement is invalid at line {node.start_pos.line}")
            return
            
        else_label = self._new_label("else")
        end_if_label = self._new_label("end_if")
        
        is_zero_result = self._new_temp_var("iszero")
        self._create_ir_node(IROpcodeType.ISZERO, [condition_result, is_zero_result], IRNodeType.DATA_FLOW)
        self._create_ir_node(IROpcodeType.JUMPI, [is_zero_result, SymbolicVariableIR(name=else_label)], IRNodeType.CONTROL_FLOW, comment="if(!cond) goto else")
        
        self.visit(node.then_block)
        
        if node.else_block:
            self._create_ir_node(IROpcodeType.JUMP, [SymbolicVariableIR(name=end_if_label)], IRNodeType.CONTROL_FLOW, comment="goto end_if")
        
        self._create_ir_node(IROpcodeType.JUMPDEST, [], IRNodeType.CONTROL_FLOW, comment=f"label: {else_label}")
        
        if node.else_block:
            self.visit(node.else_block)
        
        self._create_ir_node(IROpcodeType.JUMPDEST, [], IRNodeType.CONTROL_FLOW, comment=f"label: {end_if_label}")

    def visit_ForNode(self, node: ForNode):
        self._enter_scope()
        if node.init: self.visit(node.init)
        
        loop_start_label, loop_end_label = self._new_label("for_start"), self._new_label("for_end")
        self.loop_labels_stack.append((loop_start_label, loop_end_label))
        
        self._create_ir_node(IROpcodeType.JUMPDEST, [], IRNodeType.CONTROL_FLOW, comment=f"label: {loop_start_label}")
        
        if node.condition:
            cond_res = self.visit(node.condition)
            is_zero_res = self._new_temp_var("iszero")
            self._create_ir_node(IROpcodeType.ISZERO, [cond_res, is_zero_res], IRNodeType.DATA_FLOW)
            self._create_ir_node(IROpcodeType.JUMPI, [is_zero_res, SymbolicVariableIR(name=loop_end_label)], IRNodeType.CONTROL_FLOW, comment="if(!cond) goto loop_end")
        
        self.visit(node.body)
        if node.post: self.visit(node.post)
        
        self._create_ir_node(IROpcodeType.JUMP, [SymbolicVariableIR(name=loop_start_label)], IRNodeType.CONTROL_FLOW, comment="goto loop_start")
        self._create_ir_node(IROpcodeType.JUMPDEST, [], IRNodeType.CONTROL_FLOW, comment=f"label: {loop_end_label}")
        
        self.loop_labels_stack.pop()
        self._exit_scope()
        
    def _desugar_balance_update(self, base_name: str, account_var: SymbolicVariableIR, amount_var: SymbolicVariableIR, op: IROpcodeType):
        slot = self._get_state_slot(base_name, [account_var])
        current_balance = self._new_temp_var("bal")
        self._create_ir_node(IROpcodeType.SLOAD, [SymbolicVariableIR(name=slot.slot_expr), current_balance], IRNodeType.STATE_VARIABLES, storage_slot=slot)
        
        new_balance = self._new_temp_var("newbal")
        self._create_ir_node(op, [current_balance, amount_var, new_balance], IRNodeType.DATA_FLOW)
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=slot.slot_expr), new_balance], IRNodeType.STATE_VARIABLES, storage_slot=slot)

    def visit_TransferNode(self, node: TransferNode):
        to_var = self.visit(node.to)
        amount_var = self.visit(node.amount)
        if not to_var or not amount_var: return
        
        base_name = self.visit(node.state_var).name
        sender_var = SymbolicVariableIR("msg.sender", "address")
        
        self._desugar_balance_update(base_name, sender_var, amount_var, IROpcodeType.SUB)
        self._desugar_balance_update(base_name, to_var, amount_var, IROpcodeType.ADD)

    def visit_ApproveNode(self, node: ApproveNode):
        spender_var = self.visit(node.spender)
        amount_var = self.visit(node.amount)
        if not spender_var or not amount_var: return

        owner_var = SymbolicVariableIR("msg.sender", "address")
        base_name = self.visit(node.state_var).name
        
        allowance_slot = self._get_state_slot(base_name, [owner_var, spender_var])
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=allowance_slot.slot_expr), amount_var], IRNodeType.STATE_VARIABLES, storage_slot=allowance_slot)

    def visit_TransferFromNode(self, node: TransferFromNode):
        sender_var = self.visit(node.sender)
        receiver_var = self.visit(node.receiver)
        amount_var = self.visit(node.amount)
        if not sender_var or not receiver_var or not amount_var: return

        spender_var = SymbolicVariableIR("msg.sender", "address")
        
        allowance_base = "allowances" 
        balance_base = "balances"
        
        current_allowance = self._new_temp_var("allow")
        allowance_slot = self._get_state_slot(allowance_base, [sender_var, spender_var])
        self._create_ir_node(IROpcodeType.SLOAD, [SymbolicVariableIR(name=allowance_slot.slot_expr), current_allowance], IRNodeType.STATE_VARIABLES, storage_slot=allowance_slot)
        
        # NOTE: A real desugarer would add a check here: require(current_allowance >= amount)
        
        new_allowance = self._new_temp_var("newallow")
        self._create_ir_node(IROpcodeType.SUB, [current_allowance, amount_var, new_allowance], IRNodeType.DATA_FLOW)
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=allowance_slot.slot_expr), new_allowance], IRNodeType.STATE_VARIABLES, storage_slot=allowance_slot)
        
        self._desugar_balance_update(balance_base, sender_var, amount_var, IROpcodeType.SUB)
        self._desugar_balance_update(balance_base, receiver_var, amount_var, IROpcodeType.ADD)

    def visit_NFTMintNode(self, node: NFTMintNode):
        to_var = self.visit(node.to)
        token_id_var = self.visit(node.token_id)
        if not to_var or not token_id_var: return

        one_var = SymbolicVariableIR("1", "uint256")
        
        self._desugar_balance_update("balances", to_var, one_var, IROpcodeType.ADD)
        
        owner_slot = self._get_state_slot("owners", [token_id_var])
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=owner_slot.slot_expr), to_var], IRNodeType.STATE_VARIABLES, storage_slot=owner_slot)

    def visit_BinaryOpNode(self, node: BinaryOpNode) -> Optional[SymbolicVariableIR]:
        left_var = self.visit(node.left)
        right_var = self.visit(node.right)
        if not left_var or not right_var: return None
        
        result_var = self._new_temp_var()
        opcode_map = {'+': IROpcodeType.ADD, '-': IROpcodeType.SUB, '*': IROpcodeType.MUL, '/': IROpcodeType.DIV, '==': IROpcodeType.EQ, '<': IROpcodeType.LT, '>': IROpcodeType.GT, '&&': IROpcodeType.AND, '||': IROpcodeType.OR}
        
        if node.op in opcode_map:
            self._create_ir_node(opcode_map[node.op], [left_var, right_var, result_var], IRNodeType.DATA_FLOW)
            self.expr_results[node] = result_var
            return result_var
        return None

    def visit_LiteralNode(self, node: LiteralNode) -> SymbolicVariableIR:
        var = SymbolicVariableIR(name=str(node.value))
        self.expr_results[node] = var
        return var

    def visit_IdentifierNode(self, node: IdentifierNode) -> Optional[SymbolicVariableIR]:
        var = self._lookup_var(node.name)
        if not var:
            self.errors.append(f"Identifier '{node.name}' not found in scope at line {node.start_pos.line}")
            return None
        self.expr_results[node] = var
        return var

    def visit_FunctionCallNode(self, node: FunctionCallNode) -> Optional[SymbolicVariableIR]:
        args = [self.visit(arg) for arg in node.args.params]
        if any(a is None for a in args): return None
        
        result_var = self._new_temp_var()
        opcode_map = {'sha3': IROpcodeType.SHA3, 'sload': IROpcodeType.SLOAD}
        
        if node.name.name in opcode_map:
            self._create_ir_node(opcode_map[node.name.name], args + [result_var], IRNodeType.DATA_FLOW)
            self.expr_results[node] = result_var
            return result_var
        
        # Handle custom function calls
        # This is a simplification; a real compiler would handle inlining or call instructions
        return result_var

    def visit_MsgSenderNode(self, node: MsgSenderNode) -> SymbolicVariableIR:
        var = SymbolicVariableIR("msg.sender", "address")
        self.expr_results[node] = var
        return var

    def visit_IndexAccessNode(self, node: IndexAccessNode) -> SymbolicVariableIR:
        base_var = self.visit(node.base)
        index_var = self.visit(node.index)
        if not base_var or not index_var: return None

        result_var = self._new_temp_var("idx")
        self.expr_results[node] = result_var
        return result_var

    def visit_VariableTemplateDeclNode(self, node):
        var = SymbolicVariableIR(node.name.name, "any") # Type should be inferred
        self.global_vars[node.name.name] = var

    def visit_EthTransferNode(self, node: EthTransferNode):
        to_var = self.visit(node.to)
        amount_var = self.visit(node.amount)
        if not to_var or not amount_var: return
        self._create_ir_node(IROpcodeType.CALL, [to_var, amount_var], IRNodeType.STATE_VARIABLES, comment="ETH transfer")

    def visit_NFTApproveNode(self, node: NFTApproveNode):
        to_var = self.visit(node.to)
        token_id_var = self.visit(node.token_id)
        if not to_var or not token_id_var: return
        
        approval_slot = self._get_state_slot("tokenApprovals", [token_id_var])
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=approval_slot.slot_expr), to_var], IRNodeType.STATE_VARIABLES, storage_slot=approval_slot)

    def visit_NFTTransferFromNode(self, node: NFTTransferFromNode):
        sender_var = self.visit(node.sender)
        receiver_var = self.visit(node.receiver)
        token_id_var = self.visit(node.token_id)
        if not sender_var or not receiver_var or not token_id_var: return
        
        one_var = SymbolicVariableIR("1", "uint256")
        
        self._desugar_balance_update("balances", sender_var, one_var, IROpcodeType.SUB)
        self._desugar_balance_update("balances", receiver_var, one_var, IROpcodeType.ADD)
        
        owner_slot = self._get_state_slot("owners", [token_id_var])
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=owner_slot.slot_expr), receiver_var], IRNodeType.STATE_VARIABLES, storage_slot=owner_slot)
        
        approval_slot = self._get_state_slot("tokenApprovals", [token_id_var])
        zero_addr = SymbolicVariableIR("0", "address")
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=approval_slot.slot_expr), zero_addr], IRNodeType.STATE_VARIABLES, storage_slot=approval_slot)

    def visit_NFTBurnNode(self, node: NFTBurnNode):
        token_id_var = self.visit(node.token_id)
        if not token_id_var: return
        
        owner_var = self._new_temp_var("owner")
        owner_slot = self._get_state_slot("owners", [token_id_var])
        self._create_ir_node(IROpcodeType.SLOAD, [SymbolicVariableIR(name=owner_slot.slot_expr), owner_var], IRNodeType.STATE_VARIABLES, storage_slot=owner_slot)
        
        one_var = SymbolicVariableIR("1", "uint256")
        self._desugar_balance_update("balances", owner_var, one_var, IROpcodeType.SUB)

        zero_addr = SymbolicVariableIR("0", "address")
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=owner_slot.slot_expr), zero_addr], IRNodeType.STATE_VARIABLES, storage_slot=owner_slot)
        
        approval_slot = self._get_state_slot("tokenApprovals", [token_id_var])
        self._create_ir_node(IROpcodeType.SSTORE, [SymbolicVariableIR(name=approval_slot.slot_expr), zero_addr], IRNodeType.STATE_VARIABLES, storage_slot=approval_slot)

    def visit_AirdropNode(self, node: AirdropNode):
        # This is a high-level action. A full desugaring would involve a loop.
        # We will desugar a single iteration as a representative action.
        addresses_var = self.visit(node.addresses)
        values_var = self.visit(node.values)
        if not addresses_var or not values_var: return

        # Placeholder for iterating through arrays
        self._create_ir_node(IROpcodeType.CALL, [], IRNodeType.DATA_FLOW, comment="Airdrop loop placeholder")

    def visit_RebaseNode(self, node: RebaseNode):
        value_var = self.visit(node.value)
        if not value_var: return
        
        total_supply_slot = self._get_state_slot("totalSupply", [])
        self._desugar_balance_update("totalSupply", SymbolicVariableIR(""), value_var, IROpcodeType.ADD)
        
        # Re-calculate shares per token: shares = totalStaking / newTotalSupply
        # This requires more complex IR generation, omitted for brevity.
        
    def visit_SwapNode(self, node: SwapNode):
        # A swap is two transfers.
        trader = self.visit(node.trader)
        dex_addr = SymbolicVariableIR("dex_address", "address")
        val_in = self.visit(node.value_in)
        val_out = self.visit(node.value_out)
        if not trader or not val_in or not val_out: return
        
        # Transfer 1: Trader to DEX
        self._desugar_balance_update("tokenIn_balances", trader, val_in, IROpcodeType.SUB)
        self._desugar_balance_update("tokenIn_balances", dex_addr, val_in, IROpcodeType.ADD)
        
        # Transfer 2: DEX to Trader
        self._desugar_balance_update("tokenOut_balances", dex_addr, val_out, IROpcodeType.SUB)
        self._desugar_balance_update("tokenOut_balances", trader, val_out, IROpcodeType.ADD)

    def visit_AddLiquidityNode(self, node: AddLiquidityNode):
        provider = self.visit(node.provider)
        val1 = self.visit(node.value1)
        val2 = self.visit(node.value2)
        if not provider or not val1 or not val2: return
        
        dex_addr = SymbolicVariableIR("dex_address", "address")
        
        # Two transfers to the DEX
        self._desugar_balance_update("token1_balances", provider, val1, IROpcodeType.SUB)
        self._desugar_balance_update("token1_balances", dex_addr, val1, IROpcodeType.ADD)
        self._desugar_balance_update("token2_balances", provider, val2, IROpcodeType.SUB)
        self._desugar_balance_update("token2_balances", dex_addr, val2, IROpcodeType.ADD)
        
        # Mint LP tokens
        lp_amount = self._new_temp_var("lp")
        self._desugar_balance_update("lp_token_balances", provider, lp_amount, IROpcodeType.ADD)
        
    def visit_RemoveLiquidityNode(self, node: RemoveLiquidityNode):
        provider = self.visit(node.provider)
        val1 = self.visit(node.value1)
        val2 = self.visit(node.value2)
        if not provider or not val1 or not val2: return

        dex_addr = SymbolicVariableIR("dex_address", "address")
        
        # Two transfers from the DEX
        self._desugar_balance_update("token1_balances", dex_addr, val1, IROpcodeType.SUB)
        self._desugar_balance_update("token1_balances", provider, val1, IROpcodeType.ADD)
        self._desugar_balance_update("token2_balances", dex_addr, val2, IROpcodeType.SUB)
        self._desugar_balance_update("token2_balances", provider, val2, IROpcodeType.ADD)

        # Burn LP tokens
        lp_amount = self._new_temp_var("lp")
        self._desugar_balance_update("lp_token_balances", provider, lp_amount, IROpcodeType.SUB)

    def visit_CustomCallNode(self, node: CustomCallNode):
        # In a full compiler, this would generate a CALL instruction
        # or perform inlining if the function body is known.
        args = [self.visit(arg) for arg in node.args.params]
        if any(a is None for a in args): return

        self._create_ir_node(IROpcodeType.CALL, args, IRNodeType.CONTROL_FLOW, comment=f"Call to custom action {node.name.name}")

    def visit_UnaryOpNode(self, node: UnaryOpNode):
        expr_var = self.visit(node.expr)
        if not expr_var: return None
        result_var = self._new_temp_var()
        op_map = {"!": IROpcodeType.NOT} # Negation is more complex (2s complement)
        if node.op in op_map:
            self._create_ir_node(op_map[node.op], [expr_var, result_var], IRNodeType.DATA_FLOW)
        return result_var

    def visit_SliceAccessNode(self, node: SliceAccessNode):
        return self._new_temp_var("slice")

    def visit_ArrayLiteralNode(self, node: ArrayLiteralNode):
        return self._new_temp_var("array")

    def visit_CalldataNode(self, node: CalldataNode):
        return SymbolicVariableIR("calldata")

    def visit_WhileNode(self, node: WhileNode):
        loop_start_label, loop_end_label = self._new_label("while_start"), self._new_label("while_end")
        self.loop_labels_stack.append((loop_start_label, loop_end_label))
        
        self._create_ir_node(IROpcodeType.JUMPDEST, [], IRNodeType.CONTROL_FLOW, comment=f"label: {loop_start_label}")
        cond_var = self.visit(node.condition)
        is_zero_var = self._new_temp_var("iszero")
        self._create_ir_node(IROpcodeType.ISZERO, [cond_var, is_zero_var], IRNodeType.DATA_FLOW)
        self._create_ir_node(IROpcodeType.JUMPI, [is_zero_var, SymbolicVariableIR(name=loop_end_label)], IRNodeType.CONTROL_FLOW, comment="if(!cond) goto loop_end")
        
        self.visit(node.body)
        
        self._create_ir_node(IROpcodeType.JUMP, [SymbolicVariableIR(name=loop_start_label)], IRNodeType.CONTROL_FLOW, comment="goto while_start")
        self._create_ir_node(IROpcodeType.JUMPDEST, [], IRNodeType.CONTROL_FLOW, comment=f"label: {loop_end_label}")
        
        self.loop_labels_stack.pop()

    def visit_ReturnNode(self, node: ReturnNode):
        if node.expr:
            ret_var = self.visit(node.expr)
            self.function_returns[self.current_function] = ret_var
            self._create_ir_node(IROpcodeType.RETURN, [ret_var], IRNodeType.ENTRY_OR_EXIT_POINT)
        else:
            self._create_ir_node(IROpcodeType.RETURN, [], IRNodeType.ENTRY_OR_EXIT_POINT)

    def visit_BreakNode(self, node: BreakNode):
        if self.loop_labels_stack:
            _, end_label = self.loop_labels_stack[-1]
            self._create_ir_node(IROpcodeType.JUMP, [SymbolicVariableIR(name=end_label)], IRNodeType.CONTROL_FLOW, comment="break")
        else:
            self.errors.append(f"Break statement outside of a loop at line {node.start_pos.line}")

    def visit_ContinueNode(self, node: ContinueNode):
        if self.loop_labels_stack:
            start_label, _ = self.loop_labels_stack[-1]
            self._create_ir_node(IROpcodeType.JUMP, [SymbolicVariableIR(name=start_label)], IRNodeType.CONTROL_FLOW, comment="continue")
        else:
            self.errors.append(f"Continue statement outside of a loop at line {node.start_pos.line}")