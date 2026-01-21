from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import copy

# ======================================================================================
# SECTION 1: AST VISITOR PATTERN & NODE INFRASTRUCTURE
# ======================================================================================

class ASTVisitor:
    def visit(self, node: 'ASTNode'):
        if node is None:
            return None
        method_name = f'visit_{node.__class__.__name__}'
        visitor_method = getattr(self, method_name, self.generic_visit)
        return visitor_method(node)

    def generic_visit(self, node: 'ASTNode'):
        raise NotImplementedError(f"No visit method for node type {node.__class__.__name__}")

    def visit_children(self, node: 'ASTNode'):
        for child in node.get_children():
            self.visit(child)
            
    def visit_ProgramNode(self, node: 'ProgramNode'): self.visit_children(node)
    def visit_BlockNode(self, node: 'BlockNode'): self.visit_children(node)
    def visit_VariableTemplateDeclNode(self, node: 'VariableTemplateDeclNode'): self.visit_children(node)
    def visit_CustomActionDeclNode(self, node: 'CustomActionDeclNode'): self.visit_children(node)
    def visit_LetNode(self, node: 'LetNode'): self.visit_children(node)
    def visit_IfNode(self, node: 'IfNode'): self.visit_children(node)
    def visit_ForNode(self, node: 'ForNode'): self.visit_children(node)
    def visit_WhileNode(self, node: 'WhileNode'): self.visit_children(node)
    def visit_ReturnNode(self, node: 'ReturnNode'): self.visit_children(node)
    def visit_BreakNode(self, node: 'BreakNode'): self.visit_children(node)
    def visit_ContinueNode(self, node: 'ContinueNode'): self.visit_children(node)
    def visit_ExprStmtNode(self, node: 'ExprStmtNode'): self.visit_children(node)
    def visit_EthTransferNode(self, node: 'EthTransferNode'): self.visit_children(node)
    def visit_ApproveNode(self, node: 'ApproveNode'): self.visit_children(node)
    def visit_NFTApproveNode(self, node: 'NFTApproveNode'): self.visit_children(node)
    def visit_TransferNode(self, node: 'TransferNode'): self.visit_children(node)
    def visit_TransferFromNode(self, node: 'TransferFromNode'): self.visit_children(node)
    def visit_NFTTransferFromNode(self, node: 'NFTTransferFromNode'): self.visit_children(node)
    def visit_NFTMintNode(self, node: 'NFTMintNode'): self.visit_children(node)
    def visit_NFTBurnNode(self, node: 'NFTBurnNode'): self.visit_children(node)
    def visit_AirdropNode(self, node: 'AirdropNode'): self.visit_children(node)
    def visit_RebaseNode(self, node: 'RebaseNode'): self.visit_children(node)
    def visit_SwapNode(self, node: 'SwapNode'): self.visit_children(node)
    def visit_AddLiquidityNode(self, node: 'AddLiquidityNode'): self.visit_children(node)
    def visit_RemoveLiquidityNode(self, node: 'RemoveLiquidityNode'): self.visit_children(node)
    def visit_CustomCallNode(self, node: 'CustomCallNode'): self.visit_children(node)
    def visit_ParamListNode(self, node: 'ParamListNode'): self.visit_children(node)
    def visit_LiteralNode(self, node: 'LiteralNode'): self.visit_children(node)
    def visit_IdentifierNode(self, node: 'IdentifierNode'): self.visit_children(node)
    def visit_BinaryOpNode(self, node: 'BinaryOpNode'): self.visit_children(node)
    def visit_UnaryOpNode(self, node: 'UnaryOpNode'): self.visit_children(node)
    def visit_FunctionCallNode(self, node: 'FunctionCallNode'): self.visit_children(node)
    def visit_IndexAccessNode(self, node: 'IndexAccessNode'): self.visit_children(node)
    def visit_SliceAccessNode(self, node: 'SliceAccessNode'): self.visit_children(node)
    def visit_ArrayLiteralNode(self, node: 'ArrayLiteralNode'): self.visit_children(node)
    def visit_MsgSenderNode(self, node: 'MsgSenderNode'): self.visit_children(node)
    def visit_CalldataNode(self, node: 'CalldataNode'): self.visit_children(node)
    def visit_TypeNode(self, node: 'TypeNode'): self.visit_children(node)
    def visit_CommentNode(self, node: 'CommentNode'): self.visit_children(node)
    def visit_AnnotationNode(self, node: 'AnnotationNode'): self.visit_children(node)
    def visit_ImportNode(self, node: 'ImportNode'): self.visit_children(node)
    def visit_ErrorNode(self, node: 'ErrorNode'): self.visit_children(node)
    def visit_ParamDeclNode(self, node: 'ParamDeclNode'): self.visit_children(node)
    def visit_StructDefNode(self, node: 'StructDefNode'): self.visit_children(node)
    def visit_MemberAccessNode(self, node: 'MemberAccessNode'): self.visit_children(node)

@dataclass
class Position:
    line: int
    column: int
    def __str__(self): return f"{self.line}:{self.column}"

class ASTNode:
    start_pos: Position
    end_pos: Position
    parent: Optional['ASTNode'] = field(default=None, repr=False)

    def accept(self, visitor: ASTVisitor):
        return visitor.visit(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        return all(getattr(self, k) == getattr(other, k) for k in self.__dict__ if k != 'parent')

    def clone(self) -> 'ASTNode':
        return copy.deepcopy(self)

    def get_children(self) -> List['ASTNode']:
        children = []
        for key in self.__dict__:
            if key == 'parent': continue
            value = getattr(self, key)
            if isinstance(value, ASTNode):
                children.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ASTNode):
                        children.append(item)
        return children

    def set_parents(self):
        for child in self.get_children():
            child.parent = self
            child.set_parents()
            
    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if k == 'parent': continue
            if isinstance(v, ASTNode):
                d[k] = v.to_dict()
            elif isinstance(v, list):
                d[k] = [item.to_dict() if isinstance(item, ASTNode) else item for item in v]
            else:
                d[k] = v
        return {self.__class__.__name__: d}


# ======================================================================================
# SECTION 2: DSL TYPE SYSTEM
# ======================================================================================

class DSLType:
    def __eq__(self, other): return isinstance(other, self.__class__) and self.__dict__ == other.__dict__
    def __hash__(self): return hash(repr(self))
    def __repr__(self): return self.__class__.__name__
    def is_assignable_from(self, other: 'DSLType') -> bool: return self == other or isinstance(other, AnyType)
    def is_castable_to(self, other: 'DSLType') -> bool: return self == other

class AnyType(DSLType):
    def is_assignable_from(self, other: 'DSLType') -> bool: return True

class ErrorType(DSLType):
    def is_assignable_from(self, other: 'DSLType') -> bool: return True

class PrimitiveType(DSLType): pass
class IntType(PrimitiveType): pass
class AddressType(PrimitiveType): pass
class BoolType(PrimitiveType): pass
class StringType(PrimitiveType): pass
class VoidType(PrimitiveType):
    def is_assignable_from(self, other: 'DSLType') -> bool: return isinstance(other, VoidType)

@dataclass
class MappingType(DSLType):
    key_type: DSLType
    value_type: DSLType
    def __repr__(self): return f"Mapping({self.key_type} -> {self.value_type})"
    def is_assignable_from(self, other: 'DSLType') -> bool:
        return isinstance(other, MappingType) and self.key_type.is_assignable_from(other.key_type) and self.value_type.is_assignable_from(other.value_type)

@dataclass
class ArrayType(DSLType):
    element_type: DSLType
    def __repr__(self): return f"Array({self.element_type})"
    def is_assignable_from(self, other: 'DSLType') -> bool:
        return isinstance(other, ArrayType) and self.element_type.is_assignable_from(other.element_type)

@dataclass
class StructField:
    name: str
    type: DSLType

@dataclass
class StructType(DSLType):
    name: str
    fields: Dict[str, DSLType]
    def __repr__(self): return f"Struct({self.name})"
    def is_assignable_from(self, other: 'DSLType') -> bool:
        return isinstance(other, StructType) and self.name == other.name

@dataclass
class FunctionType(DSLType):
    param_types: List[DSLType]
    return_type: DSLType
    def __repr__(self): return f"Function(({', '.join(map(str, self.param_types))}) -> {self.return_type})"
    def is_assignable_from(self, other: 'DSLType') -> bool: return self == other

# ======================================================================================
# SECTION 3: AST NODE IMPLEMENTATIONS
# ======================================================================================

@dataclass
class ProgramNode(ASTNode):
    body: List[Union['DeclarationNode', 'StatementNode']]

@dataclass
class BlockNode(ASTNode):
    statements: List['StatementNode']

@dataclass
class ParamDeclNode(ASTNode):
    name: 'IdentifierNode'
    type_ann: 'TypeNode'

@dataclass
class ParamListNode(ASTNode):
    params: List[ParamDeclNode]

@dataclass
class DeclarationNode(ASTNode):
    name: 'IdentifierNode'
    params: ParamListNode

@dataclass
class VariableTemplateDeclNode(DeclarationNode):
    expr: 'ExpressionNode'

@dataclass
class CustomActionDeclNode(DeclarationNode):
    block: 'BlockNode'
    return_type: Optional['TypeNode'] = None
    
@dataclass
class StructDefNode(DeclarationNode):
    name: 'IdentifierNode'
    fields: List[ParamDeclNode]

@dataclass
class StatementNode(ASTNode): pass

@dataclass
class LetNode(StatementNode):
    name: 'IdentifierNode'
    type_ann: Optional['TypeNode'] = None
    expr: Optional['ExpressionNode'] = None

@dataclass
class IfNode(StatementNode):
    condition: 'ExpressionNode'
    then_block: 'BlockNode'
    else_block: Optional['BlockNode'] = None

@dataclass
class ForNode(StatementNode):
    init: Optional[Union['LetNode', 'ExprStmtNode']]
    condition: Optional['ExpressionNode']
    post: Optional['ExpressionNode']
    body: 'BlockNode'

@dataclass
class WhileNode(StatementNode):
    condition: 'ExpressionNode'
    body: 'BlockNode'

@dataclass
class ReturnNode(StatementNode):
    expr: Optional['ExpressionNode'] = None

@dataclass
class BreakNode(StatementNode): pass
@dataclass
class ContinueNode(StatementNode): pass
@dataclass
class ExprStmtNode(StatementNode):
    expr: 'ExpressionNode'

@dataclass
class DeFiActionNode(StatementNode): pass
@dataclass
class EthTransferNode(DeFiActionNode): to: 'ExpressionNode'; amount: 'ExpressionNode'
@dataclass
class ApproveNode(DeFiActionNode): spender: 'ExpressionNode'; amount: 'ExpressionNode'; state_var: 'ExpressionNode'
@dataclass
class NFTApproveNode(DeFiActionNode): to: 'ExpressionNode'; token_id: 'ExpressionNode'
@dataclass
class TransferNode(DeFiActionNode): to: 'ExpressionNode'; amount: 'ExpressionNode'; state_var: 'ExpressionNode'
@dataclass
class TransferFromNode(DeFiActionNode): sender: 'ExpressionNode'; receiver: 'ExpressionNode'; amount: 'ExpressionNode'
@dataclass
class NFTTransferFromNode(DeFiActionNode): sender: 'ExpressionNode'; receiver: 'ExpressionNode'; token_id: 'ExpressionNode'
@dataclass
class NFTMintNode(DeFiActionNode): to: 'ExpressionNode'; token_id: 'ExpressionNode'
@dataclass
class NFTBurnNode(DeFiActionNode): token_id: 'ExpressionNode'
@dataclass
class AirdropNode(DeFiActionNode): addresses: 'ExpressionNode'; values: 'ExpressionNode'
@dataclass
class RebaseNode(DeFiActionNode): value: 'ExpressionNode'
@dataclass
class SwapNode(DeFiActionNode): trader: 'ExpressionNode'; value_in: 'ExpressionNode'; value_out: 'ExpressionNode'
@dataclass
class AddLiquidityNode(DeFiActionNode): provider: 'ExpressionNode'; value1: 'ExpressionNode'; value2: 'ExpressionNode'
@dataclass
class RemoveLiquidityNode(DeFiActionNode): provider: 'ExpressionNode'; value1: 'ExpressionNode'; value2: 'ExpressionNode'
@dataclass
class CustomCallNode(DeFiActionNode):
    name: 'IdentifierNode'
    args: 'ParamListNode'

@dataclass
class ExpressionNode(ASTNode):
    inferred_type: Optional[DSLType] = field(default=None, repr=False)

@dataclass
class LiteralNode(ExpressionNode):
    value: Any
    type_name: str

@dataclass
class IdentifierNode(ExpressionNode):
    name: str
    resolved_decl: Optional[ASTNode] = field(default=None, repr=False)

@dataclass
class BinaryOpNode(ExpressionNode):
    left: ExpressionNode
    op: str
    right: ExpressionNode

@dataclass
class UnaryOpNode(ExpressionNode):
    op: str
    expr: ExpressionNode

@dataclass
class FunctionCallNode(ExpressionNode):
    name: IdentifierNode
    args: ParamListNode

@dataclass
class IndexAccessNode(ExpressionNode):
    base: ExpressionNode
    index: ExpressionNode

@dataclass
class MemberAccessNode(ExpressionNode):
    base: ExpressionNode
    member: IdentifierNode
    
@dataclass
class SliceAccessNode(ExpressionNode):
    base: ExpressionNode
    start: Optional[ExpressionNode]
    end: Optional[ExpressionNode]

@dataclass
class ArrayLiteralNode(ExpressionNode):
    elements: List[ExpressionNode]

@dataclass
class MsgSenderNode(ExpressionNode): pass
@dataclass
class CalldataNode(ExpressionNode): pass
@dataclass
class TypeNode(ASTNode):
    name: str
    width: Optional[int] = None

@dataclass
class CommentNode(ASTNode):
    text: str

@dataclass
class AnnotationNode(ASTNode):
    key: str
    value: Any

@dataclass
class ImportNode(ASTNode):
    module: str
    alias: Optional[str] = None

@dataclass
class ErrorNode(ASTNode):
    error_message: str

# ======================================================================================
# SECTION 4: SYMBOL TABLE & SCOPE MANAGEMENT
# ======================================================================================

@dataclass
class Symbol:
    name: str
    type: DSLType
    decl_node: ASTNode

class Scope:
    def __init__(self, parent: Optional['Scope'] = None):
        self.symbols: Dict[str, Symbol] = {}
        self.parent = parent
        self.children: List['Scope'] = []
        if parent:
            parent.children.append(self)

    def declare(self, name: str, sym_type: DSLType, decl_node: ASTNode) -> bool:
        if name in self.symbols: return False
        self.symbols[name] = Symbol(name, sym_type, decl_node)
        return True

    def lookup(self, name: str) -> Optional[Symbol]:
        if name in self.symbols: return self.symbols[name]
        if self.parent: return self.parent.lookup(name)
        return None
        
    def __repr__(self):
        return f"Scope(symbols={list(self.symbols.keys())})"

class SymbolTable:
    def __init__(self):
        self.root_scope = Scope()
        self.current_scope = self.root_scope

    def enter_scope(self):
        new_scope = Scope(parent=self.current_scope)
        self.current_scope = new_scope

    def exit_scope(self):
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent

    def declare(self, name: str, sym_type: DSLType, decl_node: ASTNode) -> bool:
        return self.current_scope.declare(name, sym_type, decl_node)

    def lookup(self, name: str) -> Optional[Symbol]:
        return self.current_scope.lookup(name)

# ======================================================================================
# SECTION 5: INTERMEDIATE REPRESENTATION (IR) STRUCTURES
# ======================================================================================

class IRNodeType(Enum):
    ENTRY_OR_EXIT_POINT="E"; CONTROL_FLOW="C"; STATE_VARIABLES="S"; DATA_FLOW="D"
    LOG_INFORMATION="L"; ENVIRONMENT_CONTEXT="N"; DELETED="X"; UNKNOWN="U"; PHI="P"

class IROpcodeType(Enum):
    SLOAD="SLOAD"; SSTORE="SSTORE"; SHA3="KECCAK256"; JUMP="JUMP"; JUMPI="JUMPI"
    JUMPDEST="JUMPDEST"; RETURN="RETURN"; REVERT="REVERT"; STOP="STOP"; CALL="CALL"
    DELEGATECALL="DELEGATECALL"; ADD="ADD"; SUB="SUB"; MUL="MUL"; DIV="DIV"
    AND="AND"; OR="OR"; XOR="XOR"; NOT="NOT"; ISZERO="ISZERO"; LT="LT"; GT="GT"
    EQ="EQ"; CALLER="CALLER"; CALLDATALOAD="CALLDATALOAD"; ADDRESS="ADDRESS"; PHI="PHI"

@dataclass
class SymbolicVariableIR:
    name: str
    type_name: str = "uint256"
    ssa_version: int = 0
    def get_ssa_name(self): return f"{self.name}_{self.ssa_version}"
    def clone(self): return copy.deepcopy(self)

@dataclass
class StorageSlotIR:
    base: str
    keys: List[SymbolicVariableIR] = field(default_factory=list)
    slot_expr: str = ""
    def clone(self): return copy.deepcopy(self)

@dataclass
class ExecutionNode:
    uid: int; opcode: IROpcodeType; symbol_vars: List[SymbolicVariableIR]
    predecessors_list: List[List[int]]; successors: List[int]; function_name: str
    tx_id: str; node_type: IRNodeType; offset: int
    storage_slot: Optional[StorageSlotIR] = None; comment: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    def clone(self): return copy.deepcopy(self)
    def get_def(self) -> Optional[SymbolicVariableIR]:
        if self.opcode not in [IROpcodeType.SSTORE, IROpcodeType.JUMP, IROpcodeType.JUMPI, IROpcodeType.RETURN, IROpcodeType.REVERT, IROpcodeType.STOP]:
            return self.symbol_vars[-1] if self.symbol_vars else None
        return None
    def get_uses(self) -> List[SymbolicVariableIR]:
        if self.opcode not in [IROpcodeType.SSTORE, IROpcodeType.JUMP, IROpcodeType.JUMPI, IROpcodeType.RETURN, IROpcodeType.REVERT, IROpcodeType.STOP]:
            return self.symbol_vars[:-1]
        return self.symbol_vars

@dataclass
class BasicBlock:
    uid: int
    nodes: List[ExecutionNode] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    loop_header: bool = False
    
    def get_terminator(self) -> Optional[ExecutionNode]:
        return self.nodes[-1] if self.nodes and self.nodes[-1].opcode in [IROpcodeType.JUMP, IROpcodeType.JUMPI, IROpcodeType.RETURN, IROpcodeType.REVERT, IROpcodeType.STOP] else None

@dataclass
class IRFunction:
    name: str
    entry_block_uid: Optional[int] = None
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)

    def get_reverse_postorder(self) -> List[int]:
        postorder = []
        visited = set()
        if self.entry_block_uid is None: return []
        
        def dfs(uid):
            visited.add(uid)
            block = self.blocks.get(uid)
            if block:
                for succ_uid in block.successors:
                    if succ_uid not in visited:
                        dfs(succ_uid)
            postorder.append(uid)
        
        dfs(self.entry_block_uid)
        return list(reversed(postorder))

@dataclass
class ContractIR:
    name: str
    functions: Dict[str, IRFunction]
    state_variables: Dict[str, StorageSlotIR]

@dataclass
class AnalysisMessage:
    message: str
    node: Optional[ASTNode] = None
    severity: str = "Error"
    def __repr__(self):
        pos = f"L{self.node.start_pos.line}:C{self.node.start_pos.column}" if self.node else "global"
        return f"[{self.severity}@{pos}]: {self.message}"

# ======================================================================================
# SECTION 6: AST NODE FACTORY
# ======================================================================================

class NodeFactory:
    def __init__(self):
        self._line = 1
        self._column = 1

    def set_pos(self, line: int, column: int):
        self._line, self._column = line, column

    def _pos(self): return Position(self._line, self._column)
    def _create(self, cls, **kwargs):
        node = cls(start_pos=self._pos(), end_pos=self._pos(), **kwargs)
        for child in node.get_children(): child.parent = node
        return node
    
    def program(self, body): return self._create(ProgramNode, body=body)
    def block(self, stmts): return self._create(BlockNode, statements=stmts)
    def var_template_decl(self, name, params, expr): return self._create(VariableTemplateDeclNode, name=name, params=params, expr=expr)
    def custom_action_decl(self, name, params, block, ret_type=None): return self._create(CustomActionDeclNode, name=name, params=params, block=block, return_type=ret_type)
    def let_stmt(self, name, type_ann=None, expr=None): return self._create(LetNode, name=name, type_ann=type_ann, expr=expr)
    def if_stmt(self, cond, then_b, else_b=None): return self._create(IfNode, condition=cond, then_block=then_b, else_block=else_b)
    def for_stmt(self, init, cond, post, body): return self._create(ForNode, init=init, condition=cond, post=post, body=body)
    def while_stmt(self, cond, body): return self._create(WhileNode, condition=cond, body=body)
    def return_stmt(self, expr=None): return self._create(ReturnNode, expr=expr)
    def break_stmt(self): return self._create(BreakNode)
    def continue_stmt(self): return self._create(ContinueNode)
    def expr_stmt(self, expr): return self._create(ExprStmtNode, expr=expr)
    def eth_transfer(self, to, amount): return self._create(EthTransferNode, to=to, amount=amount)
    def approve(self, spender, amount, state_var): return self._create(ApproveNode, spender=spender, amount=amount, state_var=state_var)
    def nft_approve(self, to, token_id): return self._create(NFTApproveNode, to=to, token_id=token_id)
    def transfer(self, to, amount, state_var): return self._create(TransferNode, to=to, amount=amount, state_var=state_var)
    def transfer_from(self, sender, receiver, amount): return self._create(TransferFromNode, sender=sender, receiver=receiver, amount=amount)
    def nft_transfer_from(self, sender, receiver, token_id): return self._create(NFTTransferFromNode, sender=sender, receiver=receiver, token_id=token_id)
    def nft_mint(self, to, token_id): return self._create(NFTMintNode, to=to, token_id=token_id)
    def nft_burn(self, token_id): return self._create(NFTBurnNode, token_id=token_id)
    def airdrop(self, addresses, values): return self._create(AirdropNode, addresses=addresses, values=values)
    def rebase(self, value): return self._create(RebaseNode, value=value)
    def swap(self, trader, v_in, v_out): return self._create(SwapNode, trader=trader, value_in=v_in, value_out=v_out)
    def add_liquidity(self, provider, v1, v2): return self._create(AddLiquidityNode, provider=provider, value1=v1, value2=v2)
    def remove_liquidity(self, provider, v1, v2): return self._create(RemoveLiquidityNode, provider=provider, value1=v1, value2=v2)
    def custom_call(self, name, args): return self._create(CustomCallNode, name=name, args=args)
    def param_list(self, params): return self._create(ParamListNode, params=params)
    def literal_int(self, value): return self._create(LiteralNode, value=value, type_name="Int")
    def literal_hex(self, value): return self._create(LiteralNode, value=value, type_name="Hex")
    def literal_bool(self, value): return self._create(LiteralNode, value=value, type_name="Bool")
    def literal_string(self, value): return self._create(LiteralNode, value=value, type_name="String")
    def identifier(self, name): return self._create(IdentifierNode, name=name)
    def binary_op(self, left, op, right): return self._create(BinaryOpNode, left=left, op=op, right=right)
    def unary_op(self, op, expr): return self._create(UnaryOpNode, op=op, expr=expr)
    def function_call(self, name, args): return self._create(FunctionCallNode, name=name, args=args)
    def index_access(self, base, index): return self._create(IndexAccessNode, base=base, index=index)
    def slice_access(self, base, start, end): return self._create(SliceAccessNode, base=base, start=start, end=end)
    def array_literal(self, elements): return self._create(ArrayLiteralNode, elements=elements)
    def msg_sender(self): return self._create(MsgSenderNode)
    def calldata(self): return self._create(CalldataNode)
    def type_node(self, name, width=None): return self._create(TypeNode, name=name, width=width)
    def param_decl(self, name, type_ann): return self._create(ParamDeclNode, name=name, type_ann=type_ann)
    def struct_def(self, name, fields): return self._create(StructDefNode, name=name, fields=fields)
    def member_access(self, base, member): return self._create(MemberAccessNode, base=base, member=member)