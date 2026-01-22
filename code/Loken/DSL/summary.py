# core/summary.py

from typing import List, Dict, Any, Optional, Union, Tuple, Set
from core.types import *

class AnalysisPipeline:
    def __init__(self, ast: ProgramNode):
        self.ast = ast
        self.errors: List[AnalysisMessage] = []
        self.type_results: Dict[ASTNode, DSLType] = {}

    def run(self) -> List[AnalysisMessage]:
        type_checker = TypeChecker()
        self.errors.extend(type_checker.check(self.ast))
        self.type_results = type_checker.type_map
        
        if not any(e.is_error() for e in self.errors):
            semantic_validator = SemanticValidator(self.type_results)
            self.errors.extend(semantic_validator.validate(self.ast))
        
        return self.errors

class TypeChecker(ASTVisitor):
    def __init__(self):
        self.scopes: List[Dict[str, DSLType]] = [{}]
        self.errors: List[AnalysisMessage] = []
        self.global_scope: Dict[str, DSLType] = {
            "sha3": FunctionType(param_types=[AnyType()], return_type=IntType()),
            "sload": FunctionType(param_types=[IntType()], return_type=IntType()),
            "concat": FunctionType(param_types=[IntType(), IntType()], return_type=IntType()),
            "length": FunctionType(param_types=[ArrayType(AnyType())], return_type=IntType()),
            "msg.sender": AddressType(),
            "calldata": ArrayType(IntType()),
        }
        self.scopes[0] = self.global_scope.copy()
        self.current_function_return_type: Optional[DSLType] = None
        self.type_map: Dict[ASTNode, DSLType] = {}

    def check(self, node: ASTNode) -> List[AnalysisMessage]:
        self.visit(node)
        return self.errors

    def _enter_scope(self): self.scopes.append({})
    def _exit_scope(self): self.scopes.pop()
    
    def _declare(self, name: str, type: DSLType, node: ASTNode):
        if name in self.scopes[-1]:
            self.add_error(f"Identifier '{name}' is already declared in this scope.", node)
        else:
            self.scopes[-1][name] = type

    def _lookup(self, name: str) -> Optional[DSLType]:
        for scope in reversed(self.scopes):
            if name in scope: return scope[name]
        return None

    def add_error(self, message: str, node: ASTNode):
        self.errors.append(AnalysisMessage(message, node, "Error"))

    def _check_type(self, node: ASTNode, expected: DSLType) -> DSLType:
        actual = self.visit(node)
        if not expected.is_assignable_from(actual):
            self.add_error(f"Type mismatch. Expected '{expected}', but got '{actual}'.", node)
            return ErrorType()
        return actual

    def visit_ProgramNode(self, node: ProgramNode):
        for item in node.body:
            if isinstance(item, (CustomActionDeclNode, StructDefNode)):
                self._declare_symbols(item)

        for item in node.body:
            self.visit(item)

    def _declare_symbols(self, node: Union[CustomActionDeclNode, StructDefNode]):
        if isinstance(node, CustomActionDeclNode):
            param_types = [self.visit(p.type_ann) for p in node.params.params]
            return_type = self.visit(node.return_type) if node.return_type else VoidType()
            func_type = FunctionType(param_types, return_type)
            self._declare(node.name.name, func_type, node.name)
        elif isinstance(node, StructDefNode):
            fields = {field.name.name: self.visit(field.type_ann) for field in node.fields}
            struct_type = StructType(node.name.name, fields)
            self._declare(node.name.name, struct_type, node.name)

    def visit_CustomActionDeclNode(self, node: CustomActionDeclNode):
        func_type = self._lookup(node.name.name)
        if isinstance(func_type, FunctionType):
            self.current_function_return_type = func_type.return_type
        
        self._enter_scope()
        for i, param_decl in enumerate(node.params.params):
            param_type = self.visit(param_decl.type_ann)
            self._declare(param_decl.name.name, param_type, param_decl.name)
        
        self.visit(node.block)
        self._exit_scope()
        self.current_function_return_type = None

    def visit_BlockNode(self, node: BlockNode):
        self._enter_scope()
        for stmt in node.statements: self.visit(stmt)
        self._exit_scope()

    def visit_LetNode(self, node: LetNode):
        declared_type = self.visit(node.type_ann) if node.type_ann else None
        inferred_type = self.visit(node.expr) if node.expr else None
        
        if declared_type and inferred_type:
            if not declared_type.is_assignable_from(inferred_type):
                self.add_error(f"Cannot assign expression of type '{inferred_type}' to variable '{node.name.name}' of type '{declared_type}'.", node.expr)
        
        final_type = declared_type or inferred_type or AnyType()
        self._declare(node.name.name, final_type, node.name)

    def visit_IfNode(self, node: IfNode):
        self._check_type(node.condition, BoolType())
        self.visit(node.then_block)
        if node.else_block:
            self.visit(node.else_block)

    def visit_ReturnNode(self, node: ReturnNode):
        actual_return_type = self.visit(node.expr) if node.expr else VoidType()
        expected = self.current_function_return_type or VoidType()
        if not expected.is_assignable_from(actual_return_type):
            self.add_error(f"Return statement has type '{actual_return_type}' but function expects '{expected}'.", node)

    def visit_BinaryOpNode(self, node: BinaryOpNode) -> DSLType:
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        
        numeric_ops = {'+', '-', '*', '/', '%', '**'}
        comparison_ops = {'<', '>', '<=', '>=', '==', '!='}
        logical_ops = {'&&', '||'}
        
        result_type: DSLType = ErrorType()

        if node.op in numeric_ops:
            if isinstance(left_type, IntType) and isinstance(right_type, IntType):
                result_type = IntType()
            else:
                self.add_error(f"Operator '{node.op}' requires two integer operands, but got '{left_type}' and '{right_type}'.", node)
        elif node.op in comparison_ops:
            if not left_type.is_assignable_from(right_type) and not right_type.is_assignable_from(left_type):
                 self.add_error(f"Cannot compare incompatible types '{left_type}' and '{right_type}'.", node)
            result_type = BoolType()
        elif node.op in logical_ops:
            if isinstance(left_type, BoolType) and isinstance(right_type, BoolType):
                result_type = BoolType()
            else:
                self.add_error(f"Operator '{node.op}' requires two boolean operands.", node)
        else:
            self.add_error(f"Unsupported binary operator '{node.op}'.", node)
        
        self.type_map[node] = result_type
        return result_type

    def visit_IdentifierNode(self, node: IdentifierNode) -> DSLType:
        var_type = self._lookup(node.name)
        if var_type is None:
            self.add_error(f"Undeclared identifier '{node.name}'.", node)
            var_type = ErrorType()
        self.type_map[node] = var_type
        node.inferred_type = var_type
        return var_type

    def visit_LiteralNode(self, node: LiteralNode) -> DSLType:
        type_map = {"Int": IntType(), "Hex": AddressType(), "Bool": BoolType(), "String": StringType()}
        dsl_type = type_map.get(node.type_name, AnyType())
        self.type_map[node] = dsl_type
        node.inferred_type = dsl_type
        return dsl_type
    
    def visit_FunctionCallNode(self, node: FunctionCallNode) -> DSLType:
        callee_type = self.visit(node.name)
        if not isinstance(callee_type, FunctionType):
            self.add_error(f"'{node.name.name}' is not a function and cannot be called.", node.name)
            return ErrorType()

        expected_params = callee_type.param_types
        actual_args = node.args.params if node.args else []
        if len(expected_params) != len(actual_args):
            self.add_error(f"Function '{node.name.name}' expects {len(expected_params)} arguments but received {len(actual_args)}.", node)
            return callee_type.return_type

        for i, arg_node in enumerate(actual_args):
            arg_type = self.visit(arg_node)
            if not expected_params[i].is_assignable_from(arg_type):
                self.add_error(f"Argument {i+1} for '{node.name.name}' has wrong type. Expected '{expected_params[i]}', got '{arg_type}'.", arg_node)
        
        self.type_map[node] = callee_type.return_type
        node.inferred_type = callee_type.return_type
        return callee_type.return_type

    def visit_IndexAccessNode(self, node: IndexAccessNode) -> DSLType:
        base_type = self.visit(node.base)
        index_type = self.visit(node.index)
        
        result_type: DSLType = ErrorType()
        if isinstance(base_type, MappingType):
            if not base_type.key_type.is_assignable_from(index_type):
                self.add_error(f"Mapping requires key of type '{base_type.key_type}', but got '{index_type}'.", node.index)
            result_type = base_type.value_type
        elif isinstance(base_type, ArrayType):
            if not isinstance(index_type, IntType):
                self.add_error(f"Array index must be an integer, but got '{index_type}'.", node.index)
            result_type = base_type.element_type
        else:
            self.add_error(f"Type '{base_type}' is not indexable.", node.base)
        
        self.type_map[node] = result_type
        node.inferred_type = result_type
        return result_type

    def visit_MemberAccessNode(self, node: MemberAccessNode) -> DSLType:
        base_type = self.visit(node.base)
        member_name = node.member.name
        
        result_type: DSLType = ErrorType()
        if isinstance(base_type, StructType):
            if member_name not in base_type.fields:
                self.add_error(f"Struct '{base_type.name}' has no member named '{member_name}'.", node.member)
            else:
                result_type = base_type.fields[member_name]
        else:
            self.add_error(f"Type '{base_type}' does not have members.", node.base)
            
        self.type_map[node] = result_type
        node.inferred_type = result_type
        return result_type

    def visit_TypeNode(self, node: TypeNode) -> DSLType:
        type_map = {"Int": IntType(), "Address": AddressType(), "Bool": BoolType(), "String": StringType(), "Void": VoidType()}
        dsl_type = type_map.get(node.name, AnyType())
        
        struct_type = self._lookup(node.name)
        if isinstance(struct_type, StructType):
            dsl_type = struct_type
        
        self.type_map[node] = dsl_type
        return dsl_type
        
    def generic_visit(self, node: ASTNode) -> AnyType:
        for attr in dir(node):
            if attr.startswith('_') or attr == 'parent': continue
            value = getattr(node, attr)
            if isinstance(value, ASTNode):
                self.visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ASTNode): self.visit(item)
        
        result_type = AnyType()
        self.type_map[node] = result_type
        if hasattr(node, 'inferred_type'):
            node.inferred_type = result_type
        return result_type
        
class SemanticValidator(ASTVisitor):
    def __init__(self, type_map: Dict[ASTNode, DSLType]):
        self.errors: List[AnalysisMessage] = []
        self.loop_depth: int = 0
        self.type_map = type_map
        self.current_function: Optional[CustomActionDeclNode] = None

    def validate(self, node: ASTNode) -> List[AnalysisMessage]:
        self.visit(node)
        return self.errors

    def add_error(self, message: str, node: ASTNode): self.errors.append(AnalysisMessage(message, node, "Error"))
    def add_warning(self, message: str, node: ASTNode): self.errors.append(AnalysisMessage(message, node, "Warning"))

    def visit_CustomActionDeclNode(self, node: CustomActionDeclNode):
        self.current_function = node
        self.visit(node.block)
        # Check for missing return statement in non-void functions
        if self.type_map.get(node.return_type) != VoidType() and not self._is_path_terminating(node.block):
            self.add_error(f"Function '{node.name.name}' might not return a value on all paths.", node.name)
        self.current_function = None

    def _is_path_terminating(self, node: ASTNode) -> bool:
        if isinstance(node, ReturnNode): return True
        if isinstance(node, IfNode):
            return self._is_path_terminating(node.then_block) and \
                   (node.else_block is not None and self._is_path_terminating(node.else_block))
        if isinstance(node, BlockNode):
            return any(self._is_path_terminating(stmt) for stmt in node.statements)
        return False

    def visit_ForNode(self, node: ForNode):
        self.loop_depth += 1
        self.visit_children(node)
        self.loop_depth -= 1

    def visit_WhileNode(self, node: WhileNode):
        self.loop_depth += 1
        self.visit_children(node)
        self.loop_depth -= 1

    def visit_BreakNode(self, node: BreakNode):
        if self.loop_depth == 0:
            self.add_error("'break' statement found outside of a loop.", node)

    def visit_ContinueNode(self, node: ContinueNode):
        if self.loop_depth == 0:
            self.add_error("'continue' statement found outside of a loop.", node)

    def visit_TransferNode(self, node: TransferNode):
        amount_node = node.amount
        if isinstance(amount_node, LiteralNode) and self.type_map.get(amount_node) == IntType():
            if int(amount_node.value) < 0:
                self.add_warning("Transfer amount is a negative literal. This will likely cause an underflow.", amount_node)
        self.visit_children(node)

    def visit_LetNode(self, node: LetNode):
        if node.expr is None and node.type_ann is None:
            self.add_warning(f"Variable '{node.name.name}' is declared without a type or initial value.", node)
        self.visit_children(node)
        
    def generic_visit(self, node: ASTNode):
        self.visit_children(node)

class ASTOptimizer(ASTVisitor):
    def __init__(self, type_map: Dict[ASTNode, DSLType]):
        self.type_map = type_map
        self.was_optimized = False

    def optimize(self, node: ASTNode) -> ASTNode:
        return self.visit(node)

    def visit_BinaryOpNode(self, node: BinaryOpNode) -> ASTNode:
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        if isinstance(node.left, LiteralNode) and isinstance(node.right, LiteralNode):
            if self.type_map.get(node.left) == IntType() and self.type_map.get(node.right) == IntType():
                left_val, right_val = int(node.left.value), int(node.right.value)
                new_val = None
                if node.op == '+': new_val = left_val + right_val
                elif node.op == '-': new_val = left_val - right_val
                elif node.op == '*': new_val = left_val * right_val
                
                if new_val is not None:
                    self.was_optimized = True
                    return LiteralNode(start_pos=node.start_pos, end_pos=node.end_pos, value=new_val, type_name="Int", inferred_type=IntType())
        return node
    
    def generic_visit(self, node: ASTNode) -> ASTNode:
        for attr_name in dir(node):
            if attr_name.startswith('_') or attr_name == 'parent': continue
            attr_value = getattr(node, attr_name)
            if isinstance(attr_value, ASTNode):
                setattr(node, attr_name, self.visit(attr_value))
            elif isinstance(attr_value, list):
                new_list = []
                for item in attr_value:
                    if isinstance(item, ASTNode):
                        new_list.append(self.visit(item))
                    else:
                        new_list.append(item)
                setattr(node, attr_name, new_list)
        return node

class DSLPrettyPrinter(ASTVisitor):
    def __init__(self):
        self.indent_level = 0

    def print(self, node: ASTNode) -> str:
        return self.visit(node)
    
    def _indent_str(self) -> str: return "    " * self.indent_level
    def visit_ProgramNode(self, node: ProgramNode) -> str: return "\n\n".join([self.visit(item) for item in node.body])
    def visit_CustomActionDeclNode(self, node: CustomActionDeclNode) -> str:
        params = self.visit(node.params)
        ret_type = f" -> {self.visit(node.return_type)}" if node.return_type else ""
        body = self.visit(node.block)
        return f"def {node.name.name}({params}){ret_type} {{\n{body}{self._indent_str()}}}"

    def visit_BlockNode(self, node: BlockNode) -> str:
        self.indent_level += 1
        body = "".join([f"{self._indent_str()}{self.visit(stmt)}\n" for stmt in node.statements])
        self.indent_level -= 1
        return body

    def visit_ParamDeclNode(self, node: ParamDeclNode) -> str:
        return f"{node.name.name}: {self.visit(node.type_ann)}"
    
    def visit_ParamListNode(self, node: ParamListNode) -> str: return ", ".join([self.visit(p) for p in node.params])
    def visit_TransferNode(self, node: TransferNode) -> str:
        to, amount, state_var = self.visit(node.to), self.visit(node.amount), self.visit(node.state_var)
        return f"transfer({to}, {amount}, {state_var});"

    def visit_BinaryOpNode(self, node: BinaryOpNode) -> str: return f"({self.visit(node.left)} {node.op} {self.visit(node.right)})"
    def visit_IdentifierNode(self, node: IdentifierNode) -> str: return node.name
    def visit_LiteralNode(self, node: LiteralNode) -> str: return str(node.value)
    def visit_TypeNode(self, node: TypeNode) -> str: return node.name
    def generic_visit(self, node: ASTNode) -> str: return f"<{node.__class__.__name__}>"