
from __future__ import annotations

from typing import Set, Tuple, Iterable, List
from dataclasses import dataclass

from graph import DataflowGraph, FunctionArgument, Channel, ProcessingElement


class Term: ...


@dataclass
class EmptyPermission(Term):
    def __str__(self):
        return "0"


@dataclass
class ReadPermission(Term):
    heap_object: str

    def __str__(self):
        return f"read {self.heap_object}"


@dataclass
class WritePermission(Term):
    heap_object: str

    def __str__(self):
        return f"write {self.heap_object}"


@dataclass
class PermissionVariable(Term):
    """
    Permission for a channel
    """
    channel_id: int

    def __str__(self):
        return f"p{self.channel_id}"


@dataclass
class DisjointUnion(Term):
    left: Term
    right: Term

    def __str__(self):
        return f"{self.left} + {self.right}"

    @staticmethod
    def of(first: Term, *rest: Term) -> DisjointUnion:
        for term in rest:
            first = DisjointUnion(first, term)

        return first


class Formula: ...


@dataclass
class Equality(Formula):
    left: Term
    right: Term

    def __str__(self):
        return f"{self.left} = {self.right}"


@dataclass
class Inclusion(Formula):
    """
    Permission left is included in permission right
    """
    left: Term
    right: Term

    def __str__(self):
        return f"{self.left} ⊑ {self.right}"


@dataclass
class Disjoint(Formula):
    """
    The given permission terms are disjoint
    """
    terms: Tuple[Term, ...]

    def __str__(self):
        return f"disjoint({', '.join(map(str, self.terms))})"


class MemoryPermissionSolver:
    @staticmethod
    def get_sum_of_channel_permissions(channels: Iterable[Channel]) -> Term:
        return DisjointUnion.of(*map(lambda c: PermissionVariable(c.id), channels))

    @staticmethod
    def get_heap_object_of_memory_operator(pe: ProcessingElement) -> str:
        # TODO: here I'm making an assumption that array accesses
        # would always have the first input as the array variable
        assert len(pe.inputs) == 3
        array_input = pe.inputs[0]
        assert isinstance(array_input.constant, FunctionArgument)
        return array_input.constant.variable_name

    @staticmethod
    def generate_constraints(graph: DataflowGraph) -> Tuple[Formula, ...]:
        heap_objects: Set[str] = set() # names of heap objects

        # we assume that these heap objects do not alias
        # i.e. when they are declared in the source file
        # they should have the "restrict" modifier

        for pe in graph.vertices:
            if pe.operator == "MEM_CFG_OP_LOAD" or pe.operator == "MEM_CFG_OP_STORE":
                heap_objects.add(MemoryPermissionSolver.get_heap_object_of_memory_operator(pe))

        # Constraints:
        # 1. (Affinity) for each operator, the sum of input permissions contains the sum of output permissions,
        #    and the input permissions are disjoint.
        # 2. (Read permission) for a load operator on heap object A, (read A) must be in one of the input permissions.
        # 3. (Write permission) for a store operator on heap object B, (write A) must be in one of the input permissions.
        # 4. Channels with hold=true (infinite number of constant values) must have empty permission.
        #    NOTE: this condition might be too strong
        # 5. Channels with a single initial constant value should have mutually disjoint memory permissions.

        print("found heap objects", heap_objects)

        constraints: List[Formula] = []

        for pe in graph.vertices:
            input_sum = MemoryPermissionSolver.get_sum_of_channel_permissions(pe.inputs)
            output_sum = MemoryPermissionSolver.get_sum_of_channel_permissions(sum(pe.outputs.values(), ()))

            constraints.append(Inclusion(output_sum, input_sum))

            if pe.operator == "MEM_CFG_OP_LOAD":
                heap_object = MemoryPermissionSolver.get_heap_object_of_memory_operator(pe)
                constraints.append(Inclusion(WritePermission(heap_object), input_sum))
                
            elif pe.operator == "MEM_CFG_OP_STORE":
                heap_object = MemoryPermissionSolver.get_heap_object_of_memory_operator(pe)
                constraints.append(Inclusion(WritePermission(heap_object), input_sum))

        for channel in graph.channels:
            if channel.hold:
                constraints.append(Equality(PermissionVariable(channel.id), EmptyPermission()))

        constant_channel_vars: List[PermissionVariable] = []

        for channel in graph.channels:
            if channel.constant is not None and not channel.hold:
                constant_channel_vars.append(PermissionVariable(channel.id))

        constraints.append(Disjoint(constant_channel_vars))

        for constraint in constraints:
            print(constraint)
