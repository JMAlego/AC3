#!/usr/bin/env python3
"""AC3 implementation for CONS Open Individual Assessment 2020/2021.

AC3 is implemented simplistically and would require modifications to be used
in a constraint solver.

NB: This script requires Python 3.5 or newer to work.
"""

from collections import deque
from itertools import product
from sys import version_info
from typing import (Callable, Dict, Generic, Hashable, Iterable, Iterator, List, Optional, Set,
                    Tuple, TypeVar)

try:
    from typing import Deque
except ImportError:
    # We're in a version that doesn't support Deque type.
    # Dequeues will still work the type just can't be checked.
    pass

# Types as used in this file only work in python 3.5, if we detect an earlier
# version we print a message and exit with an error. It is worth noting that
# most earlier versions will fail with a SyntaxError instead due to the type
# syntax being unsupported. This is considered a reasonable restriction given
# that version 3.5 was released in 2015.
if version_info < (3, 5):
    print("This script requires Python version >=3.5 to run.")
    exit(1)

# Type var for QueueSet
T = TypeVar('T', bound=Hashable)


class QueueSet(Generic[T]):
    """Represents a queue set."""

    def __init__(self, initial: Optional[Iterable[T]] = None):
        """Initialise queue set with optional initial values."""
        # Dequeue preserves order
        self._queue_order = deque()  # type: Deque[T]

        # Set allows fast checking of uniqueness
        self._queue_set = set()  # type: Set[T]

        # Initial values if present
        if initial is not None:
            self.enqueue_all(initial)

    def enqueue(self, x: T):
        """Enqueue a value into the queue set.

        If value is already in the set initial order is preserved.
        """
        # O(1) check for membership instead of list's O(n)
        if x not in self._queue_set:
            self._queue_set.add(x)
            self._queue_order.append(x)

    def enqueue_all(self, x: Iterable[T]):
        """Enqueue all values from an iterable into the queue set."""
        for v in x:
            self.enqueue(v)

    def dequeue(self) -> T:
        """Dequeue a value from the head of the queue set."""
        # O(1) popleft instead of list's pop(0) which is O(n)
        x = self._queue_order.popleft()
        self._queue_set.remove(x)
        return x

    def empty(self) -> bool:
        """Check if the queue set empty."""
        return len(self._queue_set) == 0


class AC3EmptyDomainException(Exception):
    """Represents an exception caused by a domain being empty during AC3 revise."""

    def __init__(self, empty_domain: str, *args: object) -> None:
        """Initialise exception."""
        super().__init__(*args)
        self.empty_domain = empty_domain


class AC3NodeConsistencyException(Exception):
    """Represents an exception caused by inconsistent node state."""


class AC3:
    """AC3 class for calculating arc consistency using the AC3 algorithm.

    This class would require further methods to be able to work as a constraint
    solver and is optimised for simply applying AC3 not working as a full
    constraint solver.
    """

    def __init__(self) -> None:
        """Initialise variables and constraints."""
        self._variables = {}  # type: Dict[str, Set[int]]
        self._constraints = {}  # type: Dict[str, Dict[str, Set[Tuple[int, int]]]]

    def get_values(self) -> List[List[int]]:
        """Get values as a list of lists for display."""
        return list(sorted(v[1]) for v in sorted(self._variables.items(), key=lambda x: x[0]))

    def add_variable(self, name: str, initial_domain: Iterable[int]):
        """Add a variable with an initial domain."""
        self._variables[name] = set(initial_domain)

    def add_arc(self, x: str, y: str, bidirectional=True):
        """Add an arc between two variables, bidirectional by default."""
        # Add other direction of arc.
        if bidirectional:
            self.add_arc(y, x, bidirectional=False)

        # Calculate all possible satisfying tuples, this represents no
        # constraints in extension. When constraints are added this pool of
        # possible tuples will get smaller.
        satisfying_tuples_set = set(product(self._variables[x], self._variables[y]))

        # Add set to constraints datastructure
        if x not in self._constraints:
            self._constraints[x] = {}
        if y not in self._constraints[x]:
            self._constraints[x][y] = satisfying_tuples_set
        else:
            self._constraints[x][y].update(satisfying_tuples_set)

    def add_constraint(self,
                       x: str,
                       y: str,
                       constraint: Callable[[int, int], bool],
                       bidirectional=True):
        """Add a constraint to arc(x,y).

        Multiple applied constraints will be and-ed together due to the way
        that constraints are stored (in extension).
        """
        # Add other direction of constraint
        if bidirectional:
            self.add_constraint(y, x, constraint, bidirectional=False)

        # By storing the constraints this way we do lose the ability to
        # distinguish which constraint removed a value from the set of
        # satisfying tuples, however, this is more efficient as we only need
        # to check one set of satisfying tuples. We are effectively calculating
        # intersection of the constraints ahead of time instead of during
        # revising.

        # As we go along we store the tuples to remove as mutating iterables
        # while iterating over them is a Bad Idea in python.
        satisfying_tuples_to_remove = set()
        for i, j in self._constraints[x][y]:
            # Run the constraint against the tuple to see if it's satisfies
            if not constraint(i, j):
                satisfying_tuples_to_remove.add((i, j))
        # Actually remove the tuples that do not satisfy the constraint.
        self._constraints[x][y].difference_update(satisfying_tuples_to_remove)

    def assign_value(self, x: str, v: int):
        """Assign a value to a variable."""
        self._variables[x] = set([v])

    def get_arcs(self,
                 x: Optional[str] = None,
                 y: Optional[str] = None) -> Iterator[Tuple[str, str]]:
        """Get all arcs with optional constraint on x or y value."""
        # Variable order is used to make the trace more intuitive to read.
        for a in sorted(self._constraints.keys()):
            if a == x or x is None:
                for b in sorted(self._constraints[a].keys()):
                    if b == y or y is None:
                        yield (a, b)

    def _node_consistent(self, node: str) -> bool:
        """Check for node consistency of the specified node."""
        for arc in self.get_arcs(x=node):
            x, y = arc
            supported = False
            for d_i, d_j in product(self._variables[x], self._variables[y]):
                if (d_i, d_j) in self._constraints[x][y]:
                    supported = True
                    break
            if not supported:
                return False

        return True

    def ac3(self):
        """Run AC3."""
        # Check for initial node consistency
        for node in sorted(self._variables):
            if not self._node_consistent(node):
                raise AC3NodeConsistencyException()

        # Queue set of arcs to check
        arcs_queue = QueueSet(self.get_arcs())  # type: QueueSet[Tuple[str, str]]

        # While we still have arcs to check
        while not arcs_queue.empty():
            # Get the arc to check
            x, y = arcs_queue.dequeue()
            print("Revising arc({},{})".format(x, y))

            # Revise arc(x,y)
            if self.revise(x, y):
                # Get new arcs to check
                new_arcs = set(self.get_arcs(y=x))
                # Remove current arc reversed, which we don't need to check
                new_arcs.remove((y, x))
                # Add all arcs to end of queue
                arcs_queue.enqueue_all(new_arcs)

    def revise(self, x, y) -> bool:
        """Revise arc(x, y).

        Returns True when a value is removed from a variable domain, False otherwise.
        """
        changed = False

        # We store unsupported values here because mutating an iterable while
        # iterating over it is a Bad Idea in python (as mentioned above).
        unsupported_variables = set()

        for d_i in self._variables[x]:
            supported = False
            for d_j in self._variables[y]:
                # Checking that all constraints are satisfied is as simple as
                # checking for membership of the satisfying tuples set.
                if (d_i, d_j) in self._constraints[x][y]:
                    supported = True
                    break
            if not supported:
                print("Removed value {} from D({})".format(d_i, x))
                unsupported_variables.add(d_i)
                changed = True

        # Actually remove unsupported values
        self._variables[x].difference_update(unsupported_variables)

        if not self._variables[x]:
            # If any variable's set is empty then we cannot satisfy all our
            # constraints so we early exit with an exception.
            raise AC3EmptyDomainException(x)

        return changed


def test(assignments: Dict[str, int] = dict()):
    """Run a test of the n-queens problem with n=6 and specified assignments."""
    print("Initial assignments: {}".format(", ".join("{}={}".format(*assignment)
                                                     for assignment in assignments.items())))

    ac3 = AC3()

    variables = ("x1", "x2", "x3", "x4", "x5", "x6")

    # Initialise each variable with an initial domain of [1..6].
    for variable in variables:
        ac3.add_variable(variable, (1, 2, 3, 4, 5, 6))

    # Initialise constraints
    for index_x, variable_x in enumerate(variables, start=1):
        for index_y, variable_y in enumerate(variables, start=1):
            if variable_x == variable_y:
                continue
            if index_y > index_x:  # if j>i
                ac3.add_arc(variable_x, variable_y)
                # Rather than manually adding the constraints in extension the
                # add_constraint method will generate the constraints and store
                # them in extension.

                # For simplicity call add_constraint multiple times as this is
                # identical to and-ing the constraints into a single constraint.

                # (𝑥𝑗 ≠ 𝑥𝑖)
                ac3.add_constraint(variable_x, variable_y, lambda i, j: i != j)

                # (𝑥𝑗 ≠ 𝑥𝑖 + (𝑗 − 𝑖))
                ac3.add_constraint(variable_x, variable_y, lambda i, j: j != (i +
                                                                              (index_y - index_x)))

                # (𝑥𝑗 ≠ 𝑥𝑖 − (𝑗 − 𝑖))
                ac3.add_constraint(variable_x, variable_y, lambda i, j: j != (i -
                                                                              (index_y - index_x)))

    # Assign or initial values
    for assignment in assignments.items():
        ac3.assign_value(*assignment)

    try:
        # Actually run AC3!
        ac3.ac3()
    except AC3EmptyDomainException as e:
        # If any domain is empty (has no valid value) then our constraints are
        # impossible to fulfil with the given initial assignments (or sometimes
        # with any initial assignments). If this happens then an exception is
        # thrown to exit early.
        print("Exiting AC3 due to empty D({})".format(e.empty_domain))
    except AC3NodeConsistencyException:
        print("Initial node state inconsistent, exiting!")
        return

    print("Final domains: {}".format(ac3.get_values()))


if __name__ == "__main__":
    # Run tests given in assessment
    test({"x1": 1, "x2": 3})
    print("---")
    test({"x1": 1, "x2": 4})
    print("---")
    test({"x1": 2, "x2": 4, "x3": 6})
