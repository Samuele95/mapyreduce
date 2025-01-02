from __future__ import annotations
from typing import List, Protocol, Tuple, Any, runtime_checkable


class ChainReducer:
    """
    ChainReducer is a builder class for creating and executing a chain of Mapper and Reducer functions.
    It manages the flow of data through multiple mapping steps and finally applies a reducer to consolidate results.

    This class follows the Builder design pattern, as described by Joshua Bloch in "Effective Java" (https://www.oreilly.com/library/view/effective-java-3rd/9780134686097/).
    Additionally, it is inspired by Apache Hadoop's ChainReducer class (https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/mapreduce/lib/chain/ChainReducer.html),
    which allows multiple Mapper instances to be chained together before applying a final Reducer.

    The design is also inspired by Finite State Machines (FSM). Each call to `run_step` represents an abstraction of a transition function in an FSM.
    The intermediate result, stored in the `_state` field, represents the current state of the FSM. The transition through different mappers simulates
    state progression, with the final reducer acting as the accepting state or final transition.

    The design of mapper and reducer services adheres to the Command design pattern (https://refactoring.guru/design-patterns/command),
    encapsulating operations as objects to promote loose coupling and reusability.
    """

    def __init__(self):
        """
        Initializes the ChainReducer with empty chain map, no reducer, and default state values.
        The `_state` field represents the current state in the FSM-inspired process.
        """
        self._chain_map = []
        self._reducer = None
        self._map_args = None
        self._state = None
        self._chain_step = 0

    @classmethod
    def build_with(cls, chain_map: List, reducer, map_args) -> ChainReducer:
        """
        Factory method to build a ChainReducer with an initial chain map, reducer, and mapping arguments.

        Args:
            chain_map (List): A list of mapper functions to be applied in sequence.
            reducer (ReducerService): The reducer function to apply after all mappers.
            map_args (Tuple): Initial arguments for the first mapper.

        Returns:
            ChainReducer: Configured ChainReducer instance.
        """
        chain_reducer = ChainReducer().add_data(map_args)
        for mapper in chain_map:
            chain_reducer.add_mapper(mapper)
        return chain_reducer.set_reducer(reducer)

    def add_data(self, data: Tuple) -> ChainReducer:
        """
        Adds initial data to the chain and sets it as the starting state if not already set.
        This represents the initial state of the FSM process.

        Args:
            data (Tuple): Initial data tuple to be passed to the mappers.

        Returns:
            ChainReducer: The ChainReducer instance for method chaining.
        """
        self._map_args = data
        if self._state is None:
            self._state = data
        return self

    def add_mapper(self, map_function: MapperService) -> ChainReducer:
        """
        Appends a mapper function to the chain map, representing a transition function in the FSM.

        Args:
            map_function (MapperService): Mapper function to be added to the chain.

        Returns:
            ChainReducer: The ChainReducer instance for method chaining.
        """
        self._chain_map.append(map_function)
        return self

    def set_reducer(self, reducer: ReducerService) -> ChainReducer:
        """
        Sets the reducer function for the chain, representing the final state or accepting state in the FSM.

        Args:
            reducer (ReducerService): Reducer function to finalize the chain process.

        Returns:
            ChainReducer: The ChainReducer instance for method chaining.
        """
        self._reducer = reducer
        return self

    @property
    def state(self):
        """
        Retrieves the current state of the ChainReducer, representing the FSM state.

        Returns:
            The current state after mapper/reducer application.
        """
        return self._state

    def run_step(self):
        """
        Executes one step of the map or reduce process depending on the current chain step.
        Each call simulates a state transition in an FSM, updating the `_state` with intermediate results.

        Returns:
            The result of the current step, representing the FSM's current state.

        Raises:
            ValueError: If there is an error during the chain operation.
        """
        try:
            if len(self._chain_map) > 0:
                if self._chain_step == 0:
                    # We are applying the first mapper with raw data, if available
                    self._state = self._chain_map[self._chain_step](*self._map_args).run()
                    self._chain_step += 1
                    return self._state
                if self._chain_step <= len(self._chain_map) - 1:
                    # We are applying an inner mapper, if available
                    self._state = self._chain_map[self._chain_step](self._state).run()
                    self._chain_step += 1
                    return self._state
                if self._chain_step == len(self._chain_map):
                    # Else, we are applying the reducer
                    self._state = self._reducer(self._state).run()
                    self._chain_step += 1
                    return self._state
            # If the reducer had already been applied, just return its result.
            # The result may be the final Reducer data or a (inner) Mapper instance.
            # If we have a Mapper instance, we are retrieving the intermediate computation data.
            return self._state
        except Exception as e:
            raise ValueError("Error in MapReduce chains operation")

    def run(self):
        """
        Executes the entire chain of mappers and reducer in sequence, simulating a complete FSM run.

        Returns:
            The final result after the reducer application, representing the final FSM state.

        Raises:
            ValueError: If no reducer is provided or if an error occurs during execution.
        """
        try:
            if self._reducer is None:
                raise ValueError("You must provide a reducer at least!")
            # Batch run the whole MapReduce chain.
            result = self._state
            if len(self._chain_map) == 0:
                # If the MapReduce chain has a Reducer only, just return its output
                result = self.run_step()
                self._reset()
                return result
            while self._chain_step <= len(self._chain_map):
                # Otherwise, compute the whole MapperChain output and lastly apply the Reducer
                result = self.run_step()
            self._reset()
            return result
        except Exception as e:
            raise ValueError("Error in MapReduce chains operation")

    def reset(self) -> ChainReducer:
        """
        Resets the chain reducer to its initial state.

        Returns:
            ChainReducer: The ChainReducer instance for method chaining.
        """
        self._reset()
        return self

    def _reset(self) -> None:
        """
        Internal method to reset chain step and state.
        """
        self._chain_step = 0
        self._state = None

@runtime_checkable
class MapperService(Protocol):
    """
    Protocol for defining a Mapper service with data property and run method.
    Implements the Command design pattern, encapsulating mapping logic within a command-like structure.
    The design of mapper and reducer services adheres to the Command design pattern (https://refactoring.guru/design-patterns/command),
    encapsulating operations as objects to promote loose coupling and reusability.
    """

    @property
    def data(self) -> List[Any]:
        ...

    def run(self) -> List[Tuple[Any, Any]]:
        ...

@runtime_checkable
class ReducerService(Protocol):
    """
    Protocol for defining a Reducer service with data property and run method.
    Implements the Command design pattern, encapsulating reduction logic within a command-like structure.

    """

    @property
    def data(self) -> List[Tuple[Any, Any]]:
        ...

    def run(self) -> Any:
        ...

class Consumer(ReducerService):
    """
    Consumer class that implements the ReducerService interface.
    It consumes MapperService output or list of tuples and consolidates results.
    """

    def __init__(self, int_list: List[Tuple[Any, Any]] | MapperService):
        """
        Initializes the Consumer with initial data or a mapper service.

        Args:
            int_list (List[Tuple[Any, Any]] | MapperService): Initial data or a mapper instance.
        """
        self._prev_data = int_list

    @property
    def data(self) -> List[Any]:
        """
        Retrieves the data from the previous mapper or directly from stored data.

        Returns:
            List[Any]: Data produced by the previous mapper or the stored list of tuples.
        """
        return self._prev_data if not isinstance(self._prev_data, MapperService) else self._prev_data.run()

    def run(self) -> Any:
        """
        Executes the consumer logic and returns consolidated data.

        Returns:
            Any: The result of consuming the data.
        """
        return self.data