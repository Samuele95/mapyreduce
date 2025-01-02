from __future__ import annotations
from dataclasses import dataclass
from functools import total_ordering
from multiprocessing import cpu_count
from typing import List, Any, Tuple
from multiprocess.pool import ThreadPool
from mapyreduce.mapyreduce import MapperService, ReducerService, ChainReducer, Consumer
import pytest

@dataclass(frozen=True)
@total_ordering
class Integer:
    """
    Integer class wraps an integer value and supports equality and ordering comparisons.
    This class is used to demonstrate the mapping and reducing process.
    """
    value: int

    def __eq__(self, other):
        if not isinstance(other, Integer):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, Integer):
            return NotImplemented
        return self.value < other.value

    def __str__(self) -> str:
        return f"Integer[{self.value}]"

    class FromInt(MapperService):
        """
        MapperService implementation that converts a list of integers to Integer objects.
        This class represents the first step in the MapReduce chain.
        """
        def __init__(self, int_list: List[Tuple[int,Integer]] | MapperService):
            self._prev_data = int_list

        @property
        def data(self) -> List[Any]:
            return self._prev_data if not isinstance(self._prev_data, MapperService) else self._prev_data.run()

        def run(self) -> List[Tuple[Any, Any]]:
            """
            Converts integers to Integer objects in parallel using ThreadPool.

            Returns:
                List of tuples where each tuple contains the original integer and its Integer wrapper.
            """
            with ThreadPool(cpu_count()) as pool:
                return pool.map(lambda x: (x, Integer(x)), self.data)

    class Square(MapperService):
        """
        MapperService implementation that squares the Integer values.
        This represents the second step in the MapReduce chain.
        """
        def __init__(self, int_list: List[Tuple[int,Integer]] | MapperService):
            self._prev_data = int_list

        @property
        def data(self) -> List[Any]:
            return self._prev_data if not isinstance(self._prev_data, MapperService) else self._prev_data.run()

        def run(self) -> List[Tuple[Any, Any]]:
            """
            Squares each Integer object in the list in parallel.

            Returns:
                List of tuples containing the original integer and its squared value wrapped in Integer.
            """
            with ThreadPool(cpu_count()) as pool:
                return pool.map(lambda x: (x[0], Integer(x[1].value ** 2)), self.data)

    class ToList(ReducerService):
        """
        ReducerService implementation that extracts the Integer values and returns them as a list.
        This class represents the final reduction step to produce a list of results.
        """
        def __init__(self, int_list: List[Tuple[int,Integer]] | MapperService):
            self._prev_data = int_list

        @property
        def data(self) -> List[Any]:
            return self._prev_data if not isinstance(self._prev_data, MapperService) else self._prev_data.run()

        def run(self) -> Any:
            """
            Extracts the Integer values and returns them as a list.

            Returns:
                List of integer values extracted from the Integer objects.
            """
            return [v.value for k,v in self.data]

    class Sum(ReducerService):
        """
        ReducerService implementation that sums up the Integer values.
        This represents an alternative reduction step to produce a single sum.
        """
        def __init__(self, int_list: List[Tuple[int,Integer]] | MapperService):
            self._prev_data = int_list

        @property
        def data(self) -> List[Any]:
            return self._prev_data if not isinstance(self._prev_data, MapperService) else self._prev_data.run()

        def run(self) -> Any:
            """
            Sums up the Integer values from the tuples.

            Returns:
                Integer object representing the sum of all values.
            """
            return Integer(sum([v.value for k,v in self.data]))

class TestIntegerChainReducer:
    """
    Test suite for the ChainReducer class using Integer mapper and reducer services.
    """
    chain_reducer = ChainReducer() \
               .add_data(([2,5,7,9],)) \
               .add_mapper(Integer.FromInt) \
               .add_mapper(Integer.Square) \
               .set_reducer(Integer.ToList)

    def test_batch_run(self):
        """
        Tests the full MapReduce chain in one go using batch processing.
        """
        assert self.chain_reducer.run() == [4, 25, 49, 81]

    def test_step_run(self):
        """
        Tests the step-by-step execution of the MapReduce chain.
        Each call to run_step() applies the next stage in the chain.
        """
        assert self.chain_reducer.run_step() == [(2, Integer(2)), (5, Integer(5)), (7, Integer(7)), (9, Integer(9))]
        assert self.chain_reducer.run_step() == [(2, Integer(4)), (5, Integer(25)), (7, Integer(49)), (9, Integer(81))]
        assert self.chain_reducer.run_step() == [4, 25, 49, 81]
        assert self.chain_reducer.run_step() == [4, 25, 49, 81]

    def test_builder(self):
        """
        Tests building and executing a ChainReducer using the build_with factory method.
        """
        assert ChainReducer.build_with(
            chain_map= [
                Integer.FromInt,
                Integer.Square
            ],
            reducer= Integer.ToList,
            map_args=([2,5,7,9],)
        ).run() == [4, 25, 49, 81]

    def test_step_run_consumer_only(self):
        """
        Tests the step execution of a ChainReducer containing a Consumer of the original data only.
        """
        assert ChainReducer().add_data(([2,5,7,9],)).set_reducer(Consumer).run_step() == ([2,5,7,9],)

    def test_consumer(self):
        """
        Tests building and executing a batch_run of a ChainReducer containing a Consumer of the original data only.
        """
        assert ChainReducer().add_data(([2,5,7,9],)).set_reducer(Consumer).run() == ([2,5,7,9],)

    def test_reducer(self):
        """
        Tests the full MapReduce chain in one go using batch processing, where the Reducer instance acts as a Fold
        operation over the working data.
        """
        assert ChainReducer() \
            .add_data(([2, 5, 7, 9],)) \
            .add_mapper(Integer.FromInt) \
            .add_mapper(Integer.Square) \
            .set_reducer(Integer.Sum).run() == Integer(159)

    def test_error_run(self):
        """
        Checks for raised Exceptions when no data and/or no Reducer objects are given to the ChainReducer.
        """
        with pytest.raises(ValueError):
            ChainReducer().run()
        with pytest.raises(ValueError):
            ChainReducer().add_mapper(Integer.FromInt).run()
