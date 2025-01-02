# MaPyReduce: A Python-based Local MapReduce Framework

## Overview

**MaPyReduce** is a local implementation of the MapReduce computational model. It offers a lightweight framework for processing data through mapper and reducer functions, allowing Python developers to build scalable data workflows. This framework emulates the core principles of the original distributed MapReduce model, but operates locally, taking advantage of Python's multiprocessing capabilities using the `multiprocess` library.

## Key Features

- **Flexible Mapper and Reducer Design**: Supports chaining multiple mapper functions and a final reducer function.
- **Command Design Pattern**: Encapsulation of mapper and reducer operations for loose coupling and reusability.
- **Finite State Machine (FSM) Inspiration**: Each step in the chain represents a state transition, culminating in a final accepting state.
- **Multiprocessing Support**: Leverages multi-core processors to perform map operations in parallel.
- **Builder Pattern for Chain Construction**: Simplifies the creation of complex workflows using a fluent API.

## Source Code Structure

### ChainReducer

The `ChainReducer` class is the core component of the framework. It manages the flow of data through a sequence of mappers and applies a final reducer to consolidate results.

#### Key Methods
- `add_data(data_tuple)`: Passes the given data to the ChainReducer. **It must be a tuple, since they will be arguments to the first Mapper instance call**.
- `add_mapper(map_function)`: Appends a mapper to the chain.
- `set_reducer(reducer)`: Sets the final reducer.
- `run()`: Executes the entire chain of mappers and reducers.
- `run_step()`: Executes a single step in the chain.
- `reset()`: Resets the state of the chain.

### Protocols

- **MapperService**: Defines the interface for mapper functions.
  - Properties:
    - `data`: The input data for the mapper.
  - Methods:
    - `run()`: Performs the mapping operation. **Parallelism must be implemented at the level of each Mapper instance's `run` method, utilizing native Python libraries or the `multiprocess` library for efficiency.**

- **ReducerService**: Defines the interface for reducer functions.
  - Properties:
    - `data`: The input data for the reducer.
  - Methods:
    - `run()`: Performs the reduction operation.

- **Consumer**: Implements the `ReducerService` protocol. This class consolidates results from previous mapper outputs or directly from a list of tuples. 
  - Properties:
    - `data`: Retrieves data produced by the previous mapper or stored list of tuples.
  - Methods:
    - `run()`: Consolidates the data and returns the result.


## Usage

### Example: Chaining Mappers and Reducers

Sample implementations of the Mapper and Reducer services are provided in the `TestIntegerChainReducer` class. More specifically, we provide:
- `Integer.FromInt`: Converts a list of integers into custom `Integer` objects. This is typically the first step in the MapReduce process.
- `Integer.Square`: Squares the values of `Integer` objects. This represents a transformation stage in the chain.
- `Integer.ToList`: Extracts the values from `Integer` objects into a plain Python list.
- `Integer.Sum`: Sums up the values of `Integer` objects into a single `Integer`.

Below is a step-by-step example demonstrating how to use the framework:

```python
from mapyreduce import ChainReducer, Integer

chain_reducer = ChainReducer() \
    .add_data(([2, 5, 7, 9],)) \ # Please note that data must be packed in a tuple.
    .add_mapper(Integer.FromInt) \
    .add_mapper(Integer.Square) \
    .set_reducer(Integer.ToList)

result = chain_reducer.run()
print(result)  # Output: [4, 25, 49, 81]
```

### Builder Method for Simplicity

The same workflow can be implemented using the `build_with` factory method:

```python
result = ChainReducer.build_with(
    chain_map=[Integer.FromInt, Integer.Square],
    reducer=Integer.ToList,
    map_args=([2, 5, 7, 9],)
).run()

print(result)  # Output: [4, 25, 49, 81]
```

### Step-by-step Execution

The framework also supports step-by-step execution:

```python
chain_reducer.run_step()  # Executes the first mapper.
chain_reducer.run_step()  # Executes the second mapper.
chain_reducer.run_step()  # Applies the reducer.
```

## Testing

A sample implementation of the computational framework is provided in the `TestIntegerChainReducer` class, which demonstrates:
- Batch execution of the entire MapReduce chain.
- Step-by-step execution.
- Using the builder method to construct and execute a chain.

Run the tests using `pytest`:

```bash
pytest mapyreduce.py
```

## Acknowledgments

This implementation draws inspiration from the following:
- **Apache Hadoop**: ChainReducer class design ([Apache Hadoop Documentation](https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/mapreduce/lib/chain/ChainReducer.html)).
- **Joshua Bloch's Effective Java**: Builder design pattern insights.
- **Finite State Machines (FSM)**: Abstraction of chain operations as state transitions.
- **Command Design Pattern**: Encapsulation of mapper and reducer operations.

---

**Disclaimer**: While this framework mimics the distributed MapReduce design, it operates locally. For large-scale distributed processing, consider frameworks like Apache Hadoop or Apache Spark.
