# Bayesian Network Inference Program

This repository contains a Python-based program that performs inference on a Bayesian network using various algorithms. It is designed to calculate probabilities for query nodes based on given evidence, supporting both exact and sampling-based inference methods.

## Features

- **Exact Inference**
- **Prior Sampling**
- **Rejection Sampling**
- **Likelihood Weighting**

The program is highly configurable, allowing users to define evidence, query nodes, and parameters such as the number of samples and runs.

---

## Usage

### Command Format
```bash
python script.py "[<evidence1,value1><evidence2,value2>][query_node1,query_node2]" --algorithm <algorithm> --sample_count <count> --runs <num_runs>
```

### Input Details
1. **Evidence**:
   - Specified in the format `<node,value>`.
   - Multiple pieces of evidence are enclosed in `[]` and separated by `><`.

2. **Query Nodes**:
   - A comma-separated list enclosed in `[]`.

3. **Algorithm**:
   - Specifies the inference algorithm:
     - `exact`: Performs exact inference.
     - `prior`: Uses prior sampling.
     - `rejection`: Uses rejection sampling.
     - `likelihood`: Uses likelihood weighting.

4. **Sample Count (`--sample_count`)**:
   - Number of samples to generate for sampling algorithms. Default: 10,000.

5. **Runs (`--runs`)**:
   - Number of iterations to run the algorithms. Default: 1.

---

## Examples

### 1. Exact Inference
Run Exact Inference for Case 1:
```bash
python script.py "[<A,f>][B,J]" --algorithm exact
```

### 2. Prior Sampling
Run Prior Sampling for Case 2:
```bash
python script.py "[<J,t><E,f>][B,M]" --algorithm prior --sample_count 10000 --runs 1
```

### 3. Rejection Sampling
Run Rejection Sampling for Case 3:
```bash
python script.py "[<M,t><J,f>][B,E]" --algorithm rejection --sample_count 10 --runs 5
```

### 4. Likelihood Weighting
Run Likelihood Weighting for Case 1:
```bash
python script.py "[<A,f>][B,J]" --algorithm likelihood --sample_count 1000 --runs 10
```

---

## Output Format
The output is formatted as:
```
<node1,probability1><node2,probability2>...
```

### Example
```bash
<B,0.0010><J,0.0500>
```

---

## Code Structure
- **`script.py`**: Main script implementing the Bayesian Network and inference methods.
- **Core Classes and Functions**:
  - `BayesianNetwork`: Defines the Bayesian network structure and CPTs.
  - `unified_inference`: Implements different inference algorithms.
  - `parse_input`: Parses user input for evidence and query nodes.

---

## Results Summary
Detailed results and performance comparisons of the algorithms are documented in the Report file**.

---

## Author
**Vishvendra Reddy Bhoomidi**  
Master's Student in Artificial Intelligence, University of Michigan-Dearborn

---

Feel free to open an issue or contribute to this repository for enhancements or bug fixes!
