import argparse
import random
from itertools import product


class BayesianNetwork:
    def __init__(self):
        self.variables = {}
        self.initialize_network()

    def initialize_network(self):
        self.variables["B"] = {(True,): 0.001, (False,): 0.999}
        self.variables["E"] = {(True,): 0.002, (False,): 0.998}
        self.variables["A"] = {
            (True, True, True): 0.70,
            (True, True, False): 0.30,
            (True, False, True): 0.01,
            (True, False, False): 0.99,
            (False, True, True): 0.70,
            (False, True, False): 0.30,
            (False, False, True): 0.01,
            (False, False, False): 0.99
        }
        self.variables["J"] = {
            (True, True): 0.9,
            (True, False): 0.1,
            (False, True): 0.05,
            (False, False): 0.95
        }
        self.variables["M"] = {
            (True, True): 0.7,
            (True, False): 0.3,
            (False, True): 0.01,
            (False, False): 0.99
        }

    def fetch_probability(self, cpt, *conditions):
        if tuple(conditions) in cpt:
            prob = cpt[tuple(conditions)]
            return prob
        else:
            return 0.0

    def normalize_dict(self, values, normalize):
        if normalize == 0:
            return {key: 0.0 for key in values}

        normalized = {}
        for key, value in values.items():
            normalized[key] = value / normalize

        return normalized

    def conditional_probability(self, variable, assignment):
        cpt = self.variables[variable]
        if variable == "A":
            conditions = (assignment["B"], assignment["E"], True)
        elif variable in ["J", "M"]:
            conditions = (assignment["A"], True)
        else:
            conditions = (True,)

        prob = self.fetch_probability(cpt, *conditions)
        return prob

    def joint_probability(self, assignment):
        probability = 1.0
        for variable in self.variables:
            prob = self.conditional_probability(variable, assignment)
            probability *= prob if assignment[variable] else (1 - prob)
        return probability

    def exact_inference(self, evidence, nodes):
        allVariable = list(self.variables.keys())
        hiddenVars = []
        for var in self.variables.keys():
            if var not in evidence:
                hiddenVars.append(var)

        totalProb = 0.0
        jointProb = 0.0

        for assignment in product([True, False], repeat=len(hiddenVars)):
            complete_assignment = {**evidence, **dict(zip(hiddenVars, assignment))}
            for var in allVariable:
                if var not in complete_assignment:
                    complete_assignment[var] = False 

            prob = self.joint_probability(complete_assignment)
            totalProb += prob

            if all(complete_assignment.get(node, False) for node in nodes):
                jointProb += prob

        return {",".join(nodes): jointProb / totalProb}

    def create_sample(self):
        sample = {}
        for variable in self.variables:
            prob = self.conditional_probability(variable, sample)
            sample[variable] = random.random() < prob
        return sample

    def sample_or_evidence(self, variable, evidence, sample):
        if variable in evidence:
            value = evidence[variable]
            prob = self.conditional_probability(variable, {**sample, variable: value})
        else:
            prob = self.conditional_probability(variable, sample)
            value = random.random() < prob
        return value, prob

    def weighted_sample(self, evidence):
        weight = 1.0
        sample = {}

        for variable in self.variables:
            value, prob = self.sample_or_evidence(variable, evidence, sample)
            sample[variable] = value
            if variable in evidence:
                weight *= prob if value else (1 - prob)

        return sample, weight

    def unified_inference(self, algorithm, sampleCount, evidence, nodes, runs=1):
        if algorithm == "exact":
            return self.exact_inference(evidence, nodes)

        total_results = {node: 0.0 for node in nodes}

        for _ in range(runs):
            samples = []
            weights = []

            for _ in range(sampleCount):
                if algorithm in {"prior", "rejection"}:
                    sample = self.create_sample()
                    if all(sample.get(var) == val for var, val in evidence.items()):
                        samples.append(sample)
                elif algorithm == "likelihood":
                    sample, weight = self.weighted_sample(evidence)
                    samples.append(sample)
                    weights.append(weight)

            if algorithm == "likelihood":
                runResult = self.weighted_probabilities(samples, weights, nodes)
            else:
                runResult = self.query_probabilities(samples, nodes)

            for node in nodes:
                total_results[node] += runResult[node]

        return self.normalize_dict(total_results, runs)

    def query_probabilities(self, samples, nodes):
        nodeCounts = {node: 0 for node in nodes}
        for sample in samples:
            for node in nodes:
                if sample[node]:
                    nodeCounts[node] += 1

        totalSamples = len(samples)

        return self.normalize_dict(nodeCounts, totalSamples)

    def weighted_probabilities(self, samples, weights, nodes):
        totalWeight = sum(weights)
        if totalWeight == 0:
            return {node: 0.0 for node in nodes}

        weightedSums = {node: 0.0 for node in nodes}
        for sample, weight in zip(samples, weights):
            for node in nodes:
                if sample[node]:
                    weightedSums[node] += weight

        return self.normalize_dict(weightedSums, totalWeight)


def parse_input(inputString):
    condition, separate = inputString.strip("[]").split("][")
    evidence = {}
    for item in condition.split("><"):
        variable, value = item.strip("<>").split(",")
        evidence[variable.strip()] = value.strip() == "t"

    nodes = [node.strip() for node in separate.split(",")]
    return evidence, nodes


def main():
    parser = argparse.ArgumentParser(description="Bayesian Network")
    parser.add_argument("input", type=str, help="Input string")
    parser.add_argument("--sample_count", type=int, default=10000, help="Number of samples")
    parser.add_argument("--algorithm", type=str, required=True, choices=["prior", "rejection", "likelihood", "exact"], help="Inference algorithm")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    args = parser.parse_args()

    network = BayesianNetwork()
    evidence, nodes = parse_input(args.input)

    result = network.unified_inference(args.algorithm, args.sample_count, evidence, nodes, runs=args.runs)

    print(f"Algorithm: {args.algorithm.capitalize()}")

    if ",".join(nodes) in result:
        print(f"[<{",".join(nodes)},{result[",".join(nodes)]:.6f}>]")
    else:
        result_string = "".join(f"[<{node},{result[node]:.6f}>]" for node in nodes)
        print(result_string)



if __name__ == "__main__":
    main()
