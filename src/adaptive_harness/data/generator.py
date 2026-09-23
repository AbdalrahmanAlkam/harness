"""Synthetic training data generator for all agent strategies with rich natural variations."""

from __future__ import annotations

import random
from typing import List, Tuple


class SyntheticDataGenerator:
    """Generates varied synthetic task examples for routing classification."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_all(self, samples_per_class: int = 800) -> List[Tuple[str, str]]:
        """Generates a balanced dataset containing tasks across all strategies."""
        data: List[Tuple[str, str]] = []
        data.extend([(text, "arithmetic") for text in self.generate_arithmetic(samples_per_class)])
        data.extend([(text, "equations") for text in self.generate_equations(samples_per_class)])
        data.extend([(text, "algorithms") for text in self.generate_algorithms(samples_per_class)])
        data.extend([(text, "text_stats") for text in self.generate_text_stats(samples_per_class)])
        data.extend([(text, "fallback") for text in self.generate_fallback(samples_per_class)])
        self.rng.shuffle(data)
        return data

    def generate_arithmetic(self, count: int) -> List[str]:
        items: List[str] = []
        ops = ["+", "-", "*", "/", "//", "%", "**"]
        phrases = [
            "calculate {expr}",
            "compute {expr}",
            "what is {expr}",
            "evaluate {expr}",
            "find the value of {expr}",
            "how much is {expr}",
            "{expr}",
            "please calculate: {expr}",
            "can you compute {expr}?",
        ]

        while len(items) < count:
            mode = self.rng.choice(["expr", "word_op", "gcd", "lcm", "mod", "factor", "power", "sqrt"])
            if mode == "expr":
                a = self.rng.randint(1, 100)
                b = self.rng.randint(1, 100)
                op = self.rng.choice(ops)
                if op == "/" or op == "//" or op == "%":
                    b = self.rng.randint(1, 25)
                    a = b * self.rng.randint(1, 20) + (self.rng.randint(0, b - 1) if op == "%" else 0)
                if op == "**":
                    a = self.rng.randint(2, 12)
                    b = self.rng.randint(2, 5)
                expr = f"{a} {op} {b}"
                if self.rng.random() < 0.4:
                    c = self.rng.randint(1, 50)
                    op2 = self.rng.choice(["+", "-", "*"])
                    expr = f"({expr}) {op2} {c}" if self.rng.random() < 0.5 else f"{expr} {op2} {c}"
                template = self.rng.choice(phrases)
                items.append(template.format(expr=expr))

            elif mode == "word_op":
                a = self.rng.randint(1, 100)
                b = self.rng.randint(1, 100)
                wop = self.rng.choice(["plus", "minus", "times", "multiplied by", "divided by"])
                if wop == "divided by":
                    b = self.rng.randint(1, 20)
                    a = b * self.rng.randint(1, 15)
                text = self.rng.choice([
                    f"what is {a} {wop} {b}",
                    f"calculate {a} {wop} {b}",
                    f"add {a} and {b}" if wop == "plus" else f"multiply {a} by {b}",
                    f"find {a} {wop} {b}",
                ])
                items.append(text)

            elif mode == "gcd":
                a = self.rng.randint(2, 200)
                b = self.rng.randint(2, 200)
                items.append(self.rng.choice([
                    f"gcd of {a} and {b}",
                    f"gcd({a}, {b})",
                    f"what is the greatest common divisor of {a} and {b}",
                    f"find gcd of {a} and {b}",
                    f"calculate gcd({a}, {b})",
                ]))

            elif mode == "lcm":
                a = self.rng.randint(2, 60)
                b = self.rng.randint(2, 60)
                items.append(self.rng.choice([
                    f"lcm of {a} and {b}",
                    f"lcm({a}, {b})",
                    f"what is the least common multiple of {a} and {b}",
                    f"find lcm of {a} and {b}",
                    f"calculate lcm({a}, {b})",
                ]))

            elif mode == "mod":
                a = self.rng.randint(10, 500)
                b = self.rng.randint(2, 50)
                items.append(self.rng.choice([
                    f"{a} mod {b}",
                    f"{a} modulo {b}",
                    f"what is {a} mod {b}?",
                    f"calculate the remainder when {a} is divided by {b}",
                    f"compute {a} % {b}",
                ]))

            elif mode == "factor":
                n = self.rng.choice([
                    self.rng.randint(10, 5000),
                    self.rng.randint(2, 50) * self.rng.randint(2, 50) * self.rng.choice([2, 3, 5, 7, 11]),
                ])
                items.append(self.rng.choice([
                    f"factor {n}",
                    f"prime factors of {n}",
                    f"find the prime factorization of {n}",
                    f"decompose {n} into prime factors",
                    f"what are the prime factors of {n}?",
                ]))

            elif mode == "power":
                a = self.rng.randint(2, 30)
                items.append(self.rng.choice([
                    f"what is {a} squared",
                    f"{a} squared",
                    f"calculate {a} squared",
                    f"what is {a} cubed",
                    f"cube of {a}",
                    f"{a} to the power of {self.rng.randint(2, 6)}",
                ]))

            elif mode == "sqrt":
                sq = self.rng.randint(2, 40) ** 2
                items.append(self.rng.choice([
                    f"sqrt of {sq}",
                    f"square root of {sq}",
                    f"calculate sqrt({sq})",
                    f"what is the square root of {sq}?",
                    f"find sqrt of {sq}",
                ]))

        return items[:count]

    def generate_equations(self, count: int) -> List[str]:
        items: List[str] = []
        vars_ = ["x", "y", "z", "t", "w", "n"]

        while len(items) < count:
            var = self.rng.choice(vars_)
            mode = self.rng.choice(["linear", "linear_word", "quadratic", "quadratic_roots"])

            if mode == "linear":
                a = self.rng.randint(1, 15)
                b = self.rng.randint(-30, 30)
                c = self.rng.randint(-50, 50)
                b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
                eq = f"{a}*{var} {b_str} = {c}" if self.rng.random() < 0.5 else f"{a}{var} {b_str} = {c}"
                items.append(self.rng.choice([
                    f"solve {eq}",
                    f"solve for {var}: {eq}",
                    f"find {var} in {eq}",
                    f"{eq}",
                    f"what value of {var} satisfies {eq}?",
                    f"calculate solution to {eq}",
                ]))

            elif mode == "linear_word":
                a = self.rng.randint(2, 10)
                b = self.rng.randint(1, 20)
                c = self.rng.randint(25, 80)
                items.append(self.rng.choice([
                    f"if {a}*{var} + {b} equals {c}, what is {var}?",
                    f"solve for {var} where {a}{var} - {b} = {c}",
                    f"find {var}: {var} + {b} = {c}",
                    f"solve {var} - {b} = {c}",
                    f"what is {var} if 2*{var} = {c}?",
                ]))

            elif mode == "quadratic":
                # Create with nice integer roots r1, r2
                r1 = self.rng.randint(-10, 10)
                r2 = self.rng.randint(-10, 10)
                # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2
                b = -(r1 + r2)
                c = r1 * r2
                b_str = f"+ {b}*{var}" if b > 0 else (f"- {abs(b)}*{var}" if b < 0 else "")
                c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
                eq = f"{var}^2 {b_str} {c_str} = 0".replace("  ", " ").strip()
                items.append(self.rng.choice([
                    f"solve {eq}",
                    f"find roots of {eq}",
                    f"quadratic equation: {eq}",
                    f"roots of {eq}",
                    f"solve the quadratic {eq}",
                    f"{eq}",
                ]))

            elif mode == "quadratic_roots":
                r = self.rng.randint(1, 15)
                c = r * r
                items.append(self.rng.choice([
                    f"roots of {var}^2 - {c} = 0",
                    f"solve {var}^2 - {c} = 0",
                    f"find zeros of {var}^2 - {c} = 0",
                    f"solve {var}^2 = {c}",
                    f"find the roots for {var}^2 - {c}",
                ]))

        return items[:count]

    def generate_algorithms(self, count: int) -> List[str]:
        items: List[str] = []

        while len(items) < count:
            mode = self.rng.choice(["sort", "binary_search", "shortest_path"])

            if mode == "sort":
                k = self.rng.randint(4, 9)
                nums = [self.rng.randint(1, 99) for _ in range(k)]
                nums_space = " ".join(map(str, nums))
                nums_comma = ", ".join(map(str, nums))

                items.append(self.rng.choice([
                    f"sort: {nums_space}",
                    f"sort: {nums_comma}",
                    f"sort this list: {nums_comma}",
                    f"sort this list efficiently: {nums_comma}",
                    f"arrange in ascending order: {nums_comma}",
                    f"order these numbers: {nums_space}",
                    f"sort the array: [{nums_comma}]",
                    f"quicksort {nums_comma}",
                    f"sort descending: {nums_space}",
                ]))

            elif mode == "binary_search":
                k = self.rng.randint(5, 10)
                data = sorted(list(set(self.rng.randint(1, 100) for _ in range(k))))
                if self.rng.random() < 0.8:
                    target = self.rng.choice(data)
                else:
                    target = self.rng.randint(1, 100)
                data_comma = ",".join(map(str, data))

                items.append(self.rng.choice([
                    f"binary-search: target={target} data={data_comma}",
                    f"binary search for target {target} in {data_comma}",
                    f"binary-search target={target} array={data_comma}",
                    f"find index of target={target} in sorted array: {data_comma}",
                    f"perform binary search: target={target} data={data_comma}",
                ]))

            elif mode == "shortest_path":
                nodes = ["A", "B", "C", "D", "E", "F"]
                n = self.rng.randint(3, 5)
                selected = nodes[:n]
                start = selected[0]
                end = selected[-1]

                # Generate random connected edges
                edges = []
                for i in range(len(selected) - 1):
                    w = self.rng.randint(1, 8)
                    edges.append(f"{selected[i]}-{selected[i+1]}:{w}")
                # Optional shortcut or extra edge
                if len(selected) >= 3 and self.rng.random() < 0.6:
                    edges.append(f"{selected[0]}-{selected[-1]}:{self.rng.randint(5, 15)}")

                edges_str = ",".join(edges)
                items.append(self.rng.choice([
                    f"shortest-path: start={start} end={end} edges={edges_str}",
                    f"shortest path from {start} to {end} with edges {edges_str}",
                    f"dijkstra shortest route start={start} end={end} edges={edges_str}",
                    f"find path: start={start} end={end} edges={edges_str}",
                ]))

        return items[:count]

    def generate_text_stats(self, count: int) -> List[str]:
        items: List[str] = []

        sample_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning enables computers to learn from empirical experience and data.",
            "Information theory establishes the fundamental limits on data compression and communication.",
            "Simplicity is prerequisite for reliability and maintainable system architecture.",
            "To be or not to be that is the quintessential existential question.",
            "Data structures and algorithms form the backbone of modern computer science.",
            "Adaptive agent harnesses route tasks dynamically with active verification.",
            "Classification uncertainty guides fallback escalation when confidence drops.",
            "Python code executes safely when expressions are strictly parsed with abstract syntax trees.",
            "Statistical language models predict next tokens based on probability distributions.",
        ]

        while len(items) < count:
            mode = self.rng.choice(["word_count", "char_freq", "entropy", "summary"])
            sent = self.rng.choice(sample_sentences)
            if self.rng.random() < 0.4:
                # Add random noise words
                extra = " ".join(self.rng.sample(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "omega"], k=3))
                sent = f"{sent} {extra}."

            if mode == "word_count":
                items.append(self.rng.choice([
                    f"word-count: {sent}",
                    f"count words in: {sent}",
                    f"word count of {sent}",
                    f"how many words are in this text: {sent}",
                    f"calculate word count: {sent}",
                ]))

            elif mode == "char_freq":
                items.append(self.rng.choice([
                    f"character-frequency: {sent}",
                    f"char frequency of: {sent}",
                    f"letter frequency in: {sent}",
                    f"find most frequent characters in: {sent}",
                    f"character-frequency: {sent[:30]}",
                ]))

            elif mode == "entropy":
                items.append(self.rng.choice([
                    f"entropy: {sent}",
                    f"text entropy of: {sent}",
                    f"calculate Shannon entropy: {sent}",
                    f"entropy of text: {sent}",
                    f"compute character entropy: {sent}",
                ]))

            elif mode == "summary":
                items.append(self.rng.choice([
                    f"summarize this text: {sent}",
                    f"analyze text: {sent}",
                    f"reading time for: {sent}",
                    f"text statistics: {sent}",
                    f"calculate stats for text: {sent}",
                ]))

        return items[:count]

    def generate_fallback(self, count: int) -> List[str]:
        items: List[str] = []

        fallback_templates = [
            "who was Napoleon Bonaparte?",
            "write a poem about the ocean and the starry night",
            "recommend a good sci-fi movie to watch tonight",
            "translate 'hello, how are you?' to Japanese",
            "how does a refrigerator work inside?",
            "what is the capital of Australia?",
            "explain quantum entanglement in simple terms",
            "tell me a funny programming joke",
            "summarize the main causes of World War I",
            "how do airplane wings generate lift?",
            "what are the symptoms of seasonal allergies?",
            "draft an email apologizing for a delayed meeting",
            "give me three recipes using chicken and broccoli",
            "what is the history of the Eiffel Tower?",
            "explain the difference between capitalism and socialism",
            "who painted the Mona Lisa?",
            "what is photosynthesis and why is it important?",
            "how do noise-cancelling headphones work?",
            "describe the plot of the novel 1984 by George Orwell",
            "what are the rules of chess for castling?",
            "can dogs eat peanut butter safely?",
            "what is the distance from the Earth to Mars?",
            "how does blockchain technology prevent double spending?",
            "suggest a workout routine for beginners",
            "write a haiku about autumn leaves",
            "why is the sky blue during the day?",
            "who discovered penicillin?",
            "what are black holes made of?",
            "give me tips for effective public speaking",
            "how does the stock market determine equity prices?",
        ]

        # Expand with variations
        prefixes = [
            "",
            "please tell me: ",
            "can you answer: ",
            "query: ",
            "question: ",
            "explain: ",
        ]

        while len(items) < count:
            base = self.rng.choice(fallback_templates)
            prefix = self.rng.choice(prefixes)
            items.append(f"{prefix}{base}".strip())

        return items[:count]
