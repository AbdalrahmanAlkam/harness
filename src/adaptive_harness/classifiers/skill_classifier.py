"""Skill & Intent Classifier determining which developer tool or capability to invoke."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SKILL_CLASSES = [
    "code_edit",
    "run_command",
    "search_explore",
    "testing",
    "ask_clarification",
    "general_reasoning",
]


@dataclass
class SkillClassificationResult:
    primary_skill: str
    confidence: float
    probabilities: Dict[str, float]
    ranked_skills: List[Tuple[str, float]]


class SkillClassifier:
    """Classifies user prompts and agent tasks into specialized software engineering skills."""

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("clf", LogisticRegression(C=2.0, max_iter=500, random_state=42)),
            ]
        )
        self._train_baseline()

    def _train_baseline(self) -> None:
        """Trains baseline model on developer intent task exemplars."""
        data = [
            # code_edit
            ("edit the file src/main.py and replace foo with bar", "code_edit"),
            ("create a new file called config.json with default settings", "code_edit"),
            ("refactor the function calculate_total to handle negative values", "code_edit"),
            ("write a python script to parse logs", "code_edit"),
            ("fix the bug in line 42 where variable is undefined", "code_edit"),
            ("update README.md with installation instructions", "code_edit"),
            ("add error handling to file_ops.py", "code_edit"),
            # run_command
            ("run git status and tell me what changed", "run_command"),
            ("execute pip install -r requirements.txt", "run_command"),
            ("run bash command to list processes", "run_command"),
            ("check python version in terminal", "run_command"),
            ("git commit -m 'initial commit'", "run_command"),
            ("run build script in shell", "run_command"),
            # search_explore
            ("find all files that import numpy", "search_explore"),
            ("search for definition of TaskClassifier in codebase", "search_explore"),
            ("grep for TODO in the src directory", "search_explore"),
            ("list files in the output directory", "search_explore"),
            ("where is the database connection initialized?", "search_explore"),
            ("explore project structure and files", "search_explore"),
            # testing
            ("run pytest on tests/test_harness.py", "testing"),
            ("run all unit tests and verify they pass", "testing"),
            ("execute pytest with verbose output", "testing"),
            ("test if the equation solver handles negative roots", "testing"),
            ("run test suite to check for regressions", "testing"),
            # ask_clarification
            ("should we use sqlite or postgres for the database?", "ask_clarification"),
            ("i want to rewrite either the frontend or the backend, what do you think?", "ask_clarification"),
            ("maybe delete all old files or keep backups?", "ask_clarification"),
            ("refactor either policy.py or harness.py, choose one", "ask_clarification"),
            ("which approach should i pick for authentication?", "ask_clarification"),
            # general_reasoning
            ("explain how expected calibration error is calculated", "general_reasoning"),
            ("what is the difference between entropy and brier score?", "general_reasoning"),
            ("how does the adaptive harness work conceptually?", "general_reasoning"),
            ("summarize our project architecture", "general_reasoning"),
            ("why is decoupling prediction from verification important?", "general_reasoning"),
        ]

        # Duplicate data for robust variance
        texts = [t for t, _ in data] * 5
        labels = [l for _, l in data] * 5
        self.pipeline.fit(texts, labels)

    def classify(self, text: str) -> SkillClassificationResult:
        """Predicts probability distribution across developer skills."""
        clean = text.strip()
        probs_raw = self.pipeline.predict_proba([clean])[0]
        classes = self.pipeline.named_steps["clf"].classes_

        prob_dict = {str(c): float(p) for c, p in zip(classes, probs_raw)}

        # Apply deterministic heuristic boost for explicit signals
        low = clean.lower()
        if any(w in low for w in ["pytest", "run test", "run tests", "unit test"]):
            prob_dict["testing"] = min(prob_dict.get("testing", 0.0) + 0.6, 0.99)
        elif any(w in low for w in ["git ", "pip install", "bash ", "shell ", "terminal "]):
            prob_dict["run_command"] = min(prob_dict.get("run_command", 0.0) + 0.5, 0.99)
        elif any(w in low for w in ["search ", "grep ", "find all", "where is", "list files"]):
            prob_dict["search_explore"] = min(prob_dict.get("search_explore", 0.0) + 0.5, 0.99)
        elif any(w in low for w in ["write file", "edit file", "replace ", "refactor ", "create file"]):
            prob_dict["code_edit"] = min(prob_dict.get("code_edit", 0.0) + 0.5, 0.99)

        # Normalize
        total = sum(prob_dict.values())
        norm_probs = {k: v / total for k, v in prob_dict.items()}

        ranked = sorted(norm_probs.items(), key=lambda x: x[1], reverse=True)
        top_skill, top_conf = ranked[0]

        return SkillClassificationResult(
            primary_skill=top_skill,
            confidence=round(top_conf, 4),
            probabilities={k: round(v, 4) for k, v in norm_probs.items()},
            ranked_skills=[(k, round(v, 4)) for k, v in ranked],
        )
