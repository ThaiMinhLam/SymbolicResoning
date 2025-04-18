from utils import Prover9Tool

prover9_tool = Prover9Tool()

# "premises-NL": [
# 1. "If a Python code is well-tested, then the project is optimized.",
# 2. "If a Python code does not follow PEP 8 standards, then it is not well‑tested.",
# 3. "All Python projects are easy to maintain.",
# 4. "All Python code is well‑tested.",
# 5. "If a Python code follows PEP 8 standards, then it is easy to maintain.",
# 6. "There exists at least one Python project that has clean and readable code.",
# 7. "If a Python code is well‑tested, then it follows PEP 8 standards.",
# 8. "If a Python project is not optimized, then it is not well‑tested.",
# 9. "There exists at least one Python project that is well‑structured.",
# 10. "If a Python project is well‑structured, then it is optimized.",
# 11. "If being well‑tested implies following PEP 8 standards, then all Python code is well‑tested.",
# 12. "If being well‑structured implies optimization, then if a Python project is not optimized, it is not well‑tested.",
# 13. "If a Python project is easy to maintain, then it is well‑tested.",
# 14. "If a Python project is optimized, then it has clean and readable code.",
# 15. "All Python projects are well‑structured.",
# 16. "All Python projects have clean and readable code.",
# 17. "There exists at least one Python project that follows best practices.",
# 18. "There exists at least one Python project that is optimized.",
# 19. "If a Python project is not well‑structured, then it does not follow PEP 8 standards."
# ],

# "questions": [
# Q1  "Based on the above premises, which conclusion is correct?
#      A. If a Python project is not optimized, then it is not well‑tested.
#      B. If all Python projects are optimized, then all Python projects are well‑structured.
#      C. If a Python project is well‑tested, then it must be clean and readable.
#      D. If a Python project is not optimized, then it does not follow PEP 8 standards.",

# Q2  "According to the above premises, is the following statement true?
#      Statement: If all Python projects are well‑structured, then all Python projects are optimized."
# ]

premises_fol =  ["∀x (WT(x) → O(x))",
            "∀x (¬PEP8(x) → ¬WT(x))",
            "∀x (EM(x))",
            "∀x (WT(x))",
            "∀x (PEP8(x) → EM(x))",
            "∀x (WT(x) → PEP8(x))",
            "∀x (WS(x) → O(x))",
            "∀x (EM(x) → WT(x))",
            "∀x (O(x) -> CR(x))",
            "∀x (WS(x))",
            "∀x (CR(x))",
            "∃x (BP(x))",
            "∃x (O(x))",
            "∀x (¬WS(x) → ¬PEP8(x))"]
# Yes / No question
yesno_fol = "(all x WS(x)) -> (all x O(x))."

choices_fol = {
    "A": "all x (-OPT(x) -> -WT(x)).",
    "B": "(all x OPT(x)) -> (all x WS(x)).",
    "C": "all x (WT(x) -> CR(x)).",
    "D": "all x (-OPT(x) -> -PEP8(x))."
}

# Yes / No question
yesno_fol = "(all x WS(x)) -> (all x O(x))."

result = prover9_tool.run(premises_fol, yesno_fol)
answer = "Yes" if result["is_valid"] else "No"
used_premises = result["used_premises"]
print(used_premises)
