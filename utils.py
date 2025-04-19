import re
import os
from typing import List, Dict, Any, Set
import subprocess
import uuid
import re

def convert_logic_syntax(expr: str) -> str:
    # Chuyển ForAll(x, ...) -> all x (...)
    expr = re.sub(r'ForAll\s*\(\s*([a-zA-Z0-9_]+)\s*,', r'all \1 (', expr)
    expr = re.sub(r'Exists\s*\(\s*([a-zA-Z0-9_]+)\s*,', r'exists \1 (', expr)
    # Đóng dấu ngoặc đúng cho all/exists
    expr = re.sub(r'\)\s*$', r')', expr)
    return expr

class Prover9Tool:
    def __init__(self):
        self.binary_path = "/data/npl/ICEK/News/Qwen_evaluate/LADR-2009-11A/bin/prover9"
        self.premise_index_cache = {}

    _LOGICAL_TOKENS: Set[str] = {
        'all', 'exists', '->', '<->', '&', '|', '-',
    }

    @staticmethod
    def _tokenize(formula: str) -> Set[str]:
        raw_tokens = re.findall(r'[A-Za-z][A-Za-z0-9_]*', formula)
        return {t.lower() for t in raw_tokens if len(t) > 1}

    def _is_relevant(self, premise: str, conclusion: str) -> bool:
        toks_p = self._tokenize(premise) - self._LOGICAL_TOKENS
        toks_c = self._tokenize(conclusion) - self._LOGICAL_TOKENS
        return bool(toks_p & toks_c)
    
    def _convert_logic_syntax(self, expr: str) -> str:
        # Chuyển ForAll(x, ...) → all x (...)
        expr = re.sub(r'\bForAll\s*\(\s*([a-zA-Z0-9_]+)\s*,', r'all \1 (', expr)
        expr = re.sub(r'\bExists\s*\(\s*([a-zA-Z0-9_]+)\s*,', r'exists \1 (', expr)
        return expr

    def _clean_fol(self, expr: str) -> str:
        
        # expr = re.sub(r'^\s*F\s*\|\s*', '', expr)

        replacements = {
            '⟹': '->', '⇒': '->', '→': '->', '=>': '->', '⊃': '->',
            '↔': '<->', '⇔': '<->', '≡': '<->',
            '¬': '-', 'not ': '-', '!': '-',
            '∧': '&', 'AND': '&', 'and': '&',
            '∨': '|', 'OR': '|', 'or': '|',
            '∀': 'all ', '∃': 'exists ',
        }

        for k, v in replacements.items():
            expr = expr.replace(k, v)

        # Sửa ForAll(x, φ) thành all x (φ)
        expr = re.sub(r'\bForAll\s*\(\s*([a-zA-Z0-9_]+)\s*,', r'all \1 (', expr)
        expr = re.sub(r'\bExists\s*\(\s*([a-zA-Z0-9_]+)\s*,', r'exists \1 (', expr)

        return expr


    def _extract_used_premises(self, output: str, premise_map: Dict[int, str]) -> List[str]:
        m_start = re.search(r'=+\s*PROOF\s*=+', output, re.I)
        m_end = re.search(r'=+\s*end of proof\s*=+', output, re.I)
        if not m_start or not m_end or m_end.start() <= m_start.end():
            return []

        proof_text = output[m_start.end():m_end.start()]

        RULES = ('clausify|resolve|unit_res|binary_res|hyper|para|factor|demod|ur_res')
        rule_pat = rf'\[(?:{RULES})\(\s*(\d+)'
        ids_rule = {int(m) for m in re.findall(rule_pat, proof_text, re.I)}

        label_pat = r'#\s*label\s*\(\s*premise_(\d+)\s*\)'
        ids_label = {int(m) for m in re.findall(label_pat, proof_text)}

        hits = {i for i in ids_rule | ids_label if i in premise_map}
        return [premise_map[i] for i in sorted(hits)]

    def run(self, premises_fol: List[str], conclusion_fol: str) -> Dict[str, Any]:
        filtered = [p for p in premises_fol if self._is_relevant(p, conclusion_fol)]
        if not filtered:
            filtered = premises_fol

        premises_fol   = [self._clean_fol(p) for p in premises_fol]
        filtered       = [self._clean_fol(p) for p in filtered]
        conclusion_fol = self._clean_fol(conclusion_fol)

        original_to_index = {p: i + 1 for i, p in enumerate(premises_fol)}

        filtered = [
            p if p.endswith('.') else p + '.'
            for p in dict.fromkeys(filtered)
        ]

        prover_input = (
            "set(auto).\n"
            "assign(max_seconds, 10).\n"
            "formulas(assumptions).\n"
        )

        premise_map = {}
        for i, prem in enumerate(filtered, 1):
            prem_clean = prem.rstrip('.')
            prover_input += f"{prem_clean} # label(premise_{i}).\n"
            self.premise_index_cache[prem_clean] = i
            premise_map[i] = prem_clean

        prover_input += (
            "end_of_list.\n\n"
            "formulas(goals).\n"
            f"{conclusion_fol.rstrip('.') + '.'}\n"
            "end_of_list.\n"
        )

        if "set(" in prover_input.split("end_of_list.")[-1]:
            raise ValueError("Lỗi cú pháp bạn eiiii")

        temp_id = str(uuid.uuid4())
        input_file = f"prover_input_{temp_id}.in"
        output_file = f"prover_output_{temp_id}.out"
        error_file = f"prover_error_{temp_id}.log"

        for file in [input_file, output_file, error_file]:
            if os.path.exists(file):
                os.remove(file)

        with open(input_file, "w", encoding="ascii") as f:
            f.write(prover_input)

        with open(input_file, "r", encoding="ascii") as f:
            written_input = f.read()

        if written_input != prover_input:
            raise RuntimeError(f"Lỗi ghi tệp: Nội dung {input_file} lệch với prover_input, coi lại bro")

        try:
            result = subprocess.run(
                f"{self.binary_path} -f {input_file}",
                shell=True,
                capture_output=True,
                text=True
            )
            output = result.stdout
            error_msg = result.stderr

        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Không thể chạy Prover9: {str(e)}")

        if "Fatal error" in output:
            raise RuntimeError(f"Prover9 gặp lỗi cú pháp: {output}")

        print(output)

        output_lower = output.lower()

        if "theorem proved" in output_lower:
            used_premises = self._extract_used_premises(output, premise_map)
            idx = [original_to_index[prem] for prem in used_premises if prem in original_to_index]
            print("✅ Trạng thái: YES – Theorem proved.")
            return {
                "Answer": "Yes",
                "used_premises": used_premises,
                "idx": idx
            }

        elif "search failed" in output_lower or "sos_empty" in output_lower:
            print("❌ Trạng thái: NO – Không chứng minh được.")
            return {
                "Answer": "No",
                "used_premises": [],
                "idx": []
            }

        elif any(k in output_lower for k in ["search gave up", "timeout"]):
            print("🟡 Trạng thái: UNCERTAIN – Hết thời gian hoặc từ bỏ tìm kiếm.")
            return {
                "Answer": "Uncertain",
                "used_premises": [],
                "idx": []
            }

        else:
            print("⚠️ Không xác định được trạng thái.")
            return {
                "Answer": "Uncertain",
                "used_premises": [],
                "idx": []
            }
