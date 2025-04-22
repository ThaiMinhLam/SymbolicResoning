
import re
import os
from typing import List, Dict, Any, Set
import subprocess
import uuid
import re

def _strip_outer_parens(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        count = 0
        for i, ch in enumerate(expr):
            if ch == '(':
                count += 1
            elif ch == ')':
                count -= 1
            if count == 0 and i < len(expr) - 1:
                break
        else:
            expr = expr[1:-1].strip()
            continue
        break
    return expr


def _normalize_ascii_fol(expr: str) -> str:
    replacements = {
        '→': '->',
        '⇒': '->',
        '⟹': '->',
        '¬': '-',
        '∧': '&',
        '∨': '|',
        '↔': '<->',
        '⇔': '<->',
        '∀': 'all ',
        '∃': 'exists ',
    }
    for k, v in replacements.items():
        expr = expr.replace(k, v)
    return expr 

class Prover9Tool:
    def __init__(self):
        self.binary_path = "/data/npl/ICEK/News/Qwen_evaluate/LADR-2009-11A/bin/prover9"
        self.save_run = "/data/npl/ICEK/News/Qwen_evaluate/prover9_solver/runs"
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
    
    def _clean_fol(self, expr: str) -> str:
    
        replacements = {
            '⟹': '->', '⇒': '->', '→': '->', '=>': '->', '⊃': '->',
            '↔': '<->', '⇔': '<->', '≡': '<->',
            '¬': '-', 'not ': '-', '!': '-',
            '∧': '&', 'AND': '&', 'and': '&',
            '∨': '|', ' OR ': '|', ' or ': '|',
            '∀': 'all ', '∃': 'exists ',
        }

        
        connective_replacements = {
            r'\bImplies\s*\(\s*(.*?)\s*,\s*(.*?)\s*\)': r'(\1 -> \2)',
            r'\bImplies\s*\(\s*(.*?)\s*,\s*(.*?)\s*\)': r'(\1 -> \2)',
            r'\bAnd\s*\(\s*(.*?)\s*,\s*(.*?)\s*\)': r'(\1 & \2)',
            r'\bOr\s*\(\s*(.*?)\s*,\s*(.*?)\s*\)': r'(\1 | \2)',
            r'\bNot\s*\(\s*(.*?)\s*\)': r'- \1',
        }

        # Apply connective transformations
        for pattern, replacement in connective_replacements.items():
            expr = re.sub(pattern, replacement, expr)

        # Apply symbol replacements
        for k, v in replacements.items():
            expr = expr.replace(k, v)

        # Handle quantifiers
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
        cleaned_premises = [
            _strip_outer_parens(_normalize_ascii_fol(self._clean_fol(p)).strip().rstrip('.'))
            for p in premises_fol
        ]
        cleaned_conclusion = _normalize_ascii_fol(self._clean_fol(conclusion_fol)).strip().rstrip('.')
        cleaned_conclusion = _strip_outer_parens(cleaned_conclusion)                
        print("cleaned premises: ",cleaned_premises)
        print("clean conclusion: ", cleaned_conclusion)

        if cleaned_conclusion in cleaned_premises:
            idx = cleaned_premises.index(cleaned_conclusion) + 1
            print("✅ Kết luận đã có sẵn trong premise.")
            return {
                "Answer": "Yes",
                "used_premises": [premises_fol[idx - 1]],
                "idx": [idx]
            }

        premises_fol = [_normalize_ascii_fol(self._clean_fol(p)) for p in premises_fol]
        conclusion_fol = _normalize_ascii_fol(self._clean_fol(conclusion_fol))
        original_to_index = {p.rstrip('.'): i + 1 for i, p in enumerate(premises_fol)}

        filtered = [p if p.endswith('.') else p + '.' for p in dict.fromkeys(premises_fol)]

        prover_input = (
            "set(auto).\nassign(max_seconds, 10).\nformulas(assumptions).\n"
        )
        premise_map = {}
        for i, prem in enumerate(filtered, 1):
            prem_clean = prem.rstrip('.')
            prover_input += f"{prem_clean} # label(premise_{i}).\n"
            self.premise_index_cache[prem_clean] = i
            premise_map[i] = prem_clean

        prover_input += "end_of_list.\n\nformulas(goals).\n"
        prover_input += f"{conclusion_fol.rstrip('.') + '.'}\nend_of_list.\n"

        temp_id = str(uuid.uuid4())
        input_file = f"{self.save_run}/prover_input_{temp_id}.in"
        output_file = f"{self.save_run}/prover_output_{temp_id}.out"

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(prover_input)

        result = subprocess.run(f"{self.binary_path} -f {input_file}", shell=True, capture_output=True, text=True)
        output = result.stdout

        output_lower = output.lower()
        if "theorem proved" in output_lower:
            used_premises = self._extract_used_premises(output, premise_map)
            idx = [original_to_index.get(prem.rstrip('.'), -1) for prem in used_premises]
            print("✅ Trạng thái: YES – Theorem proved.")
            return {"Answer": "Yes", "used_premises": used_premises, "idx": idx}
        elif "search failed" in output_lower or "sos_empty" in output_lower:
            print("❌ Trạng thái: NO – Không chứng minh được.")
            return {"Answer": "No", "used_premises": [], "idx": []}
        elif any(k in output_lower for k in ["search gave up", "timeout"]):
            print("🟡 Trạng thái: UNCERTAIN – Hết thời gian hoặc từ bỏ tìm kiếm.")
            return {"Answer": "Uncertain", "used_premises": [], "idx": []}
        else:
            print("⚠️ Không xác định được trạng thái.")
            return {"Answer": "Uncertain", "used_premises": [], "idx": []}
    
    def multiple_choice_run(self, premises_fol: List[str], conclusion_fols: List[str]) -> Dict[str, Any]:
        return_dic = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }

        # 2 trường hợp đặc biệt:
        # 1 là No hết
        # 2 là solve được hơn 2 đáp án

        answers = []
        used_premises_list = []
        used_idxs_list = []
        answers_fol = []
        for idx, conclusion_fol in enumerate(conclusion_fols):
            result = self.run(premises_fol, conclusion_fol)
            if result['Answer'] == 'Yes':
                used_premises = result['used_premises']
                used_idxs = result['idx']
                used_premises_list.append(used_premises)
                used_idxs_list.append(used_idxs)
                answers.append(return_dic[idx])
                answers_fol.append(conclusion_fol)

        return {
            "Answer": answers,
            "used_premises": used_premises_list,
            "idx": used_idxs_list,
            "Answers_FOL": answers_fol
        }
    
    def check_consistency(self, statements: List[str]) -> Dict[str, Any]:
        """
        Kiểm tra tính nhất quán của một tập mệnh đề.
        Nếu từ tập đó suy ra được 'false', thì nó mâu thuẫn (inconsistent).
        """
        if not statements or all(not s.strip() for s in statements):
            print("❌ Danh sách mệnh đề rỗng hoặc chỉ chứa khoảng trắng.")
            return {"Consistent": None}

        cleaned_statements = [_normalize_ascii_fol(self._clean_fol(s)) for s in statements]

        prover_input = (
            "set(auto).\n"
            "assign(max_seconds, 60).\n"
            "assign(max_given, 10000).\n"
            "assign(max_weight, 100).\n"
            "formulas(assumptions).\n"
        )

        valid_count = 0
        for i, s in enumerate(cleaned_statements):
            cleaned = s.strip()

            if re.match(r'^\(?all\s+[a-zA-Z0-9_]+\s*\(.*\)\)?\s*->\s*\(?all\s+[a-zA-Z0-9_]+\s*\(', cleaned):
                print(f"❌ Mệnh đề nguy hiểm có thể gây lỗi Prover9: {cleaned}")
                return {"Consistent": None}

            if cleaned:
                # Bọc lại trong ngoặc nếu là all/exists
                if cleaned.startswith("all ") or cleaned.startswith("exists "):
                    cleaned = f"({cleaned})"

                line = f"{cleaned.rstrip('.') + '.'}\n"  # ❌ KHÔNG thêm label!
                # print(f"✅ Đưa vào prover_input: {line.strip()}")
                prover_input += line
                valid_count += 1
            else:
                print(f"⚠️ Bỏ qua mệnh đề rỗng tại vị trí {i}")

        if valid_count == 0:
            print("❌ Không có mệnh đề hợp lệ để kiểm tra.")
            return {"Consistent": None}

        prover_input += (
            "end_of_list.\n\n"
            "formulas(goals).\n"
            "false.\n"
            "end_of_list.\n"
        )

        temp_id = str(uuid.uuid4())
        input_file = f"{self.save_run}/prover_input_consistency_{temp_id}.in"

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(prover_input)

        try:
            result = subprocess.run(
                f"{self.binary_path} -f {input_file}",
                shell=True,
                capture_output=True,
                text=True
            )
            output = result.stdout.lower()
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Không thể chạy Prover9: {str(e)}")


        if "theorem proved" in output:
            print("❌ Mâu thuẫn: Tập mệnh đề *KHÔNG* nhất quán.")
            return {"Consistent": False}
        elif "search failed" in output or "sos_empty" in output:
            print("✅ Nhất quán: Tập mệnh đề là *nhất quán* (không suy ra mâu thuẫn).")
            return {"Consistent": True}
        else:
            print("⚠️ Không xác định được kết quả.")
            return {"Consistent": None}



