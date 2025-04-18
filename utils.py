import re
import os
from typing import List, Dict, Any
import subprocess
import uuid

class Prover9Tool:
    def __init__(self):
        self.binary_path = "/data/npl/ICEK/News/Qwen_evaluate/LADR-2009-11A/bin/prover9"
        self.premise_index_cache = {}

   
    @staticmethod
    def _extract_used_premises(output: str, num_premises: int) -> List[int]:
        """
        Trả về các premise (1‑based) thực sự xuất hiện trong chứng minh nè các bro.
        """
        # Xác định khối PROOF
        m_start = re.search(r'=+\s*PROOF\s*=+', output, re.I)
        m_end   = re.search(r'=+\s*end of proof\s*=+', output, re.I)
        if not m_start or not m_end or m_end.start() <= m_start.end():
            return []

        proof_text = output[m_start.end():m_end.start()]

        #  Bắt số trong [rule(...)]t
        RULES = ('clausify|resolve|unit_res|binary_res|hyper|para|factor|demod|ur_res')
        rule_pat = rf'\[(?:{RULES})\(\s*(\d+)'        # chỉ cần số đầu tiên
        ids_rule = {int(m) for m in re.findall(rule_pat, proof_text, re.I)}

        #  Bắt "# label(premise_i)" – cũng chỉ trong proof_text
        label_pat = r'#\s*label\s*\(\s*premise_(\d+)\s*\)'
        ids_label = {int(m) for m in re.findall(label_pat, proof_text)}

        hits = {i for i in ids_rule | ids_label if 1 <= i <= num_premises}
        return [i  for i in sorted(hits)] # 1-based nè 

    def run(self, premises_fol: List[str], conclusion_fol: str) -> Dict[str, Any]:
        # Chuyển sang dạng Prover9 hiểu
        replacements = {
            '⟹': '->', '⇒': '->', '→': '->', '=>': '->', '⊃': '->',
            '↔': '<->', '⇔': '<->', '≡': '<->',
            '¬': '-', 'not ': '-', '!': '-',
            '∧': '&', 'AND': '&', 'and': '&',
            '∨': '|', 'OR': '|', 'or': '|',
            'ForAll(': 'all ',    'forall(': 'all ',    '∀': 'all ',
            'Exists(': 'exists ', 'exists(': 'exists ', '∃': 'exists ',
        }
        for k, v in replacements.items():
            premises_fol = [p.replace(k, v) for p in premises_fol]
            conclusion_fol = conclusion_fol.replace(k, v)

        premises_fol = [
            p if p.endswith('.') else p + '.'
            for p in dict.fromkeys(premises_fol)  
        ]

        prover_input = (
            "set(auto).\n"
            "assign(max_seconds, 10).\n"
            "formulas(assumptions).\n"
        )
        for i, prem in enumerate(premises_fol, 1):
            prem = prem.rstrip('.')  # Bỏ dấu chấm cuối
            prover_input += f"{prem} # label(premise_{i}).\n"
            self.premise_index_cache[prem] = i
        prover_input += (
            "end_of_list.\n\n"
            "formulas(goals).\n"
            f"{conclusion_fol.rstrip('.') + '.'}\n"
            "end_of_list.\n"
        )

    
        if "set(" in prover_input.split("end_of_list.")[-1]:
            raise ValueError("Lỗi cú pháp bạn eiiii")

        # In input để kiểm tra
        print("Prover9 input:\n", prover_input)

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
            if result.returncode != 0:
                with open(output_file, "w", encoding="latin-1") as f:
                    f.write(output)
                with open(error_file, "w", encoding="latin-1") as f:
                    f.write(error_msg)
                raise RuntimeError(f"Prover9 lỗi (exit code {result.returncode}): {error_msg}\nOutput: {output}")
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Không thể chạy Prover9: {str(e)}")

        if "Fatal error" in output:
            raise RuntimeError(f"Prover9 gặp lỗi cú pháp: {output}")
        is_valid = "THEOREM PROVED" in output
        used_premises = (
            self._extract_used_premises(output, len(premises_fol))
            if is_valid else []
        )

        # Xóa tệp tạm
        for file in [input_file, output_file, error_file]:
            if os.path.exists(file):
                os.remove(file)

        return {"is_valid": is_valid, "used_premises": used_premises}