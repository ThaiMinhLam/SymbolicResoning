from prover9_solver import FOL_Prover9_Program

class Solver_base:
    def __init__(self, solver):
        self.solver = solver
        self.output_list = ['True', 'False', 'Uncertain']
    
    def solve(self, logic_program):
        prover9_program = self.solver(logic_program)
        answer, error_message = prover9_program.execute_program()

        if answer in self.output_list:
            return answer, prover9_program.used_idx, prover9_program.get_used_premises()
        
        else:
            return answer, []

    def multiple_choice(self, premises_list , option_list):
        answers = []
        for i, opt in enumerate(option_list):
            logic_program = self.forming_logic_program(premises_list, opt)
            ans, _, _ = self.solve(logic_program)
            if ans == 'True':
                answers.append(self.mapping_mutiple_choice(i))
        return {
            "Answer": answers,
            "used_premises": [],
            "idx": []
        }
    
    def mapping_mutiple_choice(self, idx):
        dic = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }
        return dic[idx]

    def mapping_answer(self, ans):
        dic = {
            'True': 'Yes',
            'False': 'No',
            'Uncertain': 'Uncertain',
            'None': 'None'
        }
        return dic[ans]

    def solving_questions(self):
        """
        solve yes no / mutiple choices based on given input

        """

        pass
    
    def forming_logic_program(self, premises, conclusion):
        """
        Forming logic program based on given input
        """
        premises_fol_string = ''
        for premise in premises:
            premise_string = premise + ' ::: abc \n'
            premises_fol_string += premise_string

        choice_fol_string = conclusion + ' ::: abc \n'
        
        logic_program = f"""Premises: 
        {premises_fol_string}
        Conclusion:
        {choice_fol_string}
        """
        return logic_program



class Prover9_K(Solver_base):
    def __init__(self, solver):
        super().__init__(solver=solver)

    def multiple_choice(self, premises_list, option_lists):
        option_choice = {} # { 'A' : [1, 2]} # ans : idx list

        for id, option in enumerate(option_lists):
            logic_program = self.forming_logic_program(
                premises = premises_list,
                conclusion = option
            )
            answer, idx,_ = self.solve(logic_program)
            if answer == 'True':
                option_choice[self.mapping_mutiple_choice(id)] = idx
        

        # sort the len of list idx
        sorted_option_choice = sorted(option_choice.items(), key=lambda x: len(x[1]), reverse=True)
        # get the first element
        if len(sorted_option_choice) > 0:
            first_option = sorted_option_choice[0][0]
            # get the idx of first option
            idx = sorted_option_choice[0][1]
            # get the answer
            answer = self.mapping_answer(answer)
        else:
            first_option = 'None'
            idx = []
            answer = 'None'

        return answer, first_option, idx
    
    def solving_questions(self, premises, questions):
        """
        solve yes no / mutiple choices based on given input

        """

        for question in questions:
            if '\n' in  question:
                list_conclusion = []
                question_list = question.split('\n')
                for question in question_list:
                    if question[1] in ['A', 'B', 'C', 'D']:
                        question = question[1:]
                        list_conclusion.append(question)
                return self.mutiple_choice(
                    premises = premises,
                    conclusion = list_conclusion
                )
            
            else:
                 
                answer, idx, _ = self.solve(
                    logic_program = self.forming_logic_program(
                        premises = premises,
                        conclusion = question
                    )
                )

                return self.mapping_answer(answer), idx



import re
from typing import List, Dict, Any, Set

class Prover9_T(Solver_base):
    def __init__(self, solver):
        super().__init__(solver=solver)

    def _is_trivial_premise(self, premise: str) -> bool:
       premise = premise.strip()
       m = re.match(r'^all\s+\w+\s*\(\s*-?\s*\w+\s*\(\s*\w*\s*\)\s*\)\s*$', premise)
       return m is not None
    
    def _is_vacuous_conclusion(self, conclusion: str, premises_fol: List[str]) -> bool:

        conclusion = conclusion.strip()
        m = re.match(r'^-\s*(\w+\(.*?\))\s*->\s*-\s*(\w+\(.*?\))$', conclusion)
        if not m:
            return False

        A = m.group(1)
  
        return any(A.lower() in prem.lower() for prem in premises_fol)

    def multiple_choice(self, premises_list, option_list):      
        answers, used_premises_list, used_idxs_list, vacuous_flags = [], [], [], []

        for i, opt in enumerate(option_list):
            logic_program = self.forming_logic_program(premises_list, opt)
            prov = self.solver(logic_program)        
            ans, _ = prov.execute_program()
            if ans != 'True':
                continue

            used_premises = prov.get_used_premises()
            is_vacuous = (
                all(self._is_trivial_premise(p) for p in used_premises) or
                self._is_vacuous_conclusion(opt, premises_list)
            )

            answers.append(self.mapping_mutiple_choice(i))
            used_premises_list.append(used_premises)
            used_idxs_list.append(prov.used_idx)
            vacuous_flags.append(is_vacuous)

        if not answers:        
            return {"Answer": [], "used_premises": [], "idx": []}

        if any(not v for v in vacuous_flags):
            answers           = [a for a, v in zip(answers, vacuous_flags) if not v]
            used_premises_list = [p for p, v in zip(used_premises_list, vacuous_flags) if not v]
            used_idxs_list     = [i for i, v in zip(used_idxs_list,  vacuous_flags) if not v]

        return {"Answer": answers, "used_premises": used_premises_list, "idx": used_idxs_list}
    
    def solving_questions(self, premises, questions):
        for q in questions:
            if '\n' in q:              
                option_lines = [line[1:].strip()        
                                for line in q.splitlines()
                                if line and line[0] in 'ABCD']
                return self.multiple_choice(premises_list=premises,
                                             option_list=option_lines)  
            else:                            
                logic_program = self.forming_logic_program(premises, q)
                ans, idx, _ = self.solve(logic_program)
                return self.mapping_answer(ans), idx

