from .prover9_solver import FOL_Prover9_Program
from typing import List, Dict, Any, Set

class Solver_base:
    def __init__(self, solver):
        self.solver = solver
        self.output_list = ['True', 'False', 'Uncertain']
    
    def solve(self, logic_program):
        prover9_program = self.solver(logic_program)
        answer, error_message = prover9_program.execute_program()

        if answer in self.output_list:
            return answer, prover9_program.used_idx, prover9_program.logic_proof, error_message
        
        else:
            return answer, [], [], error_message

    def multiple_choice(self, premises_list , option_list):
        # answers = []
        # for i, opt in enumerate(option_list):
        #     logic_program = self.forming_logic_program(premises_list, opt)
        #     ans, _, _ = self.solve(logic_program)
        #     if ans == 'True':
        #         answers.append(self.mapping_mutiple_choice(i))
        # return {
        #     "Answer": answers,
        #     "used_premises": [],
        #     "idx": []
        # }
        pass
    
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
            None: 'No'
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

def extract_conclusions(text):
    return re.findall(r'^[ABCD]\s+(.*)', text, flags=re.MULTILINE)

class Prover9_K(Solver_base):
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
        return any(A.lower() in self._clean_fol(prem).lower() for prem in premises_fol)
    # def _is_vacuous_conclusion(self, conclusion: str, premises_fol: List[str]) -> bool:
    #     """
    #     Check whether a conclusion of the form A → B is vacuous,
    #     meaning the antecedent A is not used or not derivable from any premise.
    #     """
    #     conclusion = conclusion.strip()

    #     # Match implication A -> B
    #     m = re.match(r'^(.*?)\s*->\s*(.*?)$', conclusion)
    #     if not m:
    #         return False

    #     antecedent = m.group(1).strip()
    #     # Normalize and extract any function-like expression
    #     antecedent_func = re.findall(r'\w+\(.*?\)', antecedent)
    #     if not antecedent_func:
    #         return False
    #     A = antecedent_func[0]

    #     # Check if A appears in any premise
    #     antecedent_lower = A.lower()
    #     return all(antecedent_lower not in self._clean_fol(p).lower() for p in premises_fol)



    def multiple_choice(self, premises_list, option_lists):
        option_choice = {}  # { 'A' : [1, 2]} # ans : idx list
        vacuous_flags = []  # check vacuous proof cho từng option
        option_idx_list = []  # giữ index theo thứ tự id option
        
        false_idx_list = []   # lưu idx của các đáp án bị chứng minh False


        final_ans = 'A'
        idx_final_ans = []
        dic_idx_wrong_options = {}
        dic_proof_yes_options = {}
        proof_final_ans = []
        dic_proof_wrong_options = {}


        for id, option in enumerate(option_lists):
            
            logic_program = self.forming_logic_program(
                premises=premises_list,
                conclusion=option
            )
            answer, idx, proof, error_message = self.solve(logic_program)

            print(answer)
            print(error_message)
            if answer == 'True':
                # Check vacuous
                used_premises = idx
                is_vacuous_premises = all(self._is_trivial_premise(premises_list[p-1]) for p in used_premises)
                # is_vacuous_premises = len(used_premises) == 0
                is_vacuous_conclusion = self._is_vacuous_conclusion(option, premises_list)
                is_vacuous = is_vacuous_premises or is_vacuous_conclusion

                vacuous_flags.append(is_vacuous)
                option_idx_list.append(id)
                option_choice[id] = (idx, is_vacuous)
                dic_proof_yes_options[id] = idx
            elif answer == 'False':
                dic_idx_wrong_options[
                    self.mapping_mutiple_choice(id)
                ] =  idx
                dic_proof_wrong_options[
                    self.mapping_mutiple_choice(id)
                ] = proof


                

        # sort the list by idx length (số premises được dùng để chứng minh)
        sorted_option_choice = sorted(option_choice.items(), key=lambda x: len(x[1][0]), reverse=True)

        if sorted_option_choice:
            # Lọc ưu tiên những đáp án không vacuous
            non_vacuous = [(opt, (idx, vac)) for opt, (idx, vac) in sorted_option_choice if not vac]

            if non_vacuous:
                first_option, (first_idx, _) = non_vacuous[0]
            else:
                first_option, (first_idx, _) = sorted_option_choice[0]
            
            final_ans = self.mapping_mutiple_choice(first_option)
            idx_final_ans = first_idx
            proof_final_ans = dic_proof_yes_options[first_option]
            
            return {
                "final_ans" : final_ans,
                "idx_final_ans" : idx_final_ans,
                "dic_idx_wrong_options" : dic_idx_wrong_options,
                "proof_final_ans" : proof_final_ans,
                "dic_proof_wrong_options" : dic_proof_wrong_options
            }
        else:
            return {
                "final_ans" : 'A',
                "idx_final_ans" : [],
                "dic_idx_wrong_options" : dic_idx_wrong_options,
                "proof_final_ans" : [],
                "dic_proof_wrong_options" : dic_proof_wrong_options
            }

    def solving_questions(self, premises, questions):
        """
        solve yes no / multiple choices based on given input
        returns:
            final_answer, first_option (A/B/C/D), idx of final_answer, list of idx where answer == False
        """
        for question in questions:
            if '\n' in question:
                list_conclusion = [
                    ques[2:] for ques in question.split('\n')
                    if ques and ques[0] in 'ABCD' and ques[1] == ' '
                ]
                        

                # print(premises)
                print(list_conclusion)
                return self.multiple_choice(
                    premises_list=premises,
                    option_lists=list_conclusion
                )
            elif "<q>" in question:
                ans_list = []
                idx_list = []
                proof_list = []

                question_list = question.split("<q>")
                for question in question_list:
                    logic_program = self.forming_logic_program(
                        premises=premises,
                        conclusion=question
                    )
                    answer, idx, proof, error_message = self.solve(logic_program)
                    ans_list.append(self.mapping_answer(answer))
                    idx_list.append(idx)
                    proof_list.append(proof)

                return {
                    "final_ans": ans_list,
                    "idx_final_ans": idx_list,
                    "dic_idx_wrong_options": {},
                    "proof_final_ans": proof_list,
                    "dic_proof_wrong_options": {}
                }
                

            else:
                logic_program = self.forming_logic_program(
                    premises=premises,
                    conclusion=question
                )
                answer, idx, proof, error_message = self.solve(logic_program)

                print(error_message)
                    

                return {
                    "final_ans": self.mapping_answer(answer),
                    "idx_final_ans": idx,
                    "dic_idx_wrong_options": {},
                    "proof_final_ans": proof,
                    "dic_proof_wrong_options": {}
                }




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
                                if line and line[0] in 'ABCD' and line[1] == ' ']
                return self.multiple_choice(premises_list=premises,
                                             option_list=option_lines)  
            else:                            
                logic_program = self.forming_logic_program(premises, q)
                ans, idx, _ = self.solve(logic_program)
                return self.mapping_answer(ans), idx



fol_premises = [
  "∀x (Curriculum(x) ∧ WellStructured(x) ∧ HasExercises(x) → EnhancesStudentEngagement(x))",
        "∀x (Curriculum(x) ∧ EnhancesStudentEngagement(x) ∧ ProvidesAccessToAdvancedResources(x) → EnhancesCriticalThinking(x))",
        "∀x (Faculty(x) ∧ PrioritizesPedagogicalTraining(x) ∧ DevelopsCurriculum(x) → WellStructuredCurriculum(x))",
        "Prioritizes(Faculty, PedagogicalTraining) ∧ Prioritizes(Faculty, CurriculumDevelopment)",
        "∃x (Curriculum(x) ∧ PracticalExercises(x))",
        "∀x (Curriculum(x) → ProvidesAccessToAdvancedResources(x))"
]

# yesno_question_fol = ["∀x (LogicalChain(x) ∧ MeetsRequirements(x, John) → CollaborativeResearchProject(x))"]


question_fol = [
    "A CanAccessRestrictedArchives(John) ∧ ¬CanSubmitResearchProposals(John)\n"
    "B CanApplyForCollaborativeResearchProjects(John)\n"
    "C NeedsMorePublications(John) → CanAccessRestrictedArchives(John)\n"
    "D ExtendedLibraryAccess(John) ∧ ¬CanApplyForCollaborativeResearchProjects(John)"
]

# question_fol = ["∀x (LogicalChain(x) ∧ MeetsRequirements(x, John) → CollaborativeResearchProject(x))"]

# "new-fol": [
#         "∀x (FacultyMember(x) ∧ TaughtForAtLeast5Years(x) → ExtendedLibraryAccess(x))",
#         "∀x (Person(x) ∧ ExtendedLibraryAccess(x) ∧ PublishedAcademicPaper(x) → CanAccessArchives(x))",
#         "∀x (CanAccessArchives(x) ∧ CompletedResearchEthicsTraining(x) → CanSubmitResearchProposals(x))",
#         "∀x (CanSubmitResearchProposals(x) ∧ HasDepartmentalEndorsement(x) → CanApplyForCollaborativeResearchProjects(x))",
#         "Person(John) ∧ Person(John) ∧ TaughtForAtLeastYears(John, 5)",
#         "HasPublishedAcademicPaper(John) ∧ ∃x (AcademicPaper(x) ∧ PublishedBy(x, John))",
#         "∀x (Professor(x) ∧ Person(x) → CompletedResearchEthicsTraining(x))",
#         "HasDepartmentalEndorsement(John)"
#     ]


# que-fol = "CorrectConclusion(ProfessorJohn) ↔ (RenownedExpert(ProfessorJohn) ∧ PublishedNumerousPapersInPrestigiousJournals(ProfessorJohn) ∧ ReceivedNumerousAwardsForContributions(ProfessorJohn))
# \nA ∀x (Person(x) ∧ Person(x) → (CanAccessArchives(x) ∧ ¬SubmitProposals(x)))
# \nB ∀x (Person(x) → CanApplyForCollaborativeResearchProjects(x))
# \nC ∀x (NeedsMorePublications(x) → CanAccessArchives(x))
# \nD ∀x (ExtendedLibraryAccess(x) → ¬ApplyForProjects(x)):"

# "premises-nl": [
#         "If a faculty member has taught for at least 5 years, they are eligible for extended library access.",
#         "If someone has extended library access and has published at least one academic paper, they can access restricted archives.",
#         "If someone can access restricted archives and has completed research ethics training, they can submit research proposals.",
#         "If someone can submit research proposals and has a departmental endorsement, they can apply for collaborative research projects.",
#         "Professor John has taught for at least 5 years.",
#         "Professor John has published at least one academic paper.",
#         "Professor John has completed research ethics training.",
#         "Professor John has a departmental endorsement."
#     ]

# "questions": [
#         "Based on the premises, what is the correct conclusion about Professor John?
# \nA. He can access restricted archives but cannot submit proposals
# \nB. He can apply for collaborative research projects
# \nC. He needs more publications to access archives
# \nD. He is eligible for extended library access but cannot apply for projects",
#         "Does the logical chain demonstrate that Professor John meets all requirements for collaborative research projects?"
#     ]


prover9 = Prover9_K(solver=FOL_Prover9_Program)
ans = prover9.solving_questions(fol_premises, question_fol)
print(ans)