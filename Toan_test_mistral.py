
# ##### LOOOO, LOOOO, hello Nguyên, giờ có 2 option là sài Premises-NL hoặc Logic Program
# ##### t có comment ở dưới r á



import os, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)

os.environ["HF_TOKEN"] = "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"  

MODEL_NAME = "/workspace/model_hf/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,      
    device_map="auto",          
    use_auth_token=True,
)

gen_cfg = GenerationConfig(
    task               = "text-generation",
    max_new_tokens     = 1024,
    do_sample          = True,
    temperature        = 0.1,
    top_p              = 0.95,
    repetition_penalty = 1.15,
    num_beams          = 2,        
    use_cache          = True,
    
)

llama = pipeline(
    "text-generation",
    model      = model,
    tokenizer  = tokenizer,
    generation_config = gen_cfg,
)

def fo_logic_eval(prompt:str):
    outputs = llama(
        prompt,
        
    )
    return outputs[0]["generated_text"]
from typing import List

def create_template_explain(logic_program: List[str], nl_premises: List[str], premises_FOL: List[str], used_idx: List[int], question: str, answer:str, nl_flag: bool = False):
    
    logic_program = "\n".join([f"{item}" for item in (logic_program)])

    formatted_nl_premises = ""
    for idx, premise in enumerate(nl_premises, start=1):
        formatted_nl_premises += f"{idx}. {premise}\n"

    formatted_fol_premises = ""
    for idx, premise in enumerate(premises_FOL, start=1):
        formatted_fol_premises += f"{idx}. {premise}\n"

    # nl_flag: là kí hiệu dùng premises-NL hay không, nếu bật là dùng còn ko thì dùng Logic program
    if nl_flag:
        give_u = """
        I will give you:
        • A numbered list of premises in natural language.  
        • The same premises in FOL with matching numbers.  
        • One or more questions that logically follow from the premises.
        • For each question, a list called idx that names the exact premise numbers that justify the correct answer.  
        • The answer of the question. """
    else: 
        give_u = """
        I will give you:
        • Logic Program Predicates: Each line follows: `PredicateName(args) means: Description`. Only provide you a description of predicate
        • The same premises in FOL with matching numbers.  
        • One or more questions that logically follow from the premises.
        • For each question, a list called idx that names the exact premise numbers that justify the correct answer.  
        • The answer of the question."""
            
    template = f"""
    [INST] 
    You are an expert tutor in explaning first-order logic.
    Your explanations will be used for automated grading, so they **must** be precise, minimal, and follow the mandated format.
    """ + give_u + """
    ### Your task
    1. For every question, use ONLY the premises whose numbers appear in its idx list.  
    2. Write a short, clear explanation that:  
    – cites those premises by number but do NOT write again the same premises,  
    – names the logical rules you apply (e.g., contraposition, modus ponens), and  
    – shows, step by step, how the answer follows from the cited premises, and 
    - If question is Multiple Choice Question, you need to explain why other options are wrong (in case your answer is too short).
    Ignore all premises that are **not** listed in idx.  
    3. Return the explanation in clear, brief, concise but making sure logical, the explanation must be a passage with maximum 3 sentences."""
    
    if nl_flag:
        template += f"""
        ### Example 1: YES/NO QUESTION
        ***premises-NL: 
                1. "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
                2. "Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.",
                3. "Students who have passed the language proficiency exam are eligible for the international program.",
                4. "Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.",
                5. "Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.",
                6. "Students who have been awarded an honors diploma and have received a faculty recommendation qualify for the university scholarship.",
                7. "Sophia has completed the core curriculum.",
                8. "Sophia has passed the science assessment.",
                9. "Sophia has completed the research methodology course.",
                10. "Sophia has completed her capstone project.",
                11. "Sophia has completed the required community service hours."
            
        ***premises-FOL: 
                1. "ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))",
                2. "ForAll(x, (qualified_for_advanced_courses(x) ∧ completed_research_methodology(x)) → eligible_for_international_program(x))",
                3. "ForAll(x, passed_language_proficiency(x) → eligible_for_international_program(x))",
                4. "ForAll(x, (eligible_for_international_program(x) ∧ completed_capstone_project(x)) → awarded_honors_diploma(x))",
                5. "ForAll(x, (awarded_honors_diploma(x) ∧ completed_community_service(x)) → qualifies_for_scholarship(x))",
                6. "ForAll(x, (awarded_honors_diploma(x) ∧ received_faculty_recommendation(x)) → qualifies_for_scholarship(x))",
                7. "completed_core_curriculum(Sophia)",
                8. "passed_science_assessment(Sophia)",
                9. "completed_research_methodology(Sophia)",
                10. "completed_capstone_project(Sophia)",
                11. "completed_community_service(Sophia)"
            ]
        ***used_idx: [1,2,4,5,7,8,9,10,11]
        ***question: "Does Sophia qualify for the university scholarship, according to the premises?"
        ***answer:  "Yes"
        ***explain: "The university scholarship per premise 5 requires an honors diploma and community service. Premise 4 requires international program eligibility and a capstone project, premise 2 requires advanced course qualification and research methodology, and premise 1 requires core curriculum and science assessment. Premises 7–11 confirm Sophia’s curriculum, assessment, methodology, capstone, and service, satisfying all conditions for the scholarship."
        
        ### Example 2: MULTIPLE CHOICE QUESTION
        ***premises-NL:
                1. "If a faculty member has taught for at least 5 years, they are eligible for extended library access.",
                2. "If someone has extended library access and has published at least one academic paper, they can access restricted archives.",
                3. "If someone can access restricted archives and has completed research ethics training, they can submit research proposals.",
                4. "If someone can submit research proposals and has a departmental endorsement, they can apply for collaborative research projects.",
                5. "Professor John has taught for at least 5 years.",
                6. "Professor John has published at least one academic paper.",
                7. "Professor John has completed research ethics training.",
                8. "Professor John has a departmental endorsement."
        ***premises-FOL: 
                1. "ForAll(x, taught_min_five_years(x) → eligible_extended_library(x))",
                2. "ForAll(x, (eligible_extended_library(x) ∧ has_publications(x)) → can_access_restricted_archives(x))",
                3. "ForAll(x, (can_access_restricted_archives(x) ∧ completed_ethics_training(x)) → can_submit_research_proposals(x))",
                4. "ForAll(x, (can_submit_research_proposals(x) ∧ has_departmental_endorsement(x)) → can_apply_collaborative_projects(x))",
                5. "taught_min_five_years(John)",
                6. "has_publications(John)",
                7. "completed_ethics_training(John)",
                8. "has_departmental_endorsement(John)"
            ]
        ***used_idx: [1,2,3,4,5,6,7,8]
        ***question: "Based on the premises, what is the correct conclusion about Professor John?\nA. He can access restricted archives but cannot submit proposals\nB. He can apply for collaborative research projects\nC. He needs more publications to access archives\nD. He is eligible for extended library access but cannot apply for projects"
        ***answer:  "B"
        ***explain: "Professor John has taught for at least 5 years (Premise 5), so he is eligible for extended library access (by Premise 1). Because he has at least one publication (Premise 6), he can access restricted archives (Premise 2). With completed ethics training (Premise 7), he can submit research proposals (Premise 3), and with departmental endorsement (Premise 8), he can apply for collaborative research projects (Premise 4). Hence, option B is logically valid."
        ### YOUR INPUT, FILL IN **explanation using premises-FOL, idx, answer to explain
        ***premises-NL:
            {formatted_nl_premises}
        ***premises-FOL: 
            {formatted_fol_premises}
        ***question: {question}
        ***used_idx: {used_idx}
        ***answer:  {answer}
        [/INST]
        [OUTPUT]
        ***explanation: """ 
        return (template)
    else: 
        template += f"""
    ### Example 1: YES/NO QUESTION
    ***logic_program: 
            "Student(x) ::: x is a student.",
            "CompletedCoreCurriculum(x) ::: x has completed the core curriculum.",
            "PassedScienceAssessment(x) ::: x has passed the science assessment.",
            "EligibleForAdvancedCourses(x) ::: x is eligible for advanced courses.",
            "QualifiedForAdvancedCourses(x) ::: x is qualified for advanced courses.",
            "CompletedResearchMethodology(x) ::: x has completed research methodology.",
            "EligibleForInternationalProgram(x) ::: x is eligible for the international program.",
            "PassedLanguageProficiencyExam(x) ::: x has passed the language proficiency exam.",
            "CompletedCapstoneProject(x) ::: x has completed the capstone project.",
            "AwardedHonorsDiploma(x) ::: x has been awarded an honors diploma.",
            "CompletedCommunityService(x) ::: x has completed community service.",
            "QualifiesForUniversityScholarship(x) ::: x qualifies for the university scholarship.",
            "ReceivedFacultyRecommendation(x, y) ::: x has received a faculty recommendation for y.",
            "Name(x, \"Sophia\") ::: x is named Sophia.",
            "Sophia(x) ::: x is Sophia.",
            "Agent(Sophia) ::: Sophia is an agent.",
            "Verb(Completed) ::: completed is a verb.",
            "Object(ResearchMethodologyCourse) ::: research methodology course is the object of the verb completed.",
            "Knowledge(Sophia, Course) ::: Sophia has knowledge of the course.",
            "CapstoneProject(y) ::: y is the capstone project.",
            "Completed(x, y) ::: x has completed y.",
            "Adjective(Required) ::: required is an adjective.",
            "Noun(CommunityServiceHours) ::: community service hours is a noun.",
            "HoursMet(Sophia) ::: Sophia has met the required hours.",
            "QualifiesForScholarship(Sophia) ::: Sophia qualifies for the scholarship.",
            "MeetsEligibilityCriteria(Sophia) ::: Sophia meets the eligibility criteria.",
            "Scholarship(y) ::: y is a scholarship.",
            "NeedsFacultyRecommendation(x, y) ::: x needs a faculty recommendation for y.",
            "InternationalProgram(y) ::: y is an international program.",
            "EligibleFor(x, y) ::: x is eligible for y.",
            "NeedsLanguageProficiencyExamPass(Sophia) ::: Sophia needs to pass the language proficiency exam.",
            "HonorsDiploma(Sophia) ::: Sophia has been awarded an honors diploma.",
            "UniversityScholarship(y) ::: y is a university scholarship.",
            "Premises(z) ::: z are premises.",
            "Qualifies(x, y, z) ::: x"
        
    ***premises-FOL: 
            1. "ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))",
            2. "ForAll(x, (qualified_for_advanced_courses(x) ∧ completed_research_methodology(x)) → eligible_for_international_program(x))",
            3. "ForAll(x, passed_language_proficiency(x) → eligible_for_international_program(x))",
            4. "ForAll(x, (eligible_for_international_program(x) ∧ completed_capstone_project(x)) → awarded_honors_diploma(x))",
            5. "ForAll(x, (awarded_honors_diploma(x) ∧ completed_community_service(x)) → qualifies_for_scholarship(x))",
            6. "ForAll(x, (awarded_honors_diploma(x) ∧ received_faculty_recommendation(x)) → qualifies_for_scholarship(x))",
            7. "completed_core_curriculum(Sophia)",
            8. "passed_science_assessment(Sophia)",
            9. "completed_research_methodology(Sophia)",
            10. "completed_capstone_project(Sophia)",
            11. "completed_community_service(Sophia)"
        ]
    ***used_idx: [1,2,4,5,7,8,9,10,11]
    ***question: "Does Sophia qualify for the university scholarship, according to the premises?"
    ***answer:  "Yes"
    ***explain: "The university scholarship per premise 5 requires an honors diploma and community service. Premise 4 requires international program eligibility and a capstone project, premise 2 requires advanced course qualification and research methodology, and premise 1 requires core curriculum and science assessment. Premises 7–11 confirm Sophia’s curriculum, assessment, methodology, capstone, and service, satisfying all conditions for the scholarship."
    
    ### Example 2: MULTIPLE CHOICE QUESTION
    ***logic_program: 
            "FacultyMember(x) ::: Person(x)",
            "TaughtForAtLeast5Years(x) ::: Person(x) & HasTaughtForAtLeastYears(x, 5)",
            "EligibleForExtendedLibraryAccess(x) ::: Person(x) & CanAccessRestrictedArchives(x) & CompletedResearchEthicsTraining(x)",
            "Person(x) ::: x is a Person",
            "ExtendedLibraryAccess(x) ::: Person(x)",
            "PublishedAcademicPaper(x) ::: Person(x) & HasPublishedAcademicPaper(x)",
            "CanAccessRestrictedArchives(x) ::: Person(x)",
            "CompletedResearchEthicsTraining(x) ::: Person(x)",
            "CanSubmitResearchProposals(x) ::: Person(x) & HasDepartmentalEndorsement(x)",
            "HasDepartmentalEndorsement(x) ::: Person(x)",
            "CanApplyForCollaborativeResearchProjects(x) ::: Person(x)",
            "AccessRestrictedArchives(x) ::: Person(x)",
            "SubmitProposals(x) ::: Person(x)",
            "LogicalChain(x) ::: Person(x)"
    ***premises-FOL: 
            1. "ForAll(x, taught_min_five_years(x) → eligible_extended_library(x))",
            2. "ForAll(x, (eligible_extended_library(x) ∧ has_publications(x)) → can_access_restricted_archives(x))",
            3. "ForAll(x, (can_access_restricted_archives(x) ∧ completed_ethics_training(x)) → can_submit_research_proposals(x))",
            4. "ForAll(x, (can_submit_research_proposals(x) ∧ has_departmental_endorsement(x)) → can_apply_collaborative_projects(x))",
            5. "taught_min_five_years(John)",
            6. "has_publications(John)",
            7. "completed_ethics_training(John)",
            8. "has_departmental_endorsement(John)"
        ]
    ***used_idx: [1,2,3,4,5,6,7,8]
    ***question: "Based on the premises, what is the correct conclusion about Professor John?\nA. He can access restricted archives but cannot submit proposals\nB. He can apply for collaborative research projects\nC. He needs more publications to access archives\nD. He is eligible for extended library access but cannot apply for projects"
    ***answer:  "B"
    ***explain: "Professor John has taught for at least 5 years (Premise 5), so he is eligible for extended library access (by Premise 1). Because he has at least one publication (Premise 6), he can access restricted archives (Premise 2). With completed ethics training (Premise 7), he can submit research proposals (Premise 3), and with departmental endorsement (Premise 8), he can apply for collaborative research projects (Premise 4). Hence, option B is logically valid."
    ### YOUR INPUT, FILL IN **explanation using premises-FOL, idx, answer to explain
    ***logic_program:
        {logic_program}
    ***premises-FOL: 
        {formatted_fol_premises}
    ***question: {question}
    ***used_idx: {used_idx}
    ***answer:  {answer}
    [/INST]
    [OUTPUT]
    ***explanation: """ 
    return (template)
    


if __name__ == "__main__":
    import time
    start_time = time.time()
#     prompt = """
#     You are an expert in First-Order Logic (FOL) reasoning tasked with solving complex logical inference problems. You will receive an input containing natural language premises ("premises-NL"), their corresponding FOL representations ("premises-FOL"), and a set of questions. Your goal is to analyze the premises, derive valid inferences, and answer the questions accurately. Follow these steps precisely:
#     Understand the Input Structure:

#     ### The input includes:
#     "premises-NL": A list of natural language statements describing logical relationships.
#     "premises-FOL": A list of equivalent FOL formulas, using predicates like P(x) (excellent student), U(x) (receives scholarship), S(x) (participates in extracurricular activities), etc.
#     "questions": A list of questions, which may include multiple-choice options or true/false statements.
#     Each premise in "premises-FOL" corresponds to a natural language premise in "premises-NL".

#     ### Analyze the Premises:
#     Carefully parse each FOL premise to understand the logical structure.
#     Identify key predicates and their meanings (e.g., P(x) means x is an excellent student, U(x) means x receives a scholarship).
#     Note quantifiers (ForAll, Exists), implications (→), negations (¬), and conjunctions.
#     Simplify complex premises by breaking them into smaller logical components if necessary.
#     Check for logical equivalences, such as contraposition (e.g., ¬A → ¬B is equivalent to B → A).

#     ### Derive Inferences:
#     Use logical rules (e.g., modus ponens, modus tollens, universal instantiation, existential instantiation) to derive conclusions.
#     Focus on premises relevant to each question by identifying predicates and variables involved.
#     For implications (A → B), consider the contrapositive (¬B → ¬A) to uncover equivalent forms.
#     Combine premises to form chains of reasoning where applicable.
#     Avoid unnecessary assumptions; base conclusions strictly on the given premises.

#     ###Answer the Questions:

#     ##For multiple-choice questions:
#     Evaluate each option against the premises.
#     Select the option that logically follows from the premises.
#     If multiple options seem plausible, choose the one with the strongest logical support.

#     ##For true/false questions:
#     Determine if the statement is a valid inference from the premises.
#     Use contraposition or other logical transformations to verify equivalence.
#     Provide the answer in the format specified (e.g., "A" for multiple-choice, "Yes" or "No" for true/false).

#     ### Provide Explanations:
#     For each question, explain the reasoning process clearly and concisely.
#     Reference specific premises by their index (e.g., premise 2) and describe how they contribute to the conclusion.
#     Highlight key logical steps, such as contraposition, implication chaining, or predicate instantiation.
#     Ensure the explanation is sufficient to justify the answer without introducing extraneous details.

#     ### Format the Output:
#     "answers": A list of answers corresponding to the questions (e.g., ["A", "Yes"]).
#     "idx": A list of lists, where each inner list contains the indices of premises used to derive the answer for each question (e.g., [[2, 6], [2]]).
#     "explanation": A list of strings, each providing a clear explanation for the corresponding answer.
#     Ensure the output is precise, with no extraneous information.

#     ###Example Handling:
#     For a multiple-choice question like "Which statement can be inferred?", evaluate each option:
#     Option A: "If a student receives a scholarship, they must be excellent."
#     Check premise 2: (¬P(x) → ¬U(x)) implies U(x) → P(x) by contraposition.
#     Combine with premise 6: (Q(x) → P(x)), which states all enrolled students are excellent.
#     Conclude that A is valid if the premises support U(x) → P(x).
#     Option B: "If a student does not participate in extracurricular activities, then they receive a scholarship."
#     Check premise 3: (¬S(x) → ¬U(x)) directly contradicts B, so B is invalid.
#     Option C: "If a student does not submit their research paper, then they are excellent."
#     Check premise 22: (¬SubmitsResearchPaper(x) → ¬ReceivesPassingGrade(x)). No direct link to P(x), so C is unsupported.
#     Option D: "If all enrolled students are excellent, then no student without excellence can receive a scholarship."
#     Premise 6 ensures all enrolled students are excellent, and premise 2 supports U(x) → P(x), so D is plausible but redundant given A.
#     For a true/false question like "If a student receives a scholarship, they must be excellent":
#     Verify using premise 2: (¬P(x) → ¬U(x)) implies U(x) → P(x), so the statement is true.

#     ### Error Handling:
#     If a premise is ambiguous, prioritize the FOL representation over the natural language version.
#     If no conclusion can be drawn for a question, indicate that no option is supported and explain why.
#     Double-check logical derivations to avoid errors in inference.

#     ### YOUR INPUT, FILL IN **answer, **idx, **explanation:
#     **premise-NL:
#         1. "There is at least one school that has students.",
#         2. "There is at least one school that has a principal.",
#         3. "If there is at least one school that has a principal, then there is at least one school that has teachers.",
#         4. "There is at least one school that has staff.",
#         5. "Every school has a principal.",
#         6. "Every school has staff.",
#         7. "There is at least one school that has both students and teachers.",
#         8. "If there is at least one school that has students, then there is at least one school that has staff.",
#         9. "If a school does not have teachers, then it does not have students.",
#         10. "If (if there is at least one school that has a principal, then there is at least one school that has teachers), then every school has a principal.",
#         11. "If there is at least one school that has staff, then there is at least one school that has teachers.",
#         12. "If (if there is at least one school that has students, then there is at least one school that has teachers), then there is at least one school that has a principal."
#     **question: "Based on the above premises, is the statement true?\nStatement: If a school has students, then it has teachers."
        
              
# """
    prompt = """
You are an expert in First-Order Logic (FOL) reasoning tasked with solving logical inference problems. You will receive an input with:
- "premises-NL": A list of natural language statements.
- "premises-FOL": A list of corresponding FOL formulas (e.g., P(x) for "excellent student", U(x) for "receives scholarship").
- "questions": A list of questions (multiple-choice or true/false).

### Tasks:
1. **Analyze Premises**:
   - Parse FOL premises to identify logical structure (quantifiers: ForAll, Exists; operators: →, ¬, ∧).
   - Define predicate meanings (e.g., P(x) = "x is excellent").
   - Simplify complex premises and apply logical equivalences (e.g., contraposition: ¬A → ¬B ≡ B → A).

2. **Derive Inferences**:
   - Use rules (modus ponens, modus tollens, instantiation) to draw conclusions.
   - Focus on premises relevant to each question.
   - Combine premises logically, avoiding unsupported assumptions.

3. **Answer Questions**:
   - **Multiple-choice**: Evaluate options, select the most supported answer.
   - **True/False**: Verify if the statement follows from premises using logical transformations (e.g., contraposition).
   - Return answers as "A"/"B"/"C"/"D" or "Yes"/"No".

4. **Provide Explanations**:
   - Explain reasoning concisely, citing premise indices (e.g., premise 2).
   - Highlight key logical steps (e.g., contraposition, implication).
   - Avoid extraneous details.

5. **Format Output**:
   - "answers": List of answers (e.g., ["A", "Yes"]).
   - "idx": List of lists with premise indices used (e.g., [[2, 6], [2]]).
   - "explanation": List of explanation strings.

6. **Error Handling**:
   - Prioritize FOL over NL if ambiguous.
   - If no conclusion, state "No inference possible" with reason.
   - Double-check logic.

### Example:
- **Question**: "Which statement can be inferred?"
  - A: "If a student receives a scholarship, they must be excellent."
    - Premise 2: ¬P(x) → ¬U(x) → U(x) → P(x) (contraposition).
    - Premise 6: Q(x) → P(x) supports A.
  - B: "If a student does not participate, they receive a scholarship."
    - Premise 3: ¬S(x) → ¬U(x) contradicts B.
  - Answer: "A", idx: [[2, 6]], explanation: "From premise 2 (¬P(x) → ¬U(x) implies U(x) → P(x)) and premise 6 (Q(x) → P(x)), A follows."
- **Question**: "Is 'If a student receives a scholarship, they must be excellent' true?"
  - Premise 2: ¬P(x) → ¬U(x) → U(x) → P(x).
  - Answer: "Yes", idx: [[2]], explanation: "Premise 2 (¬P(x) → ¬U(x)) implies U(x) → P(x) by contraposition."

### Your Input:
**premises-NL**:
1. "There is at least one school that has students."
2. "There is at least one school that has a principal."
3. "If there is at least one school that has a principal, then there is at least one school that has teachers."
4. "There is at least one school that has staff."
5. "Every school has a principal."
6. "Every school has staff."
7. "There is at least one school that has both students and teachers."
8. "If there is at least one school that has students, then there is at least one school that has staff."
9. "If a school does not have teachers, then it does not have students."
10. "If (if there is at least one school that has a principal, then there is at least one school that has teachers), then every school has a principal."
11. "If there is at least one school that has staff, then there is at least one school that has teachers."
12. "If (if there is at least one school that has students, then there is at least one school that has teachers), then there is at least one school that has a principal."

**question**: "Based on the above premises, is the statement true?\nStatement: If a school has students, then it has teachers."

### Output:
**answer**: ...
**idx**: ...
**explanation**: ...
"""
    
    result = fo_logic_eval(prompt)
    print(result)
    end_time = time.time()
    print(end_time - start_time)



# premises_NL =[
#       "If a Python code is well-tested, then the project is optimized.",
#       "If a Python code does not follow PEP 8 standards, then it is not well-tested.",
#       "All Python projects are easy to maintain.",
#       "All Python code is well-tested.",
#       "If a Python code follows PEP 8 standards, then it is easy to maintain.",
#       "If a Python code is well-tested, then it follows PEP 8 standards.",
#       "If a Python project is well-structured, then it is optimized.",
#       "If a Python project is easy to maintain, then it is well-tested.",
#       "If a Python project is optimized, then it has clean and readable code.",
#       "All Python projects are well-structured.",
#       "All Python projects have clean and readable code.",
#       "There exists at least one Python project that follows best practices.",
#       "There exists at least one Python project that is optimized.",
#       "If a Python project is not well-structured, then it does not follow PEP 8 standards."
#     ]
#     logic_program = ["PythonCode(x) ::: If a Python code is well-tested, then the project is optimized.",
#             "WellTested(x) ::: If a Python code does not follow PEP 8 standards, then it is not well-tested.",
#             "OptimizedProject(x) ::: All Python projects are easy to maintain.",
#             "PEP8Standards(x) ::: If a Python code follows PEP 8 standards, then it is easy to maintain.",
#             "PythonProject(x) ::: If a Python code is well-tested, then it follows PEP 8 standards.",
#             "EasyToMaintain(x) ::: If a Python project is well-structured, then it is optimized.",
#             "FollowsPEP8Standards(x) ::: If a Python code is well-tested, then it follows PEP 8 standards.",
#             "WellTestedPythonCode(x) ::: If a Python project is well-structured, then it has clean and readable code.",
#             "WellStructured(x) ::: If a Python project is well-tested, then it has clean and readable code.",
#             "Optimized(x) ::: If a Python project is easy to maintain, then it is optimized.",
#             "CleanCode(x) ::: If a Python code is well-tested, then it has clean and readable code.",
#             "ReadableCode(x) ::: If a Python project is well-structured, then it has clean and readable code.",
#             "Person(y) ::: There exists at least one Python project that follows best practices.",
#             "Perfect(y) ::: There exists at least one Python project that is optimized.",
#             "FollowsBestPractices(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards.",
#             "Clean(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards.",
#             "Readable(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards."]
#     premises_FOL = [
#         "∀x (WT(x) → O(x))",
#       "∀x (¬PEP8(x) → ¬WT(x))",
#       "∀x (EM(x))",
#       "∀x (WT(x))",
#       "∀x (PEP8(x) → EM(x))",
#       "∀x (WT(x) → PEP8(x))",
#       "∀x (WS(x) → O(x))",
#       "∀x (EM(x) → WT(x))",
#       "∀x (O(x) -> CR(x))",
#       "∀x (WS(x))",
#       "∀x (CR(x))",
#       "∃x (BP(x))",
#       "∃x (O(x))",
#       "∀x (¬WS(x) → ¬PEP8(x))"
#     ]
#     idx = [1]
#     question = "Which conclusion follows with the fewest premises?\nA. If a Python project is not optimized, then it is not well-tested\nB. If all Python projects are optimized, then all Python projects are well-structured\nC. If a Python project is well-tested, then it must be clean and readable\nD. If a Python project is not optimized, then it does not follow PEP 8 standards"
#     answer =  "A" 

    test_sample = (
    """
    You are an AI assistant skilled in logical reasoning and GPA calculation based on multiple premises. You will be given a set of premises and a question. Your task is to calculate the updated GPA step-by-step, and at each step, you must verify that your calculations are logically and numerically correct. If any step contains an inconsistency or mistake, fix it before continuing.

Important rules:
- You MUST use accurate calculations (simulate a calculator).
- Never use raw grades directly.
- Grade points = grade × credits.
- GPA = total grade points ÷ total credits.
- Only passed courses (grade ≥ 2.0) are included in GPA.
- If a course was failed (grade < 2.0), ignore it entirely.
- For retaken courses, only include the most recent passed attempt in GPA — ignore all failed attempts.

Strict logic checks:
- If a course was failed in the past, and later passed, use only the passed attempt.
- Do not count the same course twice.
- Do not use a failed grade, even if it appears in the premise.

### Premises:
1. To choose a major, a student needs at least 24 general education credits, a general GPA of 2.5 or higher, and no failed general course (i.e., all grades ≥ 2.0).
2. Each general course is worth 3 credits. Retaking a failed course costs $50 per attempt.
3. Students must retake failed general courses until they pass. There is no limit on retakes if fees are paid.
4. GPA is calculated using only passed courses (grade ≥ 2.0). Failed courses do not count toward credits or GPA.
5. A student currently has 18 general credits and a GPA of 2.7. The student failed course C1 (grade 1.8, 3 credits, 2 attempts, $100 paid).
6. The student retakes C1 (grade 2.2, passed, $50 paid) and passes a new course C2 (3 credits, grade 2.5). All fees have been paid.


### FOL: 
1. FORALL(x): MajorSelection(x) ↔ (GeneralCredits(x) ≥ 24 ∧ GeneralGPA(x) ≥ 2.5 ∧ FORALL(c, GeneralCourse(c) → Grade(x, c) ≥ 2.0))
2. FORALL(c): GeneralCourse(c) → Credits(c) = 3 ∧ FORALL(x): (Attempts(x, c) > 1 → RetakeFee(x, c, 50 * (Attempts(x, c) − 1)))
3. FORALL(x, c): (Grade(x, c) < 2.0 ∧ FeesPaid(x)) → RetakeCourse(x, c) ∧ ¬LimitRetakes(x, c)
4. FORALL(x): GeneralGPA(x) = Sum(Grade(c) * Credits(c) for GeneralCourse(c), Grade(c) ≥ 2.0) / Sum(Credits(c)) ∧ (Grade(c) < 2.0 → Credits(c) = 0)
5. Student(s1) ∧ GeneralCredits(s1) = 18 ∧ GeneralGPA(s1) = 2.7 ∧ Course(s1, c1, 3, 1.8, Attempts=2) ∧ RetakeFee(s1, c1, 100)
6. Courses(s1) = [c1_reenroll(3, 2.2, Fee=50), c2(3, 2.5)] ∧ FeesPaid(s1).

### Question:
What is the student's updated general GPA after passing C1 and C2?

### Instructions:
Follow these exact steps and verify each result carefully:

Step 1: Calculate the student's current total grade points using the formula: GPA × credits. Show the exact multiplication.

Step 2: For each newly passed course (C1 and C2), calculate its grade points using grade × credits. Do NOT use the failed attempt for C1.

Step 3: Add the new grade points to the current total grade points.

Step 4: Add the passed course credits (C1 and C2) to the current credits to get the new total credits.

Step 5: Calculate the new GPA = total grade points ÷ total credits.

Step 6: Double-check that:
  - Grade points were calculated as grade × credits,
  - Only passed courses are counted,
  - Failed attempts are ignored,
  - Retaken courses count only once (most recent passed attempt).
⚠️ IMPORTANT:
When rounding the final GPA, use **standard rounding rules**:
- Keep only two decimal places.
-Use standard rounding rules: keep two decimal places. Example: 2.8143 → 2.81, 2.615 → 2.62, etc.
-Only write the final GPA once. Do not repeat or loop.
- Double-check your final division and rounding.
- DOUBLE-CHECK every computatation you just did.

Step 7: Output the final GPA, rounded to two decimal places.
Note: The current 18 credits do NOT include the previously failed course C1 (grade 1.8). It is only added now because it is passed.

### OUTPUT:
**answer:
    """
    )

    prompt_hard_samples = (
    """
    [INST] 
    You are an expert tutor in explaning first-order logic.
    Your explanations will be used for automated grading, so they **must** be precise, minimal, and follow the mandated format.


    I will give you:
    • A numbered list of premises in natural language.  
    • One or more questions that logically follow from the premises.
    
    ### Reasoning Toolkit  (always keep this section)

    • SumPoints(old_avg, old_qty, new_items[]) → total_points  
        – Converts any averaged metric back to raw “points”.  
        – Each new_items[i] = (qty_i, value_i).  
        – Formula: total_points = old_avg × old_qty + Σ(qty_i × value_i)

    • WeightedAverage(total_points, total_qty) → new_avg  
        – Handles any mean (GPA, mean score, mean cost).  
        – Rounds to 2 decimals unless asked otherwise.

    • AddPenalty(base_qty, penalties[]) → new_qty  
        – Each penalty = (amount, ±direction).  
        – direction = “‑” to subtract, “+” to add.

    • ReplaceIfHigher(old_value, candidates[]) → best_value  
        – Used for “dùng điểm cao nhất khi retake”, “choose maximum discount”, v.v.

    • CheckRequirement(requirement_set, data) → Yes / No  
        – requirement_set may include “≥, ≤, =, all‑true” clauses.  
        – Returns No if **any** clause fails.

    Rules of use  
    1. **Always reduce aggregates to base totals** with SumPoints before recomputing an average.  
    2. **Never add two averages trực tiếp**.  
    3. List **only the premises actually needed** to justify each call above.  
    4. Think privately; output only {answer, idx, explanation} JSON per question.
    ### Your task
    For every question asked, you must:
    1. **Reason step‑by‑step** with the premises (do not output these steps).  
    2. Derive the correct, concise **answer**.  
    3. Identify **all premise lines that were essential** for your reasoning (use the exact numbers that appear in the input).  
    4. Write a clear, short **explanation**, showing how those premises lead to the answer (you may cite the lines by their numbers).

    ### Example 1: 
    ***premises-NL: 
            1. "A student can withdraw up to 3 courses per academic year if remaining credits ≥ 10 (Year 1, 2) or ≥ 8 (Year 3+); each withdrawal deducts 0.5 credits from total accumulated credits.",
            2. "Courses have 3, 4, or 5 credits; withdrawn courses contribute 0 credits to semester GPA.",
            3. "Students must register for 12–18 credits per semester; withdrawals reduce semester credits.",
            4. "No regulation limits total withdrawals, but max 3 per year.",
            5. "A student with < 8 accumulated credits cannot withdraw courses.",
            6. "A student (Year 2) has 45 credits, withdrew 2 courses in Year 1 (penalty 1 credit), including C1 (4 credits, withdrawn), C2 (3 credits, withdrawn).",
            7. "In Year 2, semester S1, the student registered for 15 credits, withdrew C3 (5 credits), C4 (3 credits), attempted C5 (4 credits, passed)."
    ***questions: 
            "What is the student’s total accumulated credits after Year 2, semester S1?"
    ***idx: [1,6,7]
    ***answer:  "47"
    ***explain: "Premise 6 gives 45 credits, with 2 withdrawals in Year 1 (C1: 4 credits, C2: 3 credits, penalty 2 × 0.5 = 1 credit per Premise 1). Adjusted credits: 45 − 1 = 44. Premise 7 adds: S1 registered 15 credits, C3 (5 credits, withdrawn, 0 credits), C4 (3 credits, withdrawn, 0 credits), C5 (4 credits, passed, +4 credits), 2 withdrawals (penalty 1 credit). Total credits: 44 + 4 = 48 − 1 = 47.",
            
    ### Example 2:
    ***premises-NL: 
            1. "Nam has a GPA of 6.9 after completing 88 credits.",
            2. "Nam retook 3 previously passed courses: C1 (4 credits, initial 5.2, retake1 7.0, retake2 3.5), C2 (3 credits, initial 6.0, retake1 6.8, retake2 6.0), C3 (5 credits, initial 7.5, retake1 5.0, retake2 4.0).",
            3. "Nam repeated one failed course: C4 (3 credits, initial 3.0, repeat 6.5).",
            4. "Nam took two new courses: C5 (4 credits, score 8.0), C6 (2 credits, score 3.8).",
            5. "A retake is re-registering a passed course (same course code) to improve the grade.",
            6. "For GPA, use the highest score if ≥ 4.0; if all retake scores < 5.0, deduct 0.2 points per credit from the course’s grade points.",
            7. "Retaking a course does not add credits.",
            8. "Repeating a failed course means re-registering it.",
            9. "A failed course (score < 4.0) adds no credits.",
            10. "Repeating a failed course adds credits if the new score ≥ 4.0."
    ***question: "What are the final scores used for GPA in Nam’s retaken courses?"
    ***idx: [2,6]
    ***answer: "7.0, 6.8, 7.5"
    ***explain: "Premise 2 gives the scores for three retaken courses: C1 (4 credits, initial 5.2, retake1 7.0, retake2 3.5), C2 (3 credits, initial 6.0, retake1 6.8, retake2 6.0), C3 (5 credits, initial 7.5, retake1 5.0, retake2 4.0). Premise 6 says GPA uses the highest score if ≥ 4.0, with a penalty of 0.2 points per credit if all retake scores are < 5.0. For C1, Max(5.2, 7.0, 3.5) = 7.0, retakes 7.0, 3.5 (not all < 5.0), so score is 7.0. For C2, Max(6.0, 6.8, 6.0) = 6.8, retakes 6.8, 6.0 (not all < 5.0), so score is 6.8. For C3, Max(7.5, 5.0, 4.0) = 7.5, retakes 5.0, 4.0 (all < 5.0), but score remains 7.5 (penalty affects grade points, not score). Final scores: 7.0, 6.8, 7.5.",

    ### Example 3:
    ***premises-NL: 
            1. "A student can withdraw up to 3 courses per academic year if remaining credits ≥ 10 (Year 1, 2) or ≥ 8 (Year 3+); each withdrawal deducts 0.5 credits from total accumulated credits.",
            2. "Courses have 3, 4, or 5 credits; withdrawn courses contribute 0 credits to semester GPA.",
            3. "Students must register for 12–18 credits per semester; withdrawals reduce semester credits.",
            4. "No regulation limits total withdrawals, but max 3 per year.",
            5. "A student with < 8 accumulated credits cannot withdraw courses.",
            6. "A student (Year 2) has 45 credits, withdrew 2 courses in Year 1 (penalty 1 credit), including C1 (4 credits, withdrawn), C2 (3 credits, withdrawn).",
            7. "In Year 2, semester S1, the student registered for 15 credits, withdrew C3 (5 credits), C4 (3 credits), attempted C5 (4 credits, passed)."
        
    ***question: "How many credits were withdrawn in Year 2, semester S1, and can the student withdraw another course in semester S2 of Year 2?"
        
    ***idx: [1,3,5,7]
    ***answer: "8, No"
    ***explain: "Premise 7 gives S1 withdrawals: C3 (5 credits), C4 (3 credits), total 5 + 3 = 8 credits withdrawn. For S2 withdrawal (Premise 1): Year 2 requires remaining credits ≥ 10. Assume S2 registers 12 credits (Premise 3); withdrawing 3 credits leaves 12 − 3 = 9 < 10, fails. Premise 5: Total credits ≥ 8 (47 from Question 1), passes. Premise 1: 2 withdrawals in S1, 1 allowed in S2. Cannot withdraw in S2 due to insufficient semester credits."


    ### YOUR INPUT, FILL IN **answer, **idx, **explain
    ***premises-NL:
            1."A student can withdraw up to 3 courses per academic year if remaining credits ≥ 10 (Year 1, 2) or ≥ 8 (Year 3+); each withdrawal deducts 0.5 credits from total accumulated credits.",
            2. "Courses have 3, 4, or 5 credits; withdrawn courses contribute 0 credits to semester GPA.", 
            3. "Students must retake failed general courses until passed; no limit on attempts if fees paid.",
            4. "General GPA includes only passed courses; failed attempts contribute 0 credits.",
            5. "A student has 18 general credits, GPA 2.7, failed C1 (3 credits, grade 1.8, 2 attempts, $100 paid).",
            6. "The student retakes C1 (grade 2.2, $50), passes C2 (3 credits, grade 2.5), pays all fees."
        
    ***question: "What is the student’s updated general GPA after the new courses?"
    [/INST]
    [OUTPUT]
    ***answer:
    ***idx:
    ***explanation: """
        )




    # prompt này thì mình chỉ cần linh hoạt từ phần ### YOUR INPUT, FILL IN **explanation
    # mấy phần trên của prompt thì m giữ nguyên cũng được, few-shot thui 
    # Input: Logic program, premises-FOL, idx (từ prover9), answer(từ prover9), question(từ dataset, ko phải dạng FOL)
    prompt_using_LP = (
        """
    [INST] 
    You are an expert tutor in explaning first-order logic.
    Your explanations will be used for automated grading, so they **must** be precise, minimal, and follow the mandated format.


    I will give you:
    • A numbered list of premises in natural language.  
    • The same premises in FOL with matching numbers.  
    • One or more questions that logically follow from the premises.
    • For each question, a list called idx that names the exact premise numbers that justify the correct answer.  
    • The answer of the question. 
    • Logic Program Predicates: Each line follows: `PredicateName(args) means: Description`
    

    ### Your task
    1. For every question, use ONLY the premises whose numbers appear in its idx list.  
    2. Write a short, clear explanation that:  
    – cites those premises by number but do NOT write again the same premises,  
    – names the logical rules you apply (e.g., contraposition, modus ponens), and  
    – shows, step by step, how the answer follows from the cited premises, and 
    - If question is Multiple Choice Question, you need to explain why other options are wrong (in case your answer is too short).
    Ignore all premises that are **not** listed in idx.  
    3. Return the explanation in clear, brief, concise but making sure logical, the explanation must be a passage with maximum 3 sentences.

    ### Example 1: YES/NO QUESTION
    ***logic_program: 
            "Student(x) ::: x is a student.",
            "CompletedCoreCurriculum(x) ::: x has completed the core curriculum.",
            "PassedScienceAssessment(x) ::: x has passed the science assessment.",
            "EligibleForAdvancedCourses(x) ::: x is eligible for advanced courses.",
            "QualifiedForAdvancedCourses(x) ::: x is qualified for advanced courses.",
            "CompletedResearchMethodology(x) ::: x has completed research methodology.",
            "EligibleForInternationalProgram(x) ::: x is eligible for the international program.",
            "PassedLanguageProficiencyExam(x) ::: x has passed the language proficiency exam.",
            "CompletedCapstoneProject(x) ::: x has completed the capstone project.",
            "AwardedHonorsDiploma(x) ::: x has been awarded an honors diploma.",
            "CompletedCommunityService(x) ::: x has completed community service.",
            "QualifiesForUniversityScholarship(x) ::: x qualifies for the university scholarship.",
            "ReceivedFacultyRecommendation(x, y) ::: x has received a faculty recommendation for y.",
            "Name(x, \"Sophia\") ::: x is named Sophia.",
            "Sophia(x) ::: x is Sophia.",
            "Agent(Sophia) ::: Sophia is an agent.",
            "Verb(Completed) ::: completed is a verb.",
            "Object(ResearchMethodologyCourse) ::: research methodology course is the object of the verb completed.",
            "Knowledge(Sophia, Course) ::: Sophia has knowledge of the course.",
            "CapstoneProject(y) ::: y is the capstone project.",
            "Completed(x, y) ::: x has completed y.",
            "Adjective(Required) ::: required is an adjective.",
            "Noun(CommunityServiceHours) ::: community service hours is a noun.",
            "HoursMet(Sophia) ::: Sophia has met the required hours.",
            "QualifiesForScholarship(Sophia) ::: Sophia qualifies for the scholarship.",
            "MeetsEligibilityCriteria(Sophia) ::: Sophia meets the eligibility criteria.",
            "Scholarship(y) ::: y is a scholarship.",
            "NeedsFacultyRecommendation(x, y) ::: x needs a faculty recommendation for y.",
            "InternationalProgram(y) ::: y is an international program.",
            "EligibleFor(x, y) ::: x is eligible for y.",
            "NeedsLanguageProficiencyExamPass(Sophia) ::: Sophia needs to pass the language proficiency exam.",
            "HonorsDiploma(Sophia) ::: Sophia has been awarded an honors diploma.",
            "UniversityScholarship(y) ::: y is a university scholarship.",
            "Premises(z) ::: z are premises.",
            "Qualifies(x, y, z) ::: x"
        
    ***premises-FOL: 
            1. "ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))",
            2. "ForAll(x, (qualified_for_advanced_courses(x) ∧ completed_research_methodology(x)) → eligible_for_international_program(x))",
            3. "ForAll(x, passed_language_proficiency(x) → eligible_for_international_program(x))",
            4. "ForAll(x, (eligible_for_international_program(x) ∧ completed_capstone_project(x)) → awarded_honors_diploma(x))",
            5. "ForAll(x, (awarded_honors_diploma(x) ∧ completed_community_service(x)) → qualifies_for_scholarship(x))",
            6. "ForAll(x, (awarded_honors_diploma(x) ∧ received_faculty_recommendation(x)) → qualifies_for_scholarship(x))",
            7. "completed_core_curriculum(Sophia)",
            8. "passed_science_assessment(Sophia)",
            9. "completed_research_methodology(Sophia)",
            10. "completed_capstone_project(Sophia)",
            11. "completed_community_service(Sophia)"
        ]
    ***idx: [1,2,4,5,7,8,9,10,11]
    ***question: "Does Sophia qualify for the university scholarship, according to the premises?"
    ***answer:  "Yes"
    ***explain: "The university scholarship per premise 5 requires an honors diploma and community service. Premise 4 requires international program eligibility and a capstone project, premise 2 requires advanced course qualification and research methodology, and premise 1 requires core curriculum and science assessment. Premises 7–11 confirm Sophia’s curriculum, assessment, methodology, capstone, and service, satisfying all conditions for the scholarship."
    
    ### Example 2: MULTIPLE CHOICE QUESTION
    ***logic_program: 
            "FacultyMember(x) ::: Person(x)",
            "TaughtForAtLeast5Years(x) ::: Person(x) & HasTaughtForAtLeastYears(x, 5)",
            "EligibleForExtendedLibraryAccess(x) ::: Person(x) & CanAccessRestrictedArchives(x) & CompletedResearchEthicsTraining(x)",
            "Person(x) ::: x is a Person",
            "ExtendedLibraryAccess(x) ::: Person(x)",
            "PublishedAcademicPaper(x) ::: Person(x) & HasPublishedAcademicPaper(x)",
            "CanAccessRestrictedArchives(x) ::: Person(x)",
            "CompletedResearchEthicsTraining(x) ::: Person(x)",
            "CanSubmitResearchProposals(x) ::: Person(x) & HasDepartmentalEndorsement(x)",
            "HasDepartmentalEndorsement(x) ::: Person(x)",
            "CanApplyForCollaborativeResearchProjects(x) ::: Person(x)",
            "AccessRestrictedArchives(x) ::: Person(x)",
            "SubmitProposals(x) ::: Person(x)",
            "LogicalChain(x) ::: Person(x)"
    ***premises-FOL: 
            1. "ForAll(x, taught_min_five_years(x) → eligible_extended_library(x))",
            2. "ForAll(x, (eligible_extended_library(x) ∧ has_publications(x)) → can_access_restricted_archives(x))",
            3. "ForAll(x, (can_access_restricted_archives(x) ∧ completed_ethics_training(x)) → can_submit_research_proposals(x))",
            4. "ForAll(x, (can_submit_research_proposals(x) ∧ has_departmental_endorsement(x)) → can_apply_collaborative_projects(x))",
            5. "taught_min_five_years(John)",
            6. "has_publications(John)",
            7. "completed_ethics_training(John)",
            8. "has_departmental_endorsement(John)"
        ]
    ***idx: [1,2,3,4,5,6,7,8]
    ***question: "Based on the premises, what is the correct conclusion about Professor John?\nA. He can access restricted archives but cannot submit proposals\nB. He can apply for collaborative research projects\nC. He needs more publications to access archives\nD. He is eligible for extended library access but cannot apply for projects"
    ***answer:  "B"
    ***explain: "Professor John has taught for at least 5 years (Premise 5), so he is eligible for extended library access (by Premise 1). Because he has at least one publication (Premise 6), he can access restricted archives (Premise 2). With completed ethics training (Premise 7), he can submit research proposals (Premise 3), and with departmental endorsement (Premise 8), he can apply for collaborative research projects (Premise 4). Hence, option B is logically valid."
           

    ### YOUR INPUT, FILL IN **explanation 
    ***logic_program:
            "PythonCode(x) ::: If a Python code is well-tested, then the project is optimized.",
            "WellTested(x) ::: If a Python code does not follow PEP 8 standards, then it is not well-tested.",
            "OptimizedProject(x) ::: All Python projects are easy to maintain.",
            "PEP8Standards(x) ::: If a Python code follows PEP 8 standards, then it is easy to maintain.",
            "PythonProject(x) ::: If a Python code is well-tested, then it follows PEP 8 standards.",
            "EasyToMaintain(x) ::: If a Python project is well-structured, then it is optimized.",
            "FollowsPEP8Standards(x) ::: If a Python code is well-tested, then it follows PEP 8 standards.",
            "WellTestedPythonCode(x) ::: If a Python project is well-structured, then it has clean and readable code.",
            "WellStructured(x) ::: If a Python project is well-tested, then it has clean and readable code.",
            "Optimized(x) ::: If a Python project is easy to maintain, then it is optimized.",
            "CleanCode(x) ::: If a Python code is well-tested, then it has clean and readable code.",
            "ReadableCode(x) ::: If a Python project is well-structured, then it has clean and readable code.",
            "Person(y) ::: There exists at least one Python project that follows best practices.",
            "Perfect(y) ::: There exists at least one Python project that is optimized.",
            "FollowsBestPractices(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards.",
            "Clean(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards.",
            "Readable(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards."
    ***premises-FOL: 
        1. "ForAll(c, (well_structured(c) ∧ has_exercises(c)) → enhances_engagement(c))",
        2. "ForAll(c, (enhances_engagement(c) ∧ advanced_resources(c)) → enhances_critical_thinking(c))",
        3. "ForAll(f, (pedagogical_training(f) ∧ curriculum_development(f)) → well_structured(curriculum))",
        4. "pedagogical_training(faculty) ∧ curriculum_development(faculty)",
        5. "has_exercises(curriculum)",
        6. "advanced_resources(curriculum)"
    ***idx: [ 7, 10]
    ***question:"Does it follow that if all Python projects are well-structured, then all Python projects are optimized, according to the premises?"
    ***answer:  "Yes"   
    [/INST]
    [OUTPUT]
    ***explanation: """
        )
    
    # Còn này dùng premises-NL, ae tự tin vào LP thì dùng LP, nếu đúng thì impact mik mạnh hơn, hoặc dùng 
    # premises - NL thì siêu ổn nha
    # # Input: Premises_NL, premises-FOL, idx (từ prover9), answer(từ prover9), question(từ dataset, ko phải dạng FOL)
    prompt_using_premises_NL = (
        """
    [INST] 
    You are an expert tutor in explaning first-order logic.
    Your explanations will be used for automated grading, so they **must** be precise, minimal, and follow the mandated format.


    I will give you:
    • A numbered list of premises in natural language.  
    • The same premises in FOL with matching numbers.  
    • One or more questions that logically follow from the premises.
    • For each question, a list called idx that names the exact premise numbers that justify the correct answer.  
    • The answer of the question. 
    

    ### Your task
    1. For every question, use ONLY the premises whose numbers appear in its idx list.  
    2. Write a short, clear explanation that:  
    – cites those premises by number but do NOT write again the same premises,  
    – names the logical rules you apply (e.g., contraposition, modus ponens), and  
    – shows, step by step, how the answer follows from the cited premises, and 
    - If question is Multiple Choice Question, you need to explain why other options are wrong (in case your answer is too short).
    Ignore all premises that are **not** listed in idx.  
    3. Return the explanation in clear, brief, concise but making sure logical, the explanation must be a passage with maximum 3 sentences.

    ### Example 1: YES/NO QUESTION
    ***premises-NL: 
            1. "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
            2. "Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.",
            3. "Students who have passed the language proficiency exam are eligible for the international program.",
            4. "Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.",
            5. "Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.",
            6. "Students who have been awarded an honors diploma and have received a faculty recommendation qualify for the university scholarship.",
            7. "Sophia has completed the core curriculum.",
            8. "Sophia has passed the science assessment.",
            9. "Sophia has completed the research methodology course.",
            10. "Sophia has completed her capstone project.",
            11. "Sophia has completed the required community service hours."
        
    ***premises-FOL: 
            1. "ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))",
            2. "ForAll(x, (qualified_for_advanced_courses(x) ∧ completed_research_methodology(x)) → eligible_for_international_program(x))",
            3. "ForAll(x, passed_language_proficiency(x) → eligible_for_international_program(x))",
            4. "ForAll(x, (eligible_for_international_program(x) ∧ completed_capstone_project(x)) → awarded_honors_diploma(x))",
            5. "ForAll(x, (awarded_honors_diploma(x) ∧ completed_community_service(x)) → qualifies_for_scholarship(x))",
            6. "ForAll(x, (awarded_honors_diploma(x) ∧ received_faculty_recommendation(x)) → qualifies_for_scholarship(x))",
            7. "completed_core_curriculum(Sophia)",
            8. "passed_science_assessment(Sophia)",
            9. "completed_research_methodology(Sophia)",
            10. "completed_capstone_project(Sophia)",
            11. "completed_community_service(Sophia)"
        ]
    ***idx: [1,2,4,5,7,8,9,10,11]
    ***question: "Does Sophia qualify for the university scholarship, according to the premises?"
    ***answer:  "Yes"
    ***explain: "The university scholarship per premise 5 requires an honors diploma and community service. Premise 4 requires international program eligibility and a capstone project, premise 2 requires advanced course qualification and research methodology, and premise 1 requires core curriculum and science assessment. Premises 7–11 confirm Sophia’s curriculum, assessment, methodology, capstone, and service, satisfying all conditions for the scholarship."
    
    ### Example 2: MULTIPLE CHOICE QUESTION
    ***premises-NL:
            1. "If a faculty member has taught for at least 5 years, they are eligible for extended library access.",
            2. "If someone has extended library access and has published at least one academic paper, they can access restricted archives.",
            3. "If someone can access restricted archives and has completed research ethics training, they can submit research proposals.",
            4. "If someone can submit research proposals and has a departmental endorsement, they can apply for collaborative research projects.",
            5. "Professor John has taught for at least 5 years.",
            6. "Professor John has published at least one academic paper.",
            7. "Professor John has completed research ethics training.",
            8. "Professor John has a departmental endorsement."
    ***premises-FOL: 
            1. "ForAll(x, taught_min_five_years(x) → eligible_extended_library(x))",
            2. "ForAll(x, (eligible_extended_library(x) ∧ has_publications(x)) → can_access_restricted_archives(x))",
            3. "ForAll(x, (can_access_restricted_archives(x) ∧ completed_ethics_training(x)) → can_submit_research_proposals(x))",
            4. "ForAll(x, (can_submit_research_proposals(x) ∧ has_departmental_endorsement(x)) → can_apply_collaborative_projects(x))",
            5. "taught_min_five_years(John)",
            6. "has_publications(John)",
            7. "completed_ethics_training(John)",
            8. "has_departmental_endorsement(John)"
        ]
    ***idx: [1,2,3,4,5,6,7,8]
    ***question: "Based on the premises, what is the correct conclusion about Professor John?\nA. He can access restricted archives but cannot submit proposals\nB. He can apply for collaborative research projects\nC. He needs more publications to access archives\nD. He is eligible for extended library access but cannot apply for projects"
    ***answer:  "B"
    ***explain: "Professor John has taught for at least 5 years (Premise 5), so he is eligible for extended library access (by Premise 1). Because he has at least one publication (Premise 6), he can access restricted archives (Premise 2). With completed ethics training (Premise 7), he can submit research proposals (Premise 3), and with departmental endorsement (Premise 8), he can apply for collaborative research projects (Premise 4). Hence, option B is logically valid."
           

    ### YOUR INPUT, FILL IN **explanation 
    ***logic_program:
            "Student(x) :: If x is a student.",
            "CompletesRequiredCourses(x) :: If x completes all required courses.",
            "EligibleForGraduation(x) :: If x is eligible for graduation.",
            "MaintainsHighGPA(x, 3.5) :: If x maintains a GPA of at least 3.5.",
            "GraduateWithHonors(x) :: If x graduates with honors.",
            "GraduatesWithHonors(x) :: If x graduates with honors.",
            "CompletesThesis(x) :: If x completes a thesis.",
            "ReceivesAcademicDistinction(x) :: If x receives academic distinction.",
            "QualifiesForGraduateFellowshipProgram(x) :: If x qualifies for the graduate fellowship program.",
            "John(x) :: If x is John.",
            "CompletedCourses(x) :: If x has completed courses.",
            "RequiredCourses(x) :: If x has completed all required courses.",
            "MaintainsGPA(John, 3.8) :: If John maintains a GPA of 3.8.",
            "Thesis(y) :: If y is a thesis.",
            "Completed(z) :: If z has been completed.",
            "Relationship(x, y, z) :: If there is a relationship between x, y, and z.",
            "QualifiesForGraduateFellowship(John) :: If John qualifies for the graduate fellowship program.",
            "Fellowship(y) :: If y is a fellowship.",
            "FacultyRecommendation(x, y) :: If a faculty member recommends x.",
            "RequiredForApplication(y) :: If y is required for application.",
            "Internship(x) :: If x has completed an internship.",
            "Completed(x, internship) :: If x has completed an internship.",
            "Qualified(x) :: If x is qualified.",
            "GPA(x) :: If x has a GPA.",
            "BelowThreshold(x, HonorsThreshold) :: If x's GPA is below the honors threshold.",
            "InsufficientForHonors(x) :: If x's GPA is insufficient for honors.",
            "AcademicDistinction(x) :: If x receives academic distinction.",
            "PublishedWork(x) :: If x has published work.",
            "MeetsCriteria(x, DistinctionCriteria) :: If x meets the criteria for academic distinction.",
            "ReceivesDistinction(x) :: If x receives academic distinction."
    ***premises-FOL: 
            1. "ForAll(x, (passed_vehicle_inspection(x) ∧ has_appropriate_license(x)) → can_transport_standard_goods(x))",
            2. "ForAll(x, (can_transport_standard_goods(x) ∧ completed_hazmat_training(x) ∧ received_safety_endorsement(x)) → can_transport_hazardous_materials(x))",
            3. "ForAll(x, (can_transport_hazardous_materials(x) ∧ has_interstate_permit(x)) → can_cross_state_lines(x))",
            4. "passed_vehicle_inspection(John)",
            5. "has_appropriate_license(John)",
            6. "completed_hazmat_training(John)",
            7. "¬received_safety_endorsement(John)",
            8. "has_interstate_permit(John)"
    ***idx: [1, 2, 3, 4, 5, 6, 7, 8]
    ***question:     "Does John meet all requirements to cross state lines with hazardous cargo, according to the premises?" 
    ***answer:  "No"   
    [/INST]
    [OUTPUT]
    ***explanation: """
    )


    prompt_reasoning = """<s>[INST]
        ### Task: Generate **detailed, smooth and fully meaning** the explanation and reasoning by not only applying methods of inference, contradiction, and other techniques in the theory of First-Order Logic (FOL) but also connecting between natural language (NL) and First-Order Logic (FOL).
        You are given:
        - Logic Program: A list of First-Order Logic (FOL) predicates along with their definitions. It help you more understand the context of each provided FOL Predicate in Premises-FOL and Questions-FOL. See the example below to clearly understand **the format** between FOL predicates and their definitions:
        + Teacher(x) ::: x is a Teacher.
        + DoTestSubject(x, y) ::: student x do test on y subject.
        - Premises-FOL: A list of First-Order Logic (FOL) statements, where each FOL statement represents a premise describing a context, domain, or situation within the scope of FOL by combining the predicates, quantifiers and logical connectives. Ensure the premises reflect meaningful and logically connected relationships.
        - Questions-NL: A List of natural language (NL) questions, where each question can take the form of a **Yes/No/Uncertain question**, an **enumeration**, or an open-ended logical query.
        - Questions-FOL: A List of First-Order Logic (FOL) questions. It is similar to Questions-NL, but written in the form of FOL.
        - Answer: A list of the responses for the corresponding questions. The answer at position i can have a response of Yes, No or Uncertain; or be one of the four options (A, B, C, or D); or a value demand reasoning, analysis, and detailed explanations based on logical inference. Besides, it can include two of the three response types mentioned above when the question at position i addresses multiple issues.
        - Idx: A large list of sublists, where each list of index corresponds to *the positions in the list of statement* (Premises-FOL) used to give the answer for the respective question.

        Please follow these instructions carefully:
        1. Logical Reasoning
        - Use the information in **Premises-FOL** to infer and support your reasoning for each question.
        - Ensure you identify relevant predicates, quantifiers, and logical connectives from **Premises-FOL** that are connected to each question. Apply these FOL elements to reason through the answer logically.
        - When answering **Questions-FOL**, make sure to use formal logical methods, such as contradiction, equivalence, or inference, where necessary. Be clear about the steps of reasoning that lead to your conclusion.

        2. Generate Explanation
        - Rather than simply providing the answer, you must **explain** how the information from the input (Logic Program, Premises-FOL, Questions-NL, Questions-FOL, and Answer) supports the answer.
        - Ensure that the explanation is grounded in both the natural language reasoning (from Questions-NL, Logic Program) and the formal logical reasoning (from Premises-FOL with corresponde list of index, Questions-FOL and Logic Program).
        - The explanation should **connect the relevant premises, predicates, and logical relations** to the final answer, detailing how those elements contribute to the conclusion.
        - The explanation **can be provided in both natural language and First-Order Logic**.

        **Important Rules**:
        - The list of indices at position i represents the positions of the statements used to generate the answer at position i for the corresponding question at position i. All the data provided above serve as the basis for you to generate the explanation corresponding to answer i.
        - Consider that answers at position i may involve multiple response types (Yes, No, Uncertain, or the four options A, B, C, D) if the question addresses multiple issues or requires reasoning with detailed analysis.
        - When providing reasoning, ensure that you explain **the steps and logic** behind your conclusion, showing how the given premises and logical rules lead to the answer. For example, in cases of contradiction or inference, clearly outline how the reasoning process leads to a specific conclusion.
        - Focus on providing an explanation. Explain **why and how** the premises lead to the conclusion, referencing the logical relationships.
        - Do not use the example in the output.

        ### Input:
        - Logic Program: {logic_program}
        - Premises-FOL: {premises_fol}
        - Questions-NL: {questions_nl}
        - Questions-FOL: {questions_fol}
        - Answer: {answer}
        - Idx: {idx}
    Output: </s>"""
    

    # result = fo_logic_eval(prompt_using_LP)
    # print(result)
    # end_time = time.time()
    # print(end_time - start_time)





































# #  """
# #     [INST] 
# #     You are an expert tutor in explaning first-order logic.
# #     Your explanations will be used for automated grading, so they **must** be precise, minimal, and follow the mandated format.


# #     I will give you:
# #     • A numbered list of premises in natural language.  
# #     • The same premises in FOL with matching numbers.  
# #     • One or more questions that logically follow from the premises.
# #     • For each question, a list called idx that names the exact premise numbers that justify the correct answer.  
# #     • The answer of the question. 
# #     • Logic Program Predicates: Each line follows: `PredicateName(args) means: Description`
    

# #     ### Your task
# #     1. For every question, use ONLY the premises whose numbers appear in its idx list.  
# #     2. Write a short, clear explanation that:  
# #     – cites those premises by number but do NOT write again the same premises,  
# #     – names the logical rules you apply (e.g., contraposition, modus ponens), and  
# #     – shows, step by step, how the answer follows from the cited premises, and 
# #     - If question is Multiple Choice Question, you need to explain why other options are wrong (in case your answer is too short).
# #     Ignore all premises that are **not** listed in idx.  
# #     3. Return the explanation in clear and short but making sure logical, the explanation must be a passage with maximum 3 sentences.

# #     ### Example 1: YES/NO QUESTION
# #     ***Premises-NL: 
# #             1. "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
# #             2. "Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.",
# #             3. "Students who have passed the language proficiency exam are eligible for the international program.",
# #             4. "Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.",
# #             5. "Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.",
# #             6. "Students who have been awarded an honors diploma and have received a faculty recommendation qualify for the university scholarship.",
# #             7."Sophia has completed the core curriculum.",
# #             8. "Sophia has passed the science assessment.",
# #             9. "Sophia has completed the research methodology course.",
# #             10. "Sophia has completed her capstone project.",
# #             11. "Sophia has completed the required community service hours."
# #     ***premises-FOL: 
# #             1. "ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))",
# #             2. "ForAll(x, (qualified_for_advanced_courses(x) ∧ completed_research_methodology(x)) → eligible_for_international_program(x))",
# #             3. "ForAll(x, passed_language_proficiency(x) → eligible_for_international_program(x))",
# #             4. "ForAll(x, (eligible_for_international_program(x) ∧ completed_capstone_project(x)) → awarded_honors_diploma(x))",
# #             5. "ForAll(x, (awarded_honors_diploma(x) ∧ completed_community_service(x)) → qualifies_for_scholarship(x))",
# #             6. "ForAll(x, (awarded_honors_diploma(x) ∧ received_faculty_recommendation(x)) → qualifies_for_scholarship(x))",
# #             7. "completed_core_curriculum(Sophia)",
# #             8. "passed_science_assessment(Sophia)",
# #             9. "completed_research_methodology(Sophia)",
# #             10. "completed_capstone_project(Sophia)",
# #             11. "completed_community_service(Sophia)"
# #         ]
# #     ***idx: [1,2,4,5,7,8,9,10,11]
# #     ***question: "Does Sophia qualify for the university scholarship, according to the premises?"
# #     ***answer:  "Yes"
# #     ***explain: "The university scholarship per premise 5 requires an honors diploma and community service. Premise 4 requires international program eligibility and a capstone project, premise 2 requires advanced course qualification and research methodology, and premise 1 requires core curriculum and science assessment. Premises 7–11 confirm Sophia’s curriculum, assessment, methodology, capstone, and service, satisfying all conditions for the scholarship."
    
# #     ### Example 2: MULTIPLE CHOICE QUESTION
# #     ***Premises-NL: 
# #             1. "If a faculty member has taught for at least 5 years, they are eligible for extended library access.",
# #             2. "If someone has extended library access and has published at least one academic paper, they can access restricted archives.",
# #             3. "If someone can access restricted archives and has completed research ethics training, they can submit research proposals.",
# #             4. "If someone can submit research proposals and has a departmental endorsement, they can apply for collaborative research projects.",
# #             5. "Professor John has taught for at least 5 years.",
# #             6. "Professor John has published at least one academic paper.",
# #             7. "Professor John has completed research ethics training.",
# #             8. "Professor John has a departmental endorsement."
# #     ***premises-FOL: 
# #             1. "ForAll(x, taught_min_five_years(x) → eligible_extended_library(x))",
# #             2. "ForAll(x, (eligible_extended_library(x) ∧ has_publications(x)) → can_access_restricted_archives(x))",
# #             3. "ForAll(x, (can_access_restricted_archives(x) ∧ completed_ethics_training(x)) → can_submit_research_proposals(x))",
# #             4. "ForAll(x, (can_submit_research_proposals(x) ∧ has_departmental_endorsement(x)) → can_apply_collaborative_projects(x))",
# #             5. "taught_min_five_years(John)",
# #             6. "has_publications(John)",
# #             7. "completed_ethics_training(John)",
# #             8. "has_departmental_endorsement(John)"
# #         ]
# #     ***idx: [1,2,3,4,5,6,7,8]
# #     ***question: "Based on the premises, what is the correct conclusion about Professor John?\nA. He can access restricted archives but cannot submit proposals\nB. He can apply for collaborative research projects\nC. He needs more publications to access archives\nD. He is eligible for extended library access but cannot apply for projects"
# #     ***answer:  "B"
# #     ***explain: "Professor John has taught for at least 5 years (Premise 5), so he is eligible for extended library access (by Premise 1). Because he has at least one publication (Premise 6), he can access restricted archives (Premise 2). With completed ethics training (Premise 7), he can submit research proposals (Premise 3), and with departmental endorsement (Premise 8), he can apply for collaborative research projects (Premise 4). Hence, option B is logically valid."
           

# #     ### YOUR INPUT, FILL IN **explanation 
# #     ***premises-NL: 
# #             1. "If a student completes all required courses, they are eligible for graduation.",
# #             2. "If a student is eligible for graduation and maintains a GPA above 3.5, they graduate with honors.",
# #             3. "If a student graduates with honors and completes a thesis, they receive academic distinction.",
# #             4. "If a student receives academic distinction, they qualify for the graduate fellowship program.",
# #             5. "John has completed all required courses.",
# #             6. "John maintains a GPA of 3.8.",
# #             7. "John has completed a thesis."
# #     ***premises-FOL: 
# #             1. "ForAll(x, completed_required_courses(x) → eligible_for_graduation(x))",
# #             2. "ForAll(x, (eligible_for_graduation(x) ∧ gpa_above_3_5(x)) → graduates_with_honors(x))",
# #             3. "ForAll(x, (graduates_with_honors(x) ∧ completed_thesis(x)) → academic_distinction(x))",
# #             4. "ForAll(x, academic_distinction(x) → qualifies_for_fellowship(x))",
# #             5. "completed_required_courses(John)",
# #             6. "gpa_above_3_5(John)",
# #             7. "completed_thesis(John)"
# #     ***idx: [1, 2, 3, 4, 5, 6, 7]
# #     ***question:  "Based on the above premises, which conclusion logically follows?\nA. John qualifies for the graduate fellowship program\nB. John needs faculty recommendation for the fellowship\nC. John must complete an internship to qualify\nD. John’s GPA is insufficient for honors"           
# #     ***answer:  "A"   
# #     [/INST]
# #     [OUTPUT]
# #     ***explanation: """


# import re
# from typing import List, Tuple
# import spacy
# import time

# # load pretrained spaCy pipeline with NER once
# _nlp = spacy.load("en_core_web_sm")
# _QUANTIFIERS_LOGIC = (
#     "all ", "everyone ", "any ", "anyone ", "each ", "either "
# )

# _QUANTIFIERS_EXIST = (
#     "there exists ", "there is ", "at least one ", "every "
# )

# _RELATIVES = (
#     "whoever", "wherever", "whenever", "when"
# )

# def check_premise_impliction(premise: str):
#     doc = _nlp(premise)
#     premise = premise.lower()
#     if re.search(r'\b(only if|if|unless|provided that|in case|assuming that)\b.*?', premise, re.IGNORECASE):
#         return True
#     elif re.search(r'^if .+ then .+', premise, re.IGNORECASE):
#         return True
#     elif re.search(r'\b(will|would|must|should|can|could|or)\b', premise, re.IGNORECASE):
#         return True
#     elif premise.startswith(_RELATIVES):
#         return True
#     elif premise.startswith(_QUANTIFIERS_LOGIC):
#         return True
#     elif premise.startswith(_QUANTIFIERS_EXIST):
#         return False
#     if re.search(r'\b(due to|owing to|because of|thanks to|on account of|resulting from|according to)\b', premise.lower(), re.IGNORECASE):
#         return True
#     if re.search(r'\b(because|so that|in order that|in order to|as a result|therefore|consequently|accordingly|thus|hence|since|thereby|so)\b', premise.lower(), re.IGNORECASE):
#         return True
#     if re.search(r'\b(results in|result in|leads to|lead to|requires|require|causes|cause|means|mean)\b', premise.lower(), re.IGNORECASE):
#         return True
#     return False

# import json

# # Đọc dữ liệu từ file JSON
# with open("/data/npl/ViInfographicCaps/Contest/final_contest/final_code/data/train_v2_ful_fol.json", 'r', encoding='utf-8') as f:
#     data = json.load(f)

# if_sen = []

# # Cái này cho mệnh đề chứa if (này dễ nhận ra, test mức dễ trước)
# for item in data:
#     premises_NL = item.get("premises-NL", [])
#     for sentence in premises_NL:
#         if "if " in sentence.lower():  # lowercase để gọn hơn
#             if_sen.append(sentence)
# if_sen = list(set(if_sen))


# # Này test mức nâng cao hơn 
# high_level_sen = []

# for item in data:
#     premises_NL  = item.get("premises-NL", [])
#     premises_FOL = item.get("premises-FOL", [])
#     fol_llm      = item.get("LLM-FOL", [])

#     min_len = min(len(premises_NL), len(premises_FOL), len(fol_llm))

#     for i in range(min_len):
#         if premises_FOL[i] is None or fol_llm[i] is None:
#             continue
#         if "→" in premises_FOL[i] and "→" in fol_llm[i]:
#             high_level_sen.append(premises_NL[i])


# high_level_sen = list(set(high_level_sen))

# for sentence in high_level_sen:
# # sentence = "Students with ≥ 80 credits and < 100 CTXH hours can join extracurriculars, adding 10 CTXH hours."
#     if check_premise_impliction(sentence):
#         continue
#     else:
#         print("[NOT implication]:", sentence)

