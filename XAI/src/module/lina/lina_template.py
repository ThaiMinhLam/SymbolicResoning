# FEATURE EXTRACTION
'''
    Idea:
    - Paraphrase lại đoạn văn dài thành gọn hơn. Apply NER?? Extract obj và rel của context và câu hỏi (Chung 1 module)
        + Extract context trước --> feed obj, rel của context vào extract câu hỏi --> Dựa vào Obj của context mà extract question
        + Nếu câu hỏi chứa obj không có trong context --> Tạo obj ứng với câu hỏi
        + Idea là theo flow như này, Chain of Thoughts:
            * Context: enroll_class_C -> well_qualified
            * Câu hỏi: He is Eligible for class C, he is well qualified
                * Step 1: He is eligible for class C, he is <well_qualified> (Tại vì well_qualified có trong context)
                * Step 2: He is <eligible_class_C>, he is <well_qualified> (Fill hết obj có rồi -> Fill đến những cái chưa có)
                * Step 3: eligible_class_C -> well_qualified

    - Bắt generate từ NL ra các FOL trong đó các obj trong câu hỏi phải xuất hiện ít nhất 1 lần
    - Filter ra các premise nào phù hợp và có liên quan tới obj của câu hỏi
    - Giả xử luôn tồn tại các chuỗi rel giữa các obj liên tục nối giữa các câu FOL với nhau, sẽ có các rel ẩn đi cắt đứt chuỗi liên tục đó --> Cần tìm những FOL đó:
        + Nếu task là multiple choice bắt buộc phải có ràng buộc liên tục
        + Nếu task là binary choice, ko cần thiết tại có tag uncertain
        + List ra các premises support cho câu claim: Xét tính liên tục
        ** Xuất hiện trường hợp obj trong câu hỏi ko có trong câu trả lời -> Lúc extract rel ẩn cần thêm vào meta data là câu hỏi vào 
    - Sau khi extract được hết tất cả các mối quan hệ, xác định obj_head và obj_tail và tìm chuỗi liên tục nối giữa
    - Explaination ignore đi những chuỗi FOL ẩn, chỉ explain dựa trên các premises có sẵn.
    - Có template prompt cho từng loại câu hỏi:
        + Nếu là A, B, C, D: Xét tuyến tính xún: Xét các chuỗi ko liên quan đến rel ẩn rồi tới có liên quan, loại bỏ đi dần đến khi thỏa số lượng đáp án yêu cầu
        + Nếu là là Yes, No, Uncertain -> Ez, chỉ xét 1 record check đúng sai
        + List ra các premises support cho câu claim: Dựa vào chuỗi liên tục để extract ra idx
    - Explaination: Dựa vào chuỗi liên tục
'''

def INSPECT_EXAMPLES():
    #  ----
    #         "Formal Logical Expression 1" is:
    #         "Formal Logical Expression 2" is:
    #         "Formal Logical Expression 3" is:
    #         ...
    #         "Formal Logical Expression n" is:
    # -----
    # "premises-NL": [
    #         "If a student completes Course A, they can enroll in Course B.",
    #         "If a student enrolls in Course B and passes it, they can enroll in Course C.",
    #         "Enrollment in Course C makes a student eligible for the internship program.",
    #         "David has completed Course A.",
    #         "David has enrolled in and passed Course B."
    #     ],
    #     "premises-FOL": [
    #         "ForAll(x, complete(x, A) → enroll(x, B))",
    #         "ForAll(x, (enroll(x, B) ∧ pass(x, B)) → enroll(x, C))",
    #         "ForAll(x, enroll(x, C) → eligible_internship(x))",
    #         "complete(david, A)",
    #         "enroll(david, B) ∧ pass(david, B)"
    #     ],
    #     "questions": [
    #         "Based on the prerequisites, what is David's current eligibility status?A. Eligible for Course C but not the internship\nB. Eligible for the internship program\nC. Needs to pass Course B first\nD. Only eligible for Course B",
    #         "Does the logical progression demonstrate that David meets all requirements for the internship?"
    #     ],
    #     "answers": [
    #         "B",
    #         "Yes"
    #     ],
    #     "idx": [
    #         [
    #             2,
    #             3,
    #             5
    #         ],
    #         [
    #             2,
    #             3,
    #             5
    #         ]
    #     ],
    #     "explanation": [
    #         "Premise 5 confirms David enrolled in and passed Course B, satisfying premise 2's conditions to enroll in Course C. Premise 3 states that enrollment in Course C makes David eligible for the internship program, supporting option B. Option A is false because eligibility for Course C implies internship eligibility. Option C is false because David has passed Course B. Option D is false because David is eligible beyond Course B, for Course C and the internship.",
    #         "The requirements for the internship per premise 3 involve enrolling in Course C, which per premise 2 requires enrolling in and passing Course B. Premise 5 confirms David enrolled in and passed Course B, enabling Course C enrollment and thus internship eligibility."
    #     ]
    return

def HEAD_INSTRUCTION():
    return '''
        You are an AI reasoning assistant. Your task is to understand and answer questions based on events and rules that happen in an educational setting. The dataset contains simple, rule-based statements (like course completions, enrollments, and eligibility) and facts about individuals (like students).
        Your job is to:
        - The event must include both verb and object related to that event (submit a form → submit_form)
        - Identify logical relationships between events (e.g., "If A, then B"),
        - Track facts about entities (like what courses David completed),
        - Infer new facts by chaining rules together (e.g., if David completed_A and passed_B, then he may be David eligible_for_something),
        - Keep all names and objects consistent (do not rename),
        Avoid subject-specific knowledge—the task is about reasoning in an educational context, not teaching content.
    '''
# ---------------------------------------------------------------
def PARAPHRASE_PROMPT_TEMPLATE():
    return '''
        {instruction}
        Please simplify the following logic problem statement and convert it into a "Simplified Statement". Each "Simplified Statement" derive 3 sections "List Predicates", "List Instances".
            - Each Statement is seperated by dot formula ".". 
            - Only remove stopwords and unnecessary words in the statement, in order to keep all clause meaning in the statements with few changing.
            - Ensure that the simplified information maintains all logical relationships of the original statement. 
            - The number of "Simplified Statement" must be equal to the number of questions statement. "List Predicates" and "List Instances" number i is infered from corresponding "Simplified Statement" number i.
            - "List Predicates" contains the event of the statement, this is the combination of 1 action and 1 object. 
                + *object* contains the event of the statement, usually a noun. 
                + *actions* contains the action on the event of the statement, usually a verb, adjective, comparison symbols, etc.
                + Each predicate is formatted by this following python code: predicate = action + "_" + object
            - "List Instances" is the instance of the object, it is a specific name of a person, place, thing, idea or sometime a number (include float and integer number), etc.
        You can add "Important Infomation" by extract the core information from each statement items that you think important but is not appropriate to parallel with the Simplified Statements, focus on one or more of the following:
            - The meaning of a specific phrase or term in the statement,
            - A mathematical solution or equation relevant to the statement,
            - A logical deduction or inference based on the scenario described.
            - This information should help clarify the context and intent of the statement.
        Reminder:
            - If there is no instance in the corresponding statement, return "List Instances is:" "No Instance".

        Examples:
            Simplified Statement: Marco and Alice enroll to seminar final, they got accepted for tutorial
            List Predicates 1: enroll_seminar, accepted_tutorial
            List Instances 1: Marco, Alice
            Important Information 1: Marco and Alice both enroll to the seminar. Marco and Alice are accepted for tutorial. Need to enroll in seminar to have tutorial
            ----

            Simplified Statement 2: Choosing a major requires ≥ 30 credits, GPA ≥ 2.5 (scale 0-4), ≤ 2 violations; second-year status (≥ 24 credits).
            List Predicates 2: choose_major, require_credits, require_gpa, require_violation, require_status, second_year
            List Instances 2: 30, $2.5$, 2, 24
            Important Information 2: Need at least 30 credits, 2.5 GPA, 24 credits in second-year and at most 2 violations for choosing a major. Credits represent the amount of work a student has completed for a course. GPA is Grade Point Average. Violations refer to disciplinary or academic misconduct recorded by the institution.
            ----

            Simplified Statement 3: A student can withdraw up to 2 courses per semester if remaining credits ≥ 12; each withdrawal reduces GPA by 0.1 (scale 0-10).
            List Predicates 3: withdraw_courses, remaining_credit, require_gpa, reduces_GPA
            List Instances 3: 2, 12, $0.1%
            Important Information: Withdraw 2 courses reduce GPA by $0.2$. Withdrawal means formally removing oneself from a course after registration but before completion. Course is a unit of teaching that typically covers a specific subject or topic. A semester is a division of the academic year. Credits represent the amount of work a student has completed for a course. GPA is Grade Point Average
            ----

            Simplified Statement 4: If a student cannot teach, then they lack a research foundation.
            List Predicates 4: lack_research foundation, student_cannot_teach
            List Instances 4: No Instance
            Important Information: Research experience is foundational to effective teaching. A research foundation refers to the fundamental knowledge, skills, and experience in conducting academic or scientific research
            ----

        ### Answer Response: 
            Simplified Statement 1: <your simplified statement here>
            List Predicates 1: <comma-separated list of predicates>
            List Instances 1: <comma-separated list of instances or "No Instance">
            Important Information 1: <your extra important info>

            ----
            Simplified Statement 2: ...
            List Predicates 2: ...
            List Instances 2: ...
            Important Information 2: ...
            ... (and so on)
        
    '''


def PARAPHRASE_PROMPT_TEMPLATE_action_object():
    return f'''
        {HEAD_INSTRUCTION()}
        Please simplify the following logic problem statement and convert it into a "Simplified Statement". Each "Simplified Statement" derive 3 sections "List Objects", "List Actions", "List Instances".
            - Each Statement is seperated by dot formula ".". 
            - Only remove stopwords and unnecessary words in the statement, in order to keep all clause meaning in the statements with few changing.
            - Ensure that the simplified information maintains all logical relationships of the original statement. 
            - The number of "Simplified Statement" must be equal to the number of questions statement. "List Objects", "List Actions" and "List Instances" number i is infered from corresponding "Simplified Statement" number i.
            - "List Objects" contains the event of the statement, usually a noun. "List Actions" contains the action on the event of the statement, usually a verb, adjective, comparison symbols, etc. "List Instances" is the instance of the object, it is a specific name of a person, place, thing, idea or sometime a number (include float and integer number), etc.
        You can add "Important Infomation" by extract the core information from each statement items that you think important but is not appropriate to parallel with the Logical Statements, focus on one or more of the following:
            - The meaning of a specific phrase or term in the statement,
            - A mathematical solution or equation relevant to the statement,
            - A logical deduction or inference based on the scenario described.
            - This information should help clarify the context and intent of the statement.
        Reminder:
            - If there is no instance in the corresponding statement, return "List Instances is:" "No Instance".
            
        Examples:
            Simplified Statement: Marco and Alice enroll to seminar final, they got accepted for tutorial
            List Objects 1: seminar, tutor
            List Actions 1: enroll, accepted
            List Instances 1: Marco, Alice
            Important Information 1: Marco and Alice both enroll to the seminar. Marco and Alice are accepted for tutorial. Need to enroll in seminar to have tutorial
            ###

            Simplified Statement 2: Choosing a major requires ≥ 30 credits, GPA ≥ 2.5 (scale 0-4), ≤ 2 violations; second-year status (≥ 24 credits).
            List Objects 2: major, credits, GPA, violations, second-year status
            List Actions 2: choosing, ≥, ≤
            List Instances 2: 30, $2.5$, 2, 24
            Important Information 2: Need at least 30 credits, 2.5 GPA, 24 credits in second-year and at most 2 violations for choosing a major. Credits represent the amount of work a student has completed for a course. GPA is Grade Point Average. Violations refer to disciplinary or academic misconduct recorded by the institution.
            ###

            Simplified Statement 3: A student can withdraw up to 2 courses per semester if remaining credits ≥ 12; each withdrawal reduces GPA by 0.1 (scale 0-10).
            List Objects 3: courses, semester, credits, withdrawal, GPA
            List Actions 3: can withdraw, remaining, reduces, ≥, ≤
            List Instances 3: 2, 12, $0.1%
            Important Information: Withdraw 2 courses reduce GPA by $0.2$. Withdrawal means formally removing oneself from a course after registration but before completion. Course is a unit of teaching that typically covers a specific subject or topic. A semester is a division of the academic year. Credits represent the amount of work a student has completed for a course. GPA is Grade Point Average
            ###

            Simplified Statement 4: If a student cannot teach, then they lack a research foundation.
            List Objects 4: research foundation
            List Actions 4: cannot teach, lack
            List Instances 4: No Instance
            Important Information: Research experience is foundational to effective teaching. A research foundation refers to the fundamental knowledge, skills, and experience in conducting academic or scientific research
            ###

        ### Answer Response: 
            Simplified Statement 1: <your simplified statement here>
            List Objects 1: <comma-separated list of objects>
            List Actions 1: <comma-separated list of actions>
            List Instances 1: <comma-separated list of instances or "No Instance">
            Important Information 1: <your extra important info>

            ----
            Simplified Statement 2: ...
            List Objects 2: ...
            List Actions 2: ...
            List Instances 2: ...
            Important Information 2: ...
            ... (and so on)
        
    '''

def PARAPHRASE_EXAMPLE_PROMPT_TEMPLATE(): 
    [
    {
    "Origin_text": """
        There exists someone who has obtained both MOS certifications in Word and Excel. If x did not register for the seminar, then x is not allowed to submit the report. If x registered for the seminar, then x has completed all requirements to register for the seminar. If x did not submit the required report, then x is not allowed to submit the report for the seminar. If x is eligible to attend the seminar, then x has the required certifications.
    """,
    "Simple_text": """
        Logical Statement 1: Exists someone obtained both MOS certifications in Word and Excel.
        Logical Statement 2: If x did not register for the seminar, x is not allowed to submit the report.
        Logical Statement 3: If x registered for the seminar, x has completed all requirements to register for the seminar.
        Logical Statement 4: If x did not submit the required report, x is not allowed to submit the report for the seminar.
        Logical Statement 5: If x is eligible to attend the seminar, then x has the required certifications.
        Cross Statement: If x has not completed all requirements to register for the seminar, x is eligible to attend the seminar. 
    """,
    },
    {
    "Origin_text":"""

    """,
    "Simple_text": """
        Logical Statement 1:
        Logical Statement 2: 
        Logical Statement 3: 
        Logical Statement 4: 
        Cross Statement:
    """,
    },
    {
    "Origin_text":"""

    """,
    "Simple_text": """
        Logical Statement 1: 
        Logical Statement 2: 
        Logical Statement 3: 
        Logical Statement 4: 
        Cross Statement:
    """,
    },

]

# ---------------------------------------------------------------
def MATCHING_PROMPT_TEMPLATE_1():
    return """
        {instruction}
        You are given a statement and a list of predicates that describe parts of that statement. Your task is to match each predicate to the exact phrase in the statement that corresponds to it. Identify which part of the sentence represents each predicate.
        Note that:
            - Do not rename the predicate
            - The number of "Matching" must be equal to the number of "Statement".
            - Each predicate in "List Predicates" is seperated by comma formula ",".
            - Each predicate can be the transformation of one or many phrases in the statement. Make sure to understand the background context and identify correctly. Sometime, phrases are different due to different word form, such as: purchase, purchasing, etc, but still stand for one predicate.
            - Phrase must be contain in the statement, validating by these following python code: 
                `good_phrase = True if phrase in statement else False`
                `good_phrase = True if len(phrase.split(" ")) <= 4 else False`
        Examples:
            [INST]
            Statement: Marco and Alice enroll to seminar final, they got accepted for tutorial
            List Predicates: enroll_seminar, accepted_tutorial
            [/INST]
            Matching: enroll_seminar matches with "enroll to seminar final", accepted_tutorial matches with "accepted for tutorial"
            ---

            [INST]
            Statement: Choosing a major requires ≥ 30 credits, GPA ≥ 2.5 (scale 0-4), ≤ 2 violations; second-year status (≥ 24 credits).
            List Predicates: choose_major, require_credits, require_gpa, require_violation, require_status, second_year
            [/INST]
            Matching: choose_major matches with "choosing a major", require_credits matches with "requires", require_gpa matches with "GPA", require_violation matches with "violations", require_status matches with "status", second_year matches with "second-year"
            ---

            [INST]
            Statement: A student can withdraw up to 2 courses per semester if remaining credits ≥ 12; each withdrawal reduces GPA by 0.1 (scale 0-10).
            List Predicates: withdraw_courses, remaining_credit, reduces_GPA
            [/INST]
            Matching: withdraw_courses matches with "withdraw", withdraw_courses matches with "withdrawal", remaining_credit matches with "remaining credits", reduces_GPA matches with "reduces GPA by"
            ---

            [INST]
            Statement: If a student cannot teach, then they lack a research foundation.
            List Predicates: lack_research foundation, student_cannot_teach
            [/INST]
            Matching: lack_research_foundation matches with "lack a research foundation", student_cannot_teach matches with "a student cannot teach"

        [INST]
        Reminder:
            - The phrase usually from 2-3 words per phrase. Maximum is 4 words but no longer.
            - Do not include the examples in "Answer Response".
            - Always validating if the phrase is good, if not, keep regenerate and validate again.
            - Format your answer as a list of predicate-phrase pairs, seperate by a comma.

        ### Answer Response: 
            "Matching 1" is: <comma-separated list of matching predicate with phrase>
            "Matching 2" is: ...
            "Matching 3" is: ...
            ... (and so on) 
            
    """

def MATCHING_PROMPT_TEMPLATE():
    return f"""
        {HEAD_INSTRUCTION()}
        You are given a "Statement" and "List Predicates", "List Instances" that describe all elements of the "Statement". Your task is to match each predicate to the exact phrase in the statement that corresponds to it. Identify which part of the sentence represents each predicate.
        Note that:
            - Do not rename the predicate.
            - Each predicate in "List Predicates" is seperated by comma formula ",".
            - The phrase usually from 2-4 words per phrase, no longer.
            - Each predicate can be the transformation of one or many phrases in the statement. Make sure to understand the background context and identify correctly. Sometime, phrases are different due to different word form, such as: purchase, purchasing, etc, but still stand for one predicate.
            - Phrase must be contain in the statement, validating by this following python code: `good_phrase = True if phrase in statement else False`.
            - Each instance is listed in "List Instances" should not include in the phrase.
            - If the phrase includes a number or threshold, exclude the numeric detail.
            - Each "Matching" derived from 3 sections "Statement", "List Predicates" and "List Instances".
            - The number of "Matching" must be equal to the number of "Statement".
        Examples:
            Statement: Marco and Alice enroll to seminar final, they got accepted for tutorial
            List Predicates: enroll_seminar, accepted_tutorial
            List Instances: 
            Matching: enroll_seminar matches with "enroll to seminar final", accepted_tutorial matches with "accepted for tutorial"

            ---
            Statement: Choosing a major requires ≥ 30 credits, GPA ≥ 2.5 (scale 0-4), ≤ 2 violations; second-year status (≥ 24 credits).
            List Predicates: choose_major, require_credits, require_gpa, require_violation, require_status, second_year
            Matching: choose_major matches with "choosing a major", require_credits matches with "requires", require_gpa matches with "GPA", require_violation matches with "violations", require_status matches with "status", second_year matches with "second-year"
            
            ---
            Statement: A student can withdraw up to 2 courses per semester if remaining credits ≥ 12; each withdrawal reduces GPA by 0.1 (scale 0-10).
            List Predicates: withdraw_courses, remaining_credit, reduces_GPA
            Matching: withdraw_courses matches with "withdraw", withdraw_courses matches with "withdrawal", remaining_credit matches with "remaining credits", reduces_GPA matches with "reduces GPA by"
            
            ---
            Statement: If a student cannot teach, then they lack a research foundation.
            List Predicates: lack_research foundation, student_cannot_teach
            Matching: lack_research_foundation matches with "lack a research foundation", student_cannot_teach matches with "a student cannot teach"
            
        Reminder:
            - The phrase usually from 2-4 words per phrase, no longer.
            - Do not include the examples in "Answer Response".
            - Always validating if the phrase is good, if not, keep regenerate and validate again.
            - Each predicate in "List Predicates" is seperated by comma formula ",".
        
        ### Answer Response: 
            "Matching 1" is: <comma-separated list of matching predicate with phrase>
            "Matching 2" is: ...
            "Matching 3" is: ...
            ... (and so on) 
            
    """




# ---------------------------------------------------------------

CONTEXT_CLASSIFICATION_PROMPT = '''
    **Task:** Convert the following natural language paragraph into standard first-order logic expressions. Focus on expressing the most direct and easily understandable relationships using first-order logic. Leave sentences that are difficult to express concisely in first-order logic as they are.

    - Use standard logical symbols ∧, ∨, →, ¬, ∀, ∃ to represent logical relationships.
    - Define and use predicates such as \( P(x) \), \( Q(x, y) \), etc., to represent objects or relationships.
    - Appropriately use quantifiers ∀ and ∃ to express universal or existential statements, especially using ∀ with the statements having constraints such as all, there is no, etc., and using ∃ with the statements having constraints such as at least, only, not all, a few etc. 
    - Please make sure every word in the input is showed in your output, your task is only to add some simple first-order logic expressions.

    
    ----
'''


CONTEXT_OBJECT_CLASSIFICATION_PROMPT = '''
    **Task:** Convert the following natural language paragraph into standard first-order logic expressions. Focus on expressing the most direct and easily understandable relationships using first-order logic. Leave sentences that are difficult to express concisely in first-order logic as they are.

    - Use standard logical symbols ∧, ∨, →, ¬, ∀, ∃ to represent logical relationships.
    - Define and use predicates such as \( P(x) \), \( Q(x, y) \), etc., to represent objects or relationships.
    - Appropriately use quantifiers ∀ and ∃ to express universal or existential statements, especially using ∀ with the statements having constraints such as all, there is no, at least,  only, etc. 
    - Please make sure every word in the input is showed in your output, your task is only to add some simple first-order logic expressions.

    
    ----
'''


IE_FOL_TRANSLATION_PROMPT = '''

'''


# SYMBOLIC REASONING

FUSION_Q_O_PROMPT = '''

'''


SR_FOL_TRANSLATION_PROMPT = '''

'''
