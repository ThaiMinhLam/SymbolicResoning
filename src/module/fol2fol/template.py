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

def PARSE_QUESTION():
    return """
        You are a question decomposition assistant. When given a complex natural language question that includes multiple sub-questions joined by "and", your job is to split it into exactly the number of independent questions based on their unique intent or reasoning goal.
        **Step**:
            - Identify the seperated word: each subquestion in the question usually seperated by "and", "," or some sentence connected phrase.
            - Only split when the original sentence clearly contains **two or more independent information needs**. Do NOT paraphrase or repeat the same question in different ways.
            - If split, split by the identified seperated word
            - Recheck the output questions by combining them together and comparing with the original question. If missing information, adding that information to an approriate parse question.
            - Format your output as a numbered list.

        ### Example 1:
        Input Question:
        "How many students failed the final exam, and who got the highest score?"

        Split Questions:
        The seperated word is: ", and"
        1. How many students failed the final exam?
        2. Who got the highest score?

        ### Example 2:
        Input Question:
        "What is the average GPA in Year 3, and did John improve his GPA compared to Year 2?"

        Split Questions:
        The seperated word is: ", and"
        1. What is the average GPA in Year 3?
        2. Did John improve his GPA compared to Year 2?

        ### Now, split the following question:

    """

# ------------------REASONING----------------------------

def UNDERSTAND_BACKGROUND_PROMPT():
    return '''
        [INSTRUCTION]

        You are an expert AI trained to reason over symbolic logic programs used for educational and policy reasoning tasks. Your goal is to understand a hybrid knowledge base consisting of:

        1. **Predicates**: Each describes a the event of the premise, **Predicates** having formula such as \( P(x) \), \( Q(x, y) \), etc., to represent objects or relationships. Each **Predicate** always having list of arguments inside the parentheses. Each argument corresponds to an entity (name, year, month, age, score, number, place or proper name, etc.) which the predicate is applied to.
        2. **FOL**: Rules and facts expressed in First-Order Logic using quantifiers (∀, ∃), also use standard logical symbols ∧, ∨, →, ¬, ∀, ∃ to represent logical relationships between **Predicates**.
        3. **Natural Language Descriptions**: Informal statements equivalent to the logic.
        4. **Premises**: A statement or idea which serves as the basis for an argument. The **Premise** is a powerful concept and an important element in logical argument. 
        5. **Thoughts**: The logic strategy step-by-step how all **Predicate** are connected in **Premise**/**FOL**
        Each FOL formula is a rule or fact. The knowledge base represents constraints, definitions, and specific cases.

        ---

        ### Format

        **Logic Program Predicates**
            - Each line follows: `PredicateName(args) means: Description`
            - **Important Note**:
                + This describes what each predicate means, how each argument affect each others, and also the context where each **Predicate** is put in.
                + Description in describe the rules for its **Predicate**. When logically reasoning in the step **Thoughts**, make sure all the rules are strictly followed.

        **Logic Program Premises** 
            - Each line follows: `FOL means: Description`
            - **Important Note**:
                + Each line is a logic formula FOL representing either a domain constraint or a fact. They may be general rules or instance-specific information. These FOL is the logic transfomation from corresponding **Premise**
                + Description in describe the relationships beetween many **Predicates**. When logically reasoning in the step **Thoughts**, make sure all the relationships are ensured without lossing information. Also understand the context across multiple **FOL** lines to make sure you understand whole story background and logic that **FOL** want you to know.

        ---

        ### Your Goals
        You should:
        - Understand the meaning of each predicate and how it relates to real-world educational policy
        - Give the accurate answer and reason for the complex questions based on the premises
        - Identify whether a query is entailed by the premises
        - Perform step-by-step symbolic reasoning (forward chaining or backward chaining)
        - Some questions need basic mathematical background (addition, subtraction, multiplication, division). Understand the context of the premises base on above information is all-you-need.
        ---


        ### You Can Now:
        Answer the question based only the **Logic Program Predicates** and **Logic Program Premises**.
        Read and analyze the **Premises** from **Logic Program Premises** to understand the overall context.
            - Give **Thoughts** clearly step-by-step with approriate premise's id.
            - Provide an accurate answer base on the question you received:
                - If the question is binary-type question: The "Answer" is Yes, No or Uncertain.
                - If the question is multiple-choice question: The "Answer" is A, B, C or D.
                - If the question requiring a specific value, (e.g., What is the minimum GPA required?): The "Answer" is a number (either integer or float).
                - If the question requiring multiples values, (e.g., How many students has more than 8 scores?): The "Answer" is unlimited list of number (either integer or float).
            - Provide reasoning steps to justify your answers
        Please make sure your reasoning is directly deduced from the "Premises" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.
        ---

        [RESPONSE]

        ### Input:
        You are now provided with a logic program and question. Read all logic program and explanation before reasoning.
        **Logic Program Predicates**: "{lp_predicates}" 
        **Logic Program Premises**: "{lp_premises}" 

        ### Answer Response:
        **Thoughts**: "Let us think step by step: "
        **Answer**: "Now we know that the correct answer is: "
    '''


def UNDERSTAND_BACKGROUND_PROMPT_WITHOUT_PREMISE():
    return '''
        [INSTRUCTION]

        You are an expert AI trained to reason over symbolic logic programs used for educational and policy reasoning tasks. Your goal is to understand a hybrid knowledge base consisting of:

        1. **Predicates**: Each describes a the event of the premise, **Predicates** having formula such as \( P(x) \), \( Q(x, y) \), etc., to represent objects or relationships. Each **Predicate** always having list of arguments inside the parentheses. Each argument corresponds to an entity (name, year, month, age, score, number, place or proper name, etc.) which the predicate is applied to.
        2. **FOL**: Rules and facts expressed in First-Order Logic using quantifiers (∀, ∃), also use standard logical symbols ∧, ∨, →, ¬, ∀, ∃ to represent logical relationships between **Predicates**.
        3. **Natural Language Descriptions**: Informal statements equivalent to the logic.
        5. **Thoughts**: The logic strategy step-by-step how all **Predicate** are connected in **Premise**/**FOL**
        Each FOL formula is a rule or fact. The knowledge base represents constraints, definitions, and specific cases.

        ---

        ### Format

        **Logic Program Predicates**
            - Each line follows: `PredicateName(args) means: Description`
            - **Important Note**:
                + This describes what each predicate means, how each argument affect each others, and also the context where each **Predicate** is put in.
                + Description in describe the rules for its **Predicate**. When logically reasoning in the step **Thoughts**, make sure all the rules are strictly followed.

        **Context FOL** 
            - Each line follows: `FOL`.
            - Understand the **FOL** by following step:
                + Identify the predicates used in the FOL formula by referring to the definitions provided in the **Logic Program Predicates**.
                + Determine the relationships between the predicates — how they influence and interact with one another, and under what conditions they are constrained or activated.
                + When reasoning, select and understand only some **FOL** that the most relative to the question to explain.
            
            - **Important Note**:
                + Each line is a logic formula FOL representing either a domain constraint or a fact. They may be general rules or instance-specific information. These FOL is the logic transfomation from corresponding natural languages premises - contain overall context and this is the main context that you must based on when reasoning the answer. 
                + **FOL** describe the relationships beetween many **Predicates**. When logically reasoning in the step **Thoughts**, make sure all the relationships are ensured without lossing information. Also understand the context across multiple **FOL** lines to make sure you understand whole story background and logic that **FOL** want you to know.

        ---

        ### Your Goals
        You should:
            - Understand the meaning of each predicate and how it relates to real-world educational policy
            - Identify whether a query is entailed by the **FOL**
            - Perform step-by-step symbolic reasoning (forward chaining or backward chaining)
            - Some questions need basic mathematical background (addition, subtraction, multiplication, division). Understand the context of the premises base on above information is all-you-need.
            - Give the accurate answer and reason for the complex questions based on the **Predicate** and **FOL**.
        ---


        ### You Can Now:
        Answer the question based only the **Logic Program Predicates** and **Logic Program Premises**.
        Read and analyze the **FOL** from **Context FOL** to understand the overall context.
            - Give **Thoughts** clearly step-by-step with approriate premise's id.
            - Provide an accurate answer base on the question you received:
                - If the question is binary-type question: The "Answer" is Yes, No or Uncertain.
                - If the question is multiple-choice question: The "Answer" is A, B, C or D.
                - If the question requiring a specific value, (e.g., What is the minimum GPA required?): The "Answer" is a number (either integer or float).
                - If the question requiring multiples values, (e.g., How many students has more than 8 scores?): The "Answer" is unlimited list of number (either integer or float).
            - Provide reasoning steps to justify your answers
        Please make sure your reasoning is directly deduced from the "Context FOL" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.
        ---

        [RESPONSE]

        ### Input:
        You are now provided with a logic program and question. Read all logic programs and explanation before reasoning.
        **Logic Program Predicates**: "{lp_predicates}" 
        **Logic Program Premises**: "{lp_premises}" 

        ### Answer Response:
        **Thoughts**: "Let us think step by step: "
        **Answer**: "Now we know that the correct answer is: "
    '''


#----------------------------MAPPING---------------------
def REDUCE_AND_MATCHING_PREDICATE_PROMPT_v1():
    return '''
        You are given a list of predicate definitions used in a logic programming context. Each predicate is annotated with a description of what it means. Your task is to:

        1. Identify any redundant predicates that can be expressed using other more general (overall) predicates.
        2. Replace those redundant predicates with expressions that use the more general predicates, while preserving their original meaning.
        3. Output a cleaned version of the predicate list, using only the most general and necessary predicates.

        Only remove a predicate if it is fully expressible using other predicates and does not carry unique semantic value.

        --- INPUT ---

        Predicate List - List of predicates:

        --- OUTPUT ---

        1. Redundant predicates identified and replaced (with explanation):
            - Example: PredicateA(x) is redundant because it is equivalent to PredicateB(x, 3); replace PredicateA(x) ⇒ PredicateB(x, 3)

        2. Cleaned Predicate List (with redundancies removed and only general forms kept):
            - PredicateX(x, y) ::: description...
            - PredicateY(x) ::: description...
            ...

        Ensure clarity, conciseness, and semantic preservation in your transformation.
    '''

# Note: All final distinct **Predicates** do not duplicate names or arities. If many **Predicates** having name and definition similar to each others more than 90%, combine to only one **Predicate**. Each **Predicate** must be uniquely named with a distinct number of arguments.
#         4. Keep the main predicates that has the importance role such as: 
# - Example: PredicateA(x) is redundant because it is equivalent to PredicateB(x, 3); replace PredicateA(x) ⇒ PredicateB(x, 3)
def REDUCE_AND_MATCHING_PREDICATE_PROMPT_chuẩn_official():
    return '''
        [INSTRUCTION]
        You are given a list of predicate and its definitions (seperated by :::) which used in a logic programming context. Your task is to:
        1. Identify any redundant predicates that can be expressed using other more general (overall) predicate.
        2. Replace those redundant predicates with expressions that use the more general predicates, while preserving their original meaning.
            - Example: PredicateA(args) is redundant because it is equivalent to PredicateB(args); replace PredicateA(args) ⇒ PredicateB(args)
        Only remove a predicate if it is fully expressible using other predicate and does not carry unique semantic value.
        1. Redundant predicates identified and replaced.
        2. General predicate derived from list predicate.
        3. Always check if the replaced predicate definition can be express by the new predicate ones.
        4. Keep the main predicates that has important role. These main predicate has role such as: {main_predicates} (1 predicate per role is kept).
        5. Do not give explanation
        [RESPONSE]

        

        ### Output
        "List Redundant Predicates": Now we know that the replacement are: 
            <predicate> is redundant and can be replaced by <predicate>
            ...
    '''

def REDUCE_AND_MATCHING_PREDICATE_PROMPT():
    return '''
        [INSTRUCTION]
        You are given a list of predicate and its definitions (seperated by :::) which used in a logic programming context. Your task is to:
        1. Identify any redundant predicates that can be expressed using another more general (overall) predicate.
        2. Replace those redundant predicates with expressions that use one or two general predicates, while preserving their original meaning.
            - Example: PredicateA(args) is redundant because it is equivalent to PredicateB(args); replace PredicateA(args) ⇒ PredicateB(args)
            - Or Example: PredicateA(args) is redundant because it is equivalent the combination of PredicateB(args) and PredicateC(args) using logical conjunction ∧; replace PredicateA(args) ⇒ PredicateB(args) ∧ PredicateB(args)
        
        Only remove a predicate if it is fully expressible using other predicate and does not carry unique semantic value.
        1. Redundant predicates identified and replaced.
        2. General predicate derived from list predicate.
        3. Always check if the replaced predicate definition can be express by the new predicate ones.
        4. Keep the main predicates that has important role. These main predicate has role such as: {main_predicates} (1 predicate per role is kept).
        5. Do not give explanation
        6. If predicate can be replace by two another predicates, some logical conjunction can be used is: ∧ (and), ∨ (or).
        [RESPONSE]

        

        ### Output
        "List Redundant Predicates": Now we know that the replacement are: 
            <predicate> is redundant and can be replaced by <predicate>
            ...
    '''


def REDUCE_AND_MATCHING_PREDICATE_PROMPT_ổn_chỉnh_lại_như_bìu():
    return '''
        [INSTRUCTION]
        You are given a list of predicate and its definitions which used in a logic programming context. Your task is to:
        1. Identify any redundant predicates that can be expressed using other more general (overall) predicate.
        2. Replace those redundant predicates with expressions that use the more general predicates, while preserving their original meaning.
        3. Output a cleaned version of the predicate list, using only the most general and necessary predicates.

        Only remove a predicate if it is fully expressible using other predicate and does not carry unique semantic value.
        1. Redundant predicates identified and replaced (with explanation):
            - Example: PredicateA(x) is redundant because it is equivalent to PredicateB(x, 3); replace PredicateA(x) ⇒ PredicateB(x, 3)
        2. Only replaced redundant predicate, not all of the predicates.
        3. General predicate can not be replaced.
        4. Always check if the replaced predicate definition can be express by the new predicate ones.
        5. Keep the main predicates that has the importance role. These main predicate has role such as: "accumulated credits", "attempt", "cannot withdraw", "contribute gpa", "have credits", "max withdrawals", "no regulation", "penalty", "register credits", "remaining credits", "withdraw courses", "withdraw credits".
        [RESPONSE]

        ### Input
        **List Predicate and its Definition**: List of <predicate ::: definition>

        ### Output
        For each replacement, return by using these following format: `Predicate: <OldPredicate>(...) → Predicate <NewPredicate>(...)`
        **List Redundant Predicates**:
            <replacement1>
            <replacement2>
            ...
    '''


def REDUCE_AND_MATCHING_PREDICATE_PROMPT_v2():
    return '''
        [INSTRUCTION]
        You are given a list of predicate and its definitions which used in a logic programming context. Each predicate is annotated with a description of what it means. Your task is to:
        1. Identify any redundant predicates that can be expressed using other more general (overall) predicates.
        2. Replace those redundant predicates with expressions that use the more general predicates, while preserving their original meaning.
        3. Output a distinct cleaned version of the predicate list, using only the most general and necessary predicates.

        Only remove a predicate if it is fully expressible using other predicates and does not carry unique semantic value.
        1. Redundant predicates identified and replaced (with explanation):
            - Example: PredicateA(x) is redundant because it is equivalent to PredicateB(x, 3); replace PredicateA(x) ⇒ PredicateB(x, 3)

        2. Cleaned Predicate List (with redundancies removed and only general forms kept):
            - PredicateX(x, y) ::: description...
            - PredicateY(x) ::: description...
            ...

        Note:
            - All final distinct predicates do not duplicate names or arities. **Each predicate must be uniquely named with a distinct number of arguments.**

        [RESPONSE]

        ### Input
        **List Predicate and its Definition**: <List of **Predicate** ::: **Definition**>

        ### Output
        **List Redundant Predicates**: 
            <Predicate> is replaced by <Predicate>
            ...

        **List General Predicates**:
            <Predicate>
            ...
    '''


def REDUCE_AND_MATCHING_PREDICATE_PROMPT_CONVERT_FOL():
    return '''
        [INSTRUCTION]
        You are provided with:
            - A list of predicate definitions used in a logic programming context, each annotated with a description of its meaning.
            - A set of First-Order Logic (FOL) formulas constructed using these predicates.
        1. Identify any redundant predicates that can be expressed using other more general (overall) predicates.
        2. Replace those redundant predicates with expressions that use the more general predicates, while preserving their original meaning.
        3. Output a cleaned version of the predicate list, using only the most general and necessary predicates.

        Only remove a predicate if it is fully expressible using other predicates and does not carry unique semantic value.
        1. Redundant predicates identified and replaced (with explanation):
            - Example: PredicateA(x) is redundant because it is equivalent to PredicateB(x, 3); replace PredicateA(x) ⇒ PredicateB(x, 3)

        2. Cleaned Predicate List (with redundancies removed and only general forms kept):
            - PredicateX(x, y) ::: description...
            - PredicateY(x) ::: description...
            ...

        3. Reconstruct the FOL Formulas: Rewrite the original FOL statements using the cleaned set of predicates, ensuring that all logical semantics remain intact.


        
        [RESPONSE]

        ### Input
        Predicate List: <List of **Predicate**>

        ### Output
        **List Redundant Predicates**: 
            <Predicate> is replaced by <Predicate>
            ...

        **List General Predicates**:
            <Predicate>
            ...
    '''

#---------------------------------EXTRACT MAIN PREDICATE--------------
def EXTRACT_MAIN_PREDICATE():
    return '''
        {instruction}
        Please simplify the following logic problem statement and convert it into a "Simplified Statement". Each "Simplified Statement" derive "List Predicates" section.
            - Only remove stopwords and unnecessary words in the statement, in order to keep all clause meaning in the statements with few changing.
            - Ensure that the simplified information maintains all logical relationships of the original statement. 
            - "List Predicates" number i is infered from corresponding "Simplified Statement" number i.
            - "List Predicates" contains the event of the statement, this is the combination of 1 action and 1 object. 
                + *object* contains the event of the statement, usually a noun. 
                + *actions* contains the action on the event of the statement, usually a verb, adjective, comparison symbols, etc.
                + Each predicate is formatted by this following python code: predicate = action + "_" + object.
                + Not include negative word like not, do_not, etc., in the predicate.
        Examples:
            Simplified Statement: Marco and Alice enroll to seminar final, they got accepted for tutorial
            List Predicates 1: enroll_seminar, accepted_tutorial
            ----

            Simplified Statement 2: Choosing a major requires ≥ 30 credits, GPA ≥ 2.5 (scale 0-4), ≤ 2 violations; second-year status (≥ 24 credits).
            List Predicates 2: choose_major, require_credits, require_gpa, require_violation, require_status, second_year
            ----

            Simplified Statement 3: A student can withdraw up to 2 courses per semester if remaining credits ≥ 12; each withdrawal reduces GPA by 0.1 (scale 0-10).
            List Predicates 3: withdraw_courses, remaining_credit, require_gpa, reduces_GPA
            ----

            Simplified Statement 4: If a student cannot teach, then they lack a research foundation.
            List Predicates 4: lack_research foundation, student_cannot_teach
            ----

        ### Answer Response: 
            Simplified Statement 1: <your simplified statement here>
            List Predicates 1: <comma-separated list of predicates>

            ----
            Simplified Statement 2: ...
            List Predicates 2: ...
            ... (and so on)
    '''


#---------------------------------LOGIC PROGRAM------------------------
def LOGIC_PROGRAM_EXTRACTION_PROMPTING():
    return '''
        [INSTRUCTION]
        Define the meaning of each FOL predicate individually by directly extracting from the corresponding natural language (NL) statement. You are given:
        1. You are given the list include pairs of <Statement, Predicate> and additional <Question-Predicates>:
            - **Statements**: A list of Natural Language (NL) statements, where each statement describes a context, domain, or situation of predicates.
            - **Predicates**: A corresponding list of predicates extracted from corresponding Natural Language (NL) statement, contain name of predicate and its input arguments. 
        2. List of **Predicates** number i is infered from corresponding **NL Statement** number i.
        3. Each predicate from list is seperated by a comma ", ".

        Please follow these instructions carefully:
        1. Read and understand all logical relationships of predicates in the corresponding statement.
        2. Understand the meaning of each predicate based on the context of the corresponding statement. This information should help clarify the context and intent of the statement.
        3. Using a logical deduction or inference based on the scenario described if needed.
        4. Extracting the definition and each predicate from list of **Predicates**, define which each argument stand for and what is that predicate meaning according to the definition of its arguments.

        **Important Note**:
            - Looping through each predicate in each **Predicates** by this following Python code: `for predicate in predicates.split(", ")`.
            - Only give the name of each predicate and its definition in the output.
            - Definition should include 2 parts: 
                + What is the meaning of each argument (arg_x is ...)
                + What action that predicate do on each argument (something do arg_x, arg_y)
            - Extracting the definition of each predicate, base on the context of the statement and meaning of its predicate, for example (these examples will not be included in the output):
                + Teacher(x) is: x is a Teacher
                + DoTestSubject(x, y) is: student x do test on y subject
                + ContributesCredits(x, y) is: course x contributes y credits to GPA
            - Using the argument variable instead of its meaning to express the definition:
            - Do not include examples and explanation in the output.
#           - Only one line output per predicate
            - Following the input format and output format below:
        
        ### Input:
            Predicates 1: <comma-separated list of predicates>
            Statement 1: <your statement here>
            .... (and so on)


        ### Output
           Predicate 1: <each predicate in list of predicates>
           Definition 1: <your definition here>
            .... (and so on)
    '''
    
# Predicates 1: <comma-separated list of predicates>
#             Definition 1: <your definition here>

def LOGIC_PROGRAM_EXTRACTION_PROMPTING_NEW():
    return '''
    ### Task: Define the meaning of each FOL predicate individually by extracting and reformulating from the corresponding natural language (NL) statement into a short, complete sentence.

    You are given:
    - A list of Natural Language (NL) statements, where each statement describes a context, domain, or situation.
    - A corresponding list of lists of First-Order Logic (FOL) predicates extracted from those contexts.
    - Each NL statement at position i corresponds exactly to the list of FOL predicates at position i.

    Please follow these instructions carefully:
    1. Interpret the NL statement: 
    - Understand the general context and concepts described.
    
    2. Define each FOL predicate:
    - For each predicate in the list at position i:
        - Carefully read the corresponding NL statement.
        - Identify the smallest possible phrase (or fragment) directly from the NL statement that fully captures the intended meaning of the predicate:
            - The fragment must be short, precise, and taken from the NL wording.
            - It must accurately and completely reflect the meaning required by the predicate.
        - Check the selected phrase by answering::
            (a) Does it fully cover the intended meaning of the predicate?
            (b) Is it directly quoted or minimally adapted from the NL statement without adding or omitting information?
            (c) Is the definition short, clear, and faithful to the wording and semantics of the NL statement?
        - Only after verifying all conditions (a), (b), and (c) are satisfied, output the final definition.
        
    3. Use the required output format rules:
    - For each predicate to be defined, output must strictly follow this exact structure for each line: [FOL-Predicate] ::: [Natural Language Definition]
        - Use exactly three colons and spaces (" ::: ") as the separator between the FOL predicate and its definition.
        - Do not add numbering, bullet points, extra text, or explanations beyond the required format.

    **Additional guidelines**:
    - Keep each definition concise, accurate, and faithful to the given NL statement.
    - Only describe what each predicate represents.
    - Each predicate must have exactly one output line; do not combine multiple predicates into one line.
    '''