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

def UNDERSTAND_BACKGROUND_PROMPT_EN():
    return '''
[INSTRUCTION]
You are an AI expert in hybrid symbolic logic and natural language reasoning.
Your task is to reason over a knowledge base consisting of:

1. **Logic Program Predicates**: use only to understand symbols in the premises.
2. **First-Order Logic Premises**: the authoritative source for inference.
3. A **Question** and a **User Answer** provided by the user.

You must:
- Classify the **question type** (Yes/No, Multiple-Choice, Specific-Value).
- Select the relevant **premises**.
- Provide a **step-by-step rationale**, referencing premise identifiers.
- Present the **final answer** and explain why the given answer is correct.

### Input:
- **Logic Program Predicates**:
{lp_predicates}

- **First-Order Logic Premises**:
{lp_premises}

- **Question**:
{question}

- **User Answer**:
{answer}

### Output format:
1. **Question Type**: ...
2. **Selected Premises**: e.g., P1, P3
3. **Rationale**:
   - Step 1: ... (referencing P1)
   - Step 2: ... (referencing P3)
4. **Final Answer**: ...
5. **Explanation**: Summary of why `{answer}` is correct.

[END OF INSTRUCTION]
'''


def UNDERSTAND_BACKGROUND_PROMPT_v2():
    return '''
[INSTRUCTION]

You are an expert AI specializing in symbolic logic reasoning for educational and policy domains. Your task is to understand and reason over a hybrid knowledge base composed of:

1. **Predicates**: Symbolic expressions (e.g., \( P(x) \), \( Q(x, y) \)) that describe objects, events, or relationships. Each predicate has a list of arguments, each representing an entity (e.g., name, year, month, age, score, place, number).
2. **First-Order Logic (FOL)**: Rules and facts expressed using logical quantifiers (∀, ∃) and standard operators (∧, ∨, →, ¬). FOL formulas formalize relationships among predicates.
3. **Natural Language Descriptions**: Informal paraphrases of the logical meaning.
4. **Premises**: Statements serving as the basis for logical arguments. Each premise corresponds to a formal FOL rule or fact.
5. **Thoughts**: Step-by-step logical strategies that show how predicates connect to each other within the premises and FOL rules.

Each FOL formula represents a constraint, definition, or specific case within the domain.

---

### Knowledge Base Format

**Logic Program Predicates**:
- Each line follows: `PredicateName(arguments) means: Description`
- **Important**:
    + The description defines what the predicate expresses, how its arguments interact, and the context of its application.
    + Logical reasoning in the **Thoughts** must strictly adhere to these definitions.

**Logic Program Premises**:
- Each line follows: `FOL means: Description`
- **Important**:
    + Each line presents a formal FOL rule transformed from its corresponding premise.
    + The description explains how multiple predicates relate within the rule.
    + Reasoning must preserve all stated relationships without omitting information.
    + Analyze across multiple premises if necessary to fully understand the background context.

---

### Reasoning Goals

You should:
- Understand the meaning of each predicate and its connection to real-world educational policy.
- Accurately answer complex questions based on the premises.
- Determine if a given query is entailed by the premises.
- Perform symbolic reasoning (forward chaining or backward chaining).
- Perform basic calculations (addition, subtraction, multiplication, division) if needed.
- Base your reasoning only on the provided predicates and premises without introducing external commonsense knowledge.

---

### Answering Instructions

When answering:
- Provide **Thoughts**: clear step-by-step reasoning, referencing relevant premise IDs if needed.
- Provide **Answer** according to the question type:
    - Binary questions: Yes, No, or Uncertain
    - Multiple-choice questions: A, B, C, or D
    - Specific value questions: An integer or float
    - List-type questions: A list of numbers (integer or float)

---

[RESPONSE]

### Input:
You are provided with a logic program and a question. Analyze all information carefully before answering.

- **Logic Program Predicates**: 
{lp_predicates}

- **Logic Program Premises**: 
{lp_premises}

### Answer Response:
**Thoughts**: "Let us think step by step: "
**Answer**: "Thus, the correct answer is: "
    '''


def UNDERSTAND_BACKGROUND_PROMPT_v3():
    return '''
[INSTRUCTION]

You are an expert AI specializing in symbolic logic reasoning for educational and policy domains. Your task is to understand and reason over a hybrid knowledge base composed of:

1. **Predicates**: Symbolic expressions (e.g., \( P(x) \), \( Q(x, y) \)) that describe objects, events, or relationships. Each predicate has a list of arguments, each representing an entity (e.g., name, year, month, age, score, place, number).
2. **First-Order Logic (FOL)**: Rules and facts expressed using logical quantifiers (∀, ∃) and standard operators (∧, ∨, →, ¬). FOL formulas formalize relationships among predicates.
3. **Natural Language Descriptions**: Informal paraphrases of the logical meaning.
4. **Premises**: Statements serving as the basis for logical arguments. Each premise corresponds to a formal FOL rule or fact.
5. **Thoughts**: Step-by-step logical strategies that show how predicates connect to each other within the premises and FOL rules.

Each FOL formula represents a constraint, definition, or specific case within the domain.

---

### Knowledge Base Format

**Logic Program Predicates**:
- Each line follows: `PredicateName(arguments) means: Description`
- **Important**:
    + The description defines what the predicate expresses, how its arguments interact, and the context of its application.
    + Logical reasoning in the **Thoughts** must strictly adhere to these definitions.

**Logic Program Premises**:
- Each line follows: `FOL means: Description`
- **Important**:
    + Each line presents a formal FOL rule transformed from its corresponding premise.
    + The description explains how multiple predicates relate within the rule.
    + Reasoning must preserve all stated relationships without omitting information.
    + Analyze across multiple premises if necessary to fully understand the background context.
    + You do not need to use all the premises. Choose only the relevant premises needed to answer the question.

---

### Reasoning Goals

You should:
- Understand the meaning of each predicate and its connection to real-world educational policy.
- Accurately answer complex questions based on the premises. You may not need to use all the premises — only use the ones most relevant for answering the question.
- Select the appropriate premises and use them to answer the question.
- If the question asks for a Yes/No answer, provide the most accurate Yes or No based on the premises.
- If the question is a multiple-choice question, choose the correct option (A, B, C, or D) based on the premises.
- You may perform symbolic reasoning (forward chaining or backward chaining) based on the selected premises.
- Perform basic calculations (addition, subtraction, multiplication, division) if needed.
- Base your reasoning only on the provided predicates and premises without introducing external commonsense knowledge.

---

### Answering Instructions

When answering:
- Provide **Thoughts**: clear step-by-step reasoning, referencing relevant premise IDs if needed.
- Provide **Answer** according to the question type:
    - Binary questions: Yes, No, or Uncertain
    - Multiple-choice questions: A, B, C, or D
    - Specific value questions: An integer or float
    - List-type questions: A list of numbers (integer or float)

---

[RESPONSE]

### Input:
You are provided with a logic program and a question. Analyze all information carefully before answering.

- **Logic Program Predicates**: 
{lp_predicates}

- **Logic Program Premises**: 
{lp_premises}

### Answer Response:
**Thoughts**: "Let us think step by step: I will identify the most relevant premises and use them to answer the question."
**Answer**: "Thus, the correct answer is: "
)

'''

def UNDERSTAND_BACKGROUND_PROMPT_v4():
    return '''
[INSTRUCTION]

You are an expert AI specializing in symbolic logic reasoning for educational and policy domains. Your task is to understand and reason over a hybrid knowledge base composed of:

1. **Predicates**: Symbolic expressions (e.g., P(x), Q(x, y)) that describe objects, events, or relationships. Each predicate has a list of arguments, each representing an entity (e.g., name, year, month, age, score, place, number).
   - **Note**: These definitions are provided purely to explain the semantics of symbols used in the premises. **Do not treat these as standalone facts**; use them only to interpret the premises.
2. **First-Order Logic (FOL) Premises**: Rules and facts expressed using logical quantifiers (∀, ∃) and standard operators (∧, ∨, →, ¬). These premises are the primary basis for inference.
3. **Natural Language Descriptions**: Informal paraphrases of the logical meaning of predicates and premises.
4. **Thoughts**: Step-by-step logical strategies that show how premises combine to yield the answer, referencing predicate semantics as needed.

Each FOL formula represents a constraint, definition, or specific case within the domain.

---

### Knowledge Base Format

- **Logic Program Predicates** (`{lp_predicates}`):
  - Each line: `PredicateName(arguments) means: Description`
  - **Important**: Use these only to interpret symbols in the premises. Do not draw independent conclusions from predicates alone.

- **Logic Program Premises** (`{lp_premises}`):
  - Each line: `FOL means: Description`
  - **Important**: These are the authoritative rules/facts. Your inference must be based on these. You may reference predicate semantics to clarify terms, but the logical reasoning must derive from these premises.

---

### Reasoning Goals

You should:
- Primarily use the **Logic Program Premises** to derive conclusions.
- Use **Logic Program Predicates** only to understand the meaning of symbols within those premises.
- Identify exactly which premises are used in your reasoning.
- Provide clear, step-by-step **Thoughts**, referencing premise IDs (e.g., P1, P3).
- Provide the final **Answer** to the question.

---

### Answering Instructions

1. Identify the question type (Yes/No, multiple-choice, or specific value).
2. Determine which premises from `{lp_premises}` are relevant and will be used. List their IDs under **Selected Premises**.
3. Provide **Thoughts**: detailed, step-by-step reasoning that shows how each selected premise leads toward the answer. Reference predicate semantics only when needed to interpret a premise.
4. Provide **Answer** in the appropriate format.

---

[RESPONSE]

### Input:
- **Logic Program Predicates**:
{lp_predicates}

- **Logic Program Premises**:
{lp_premises}

- **Question**:
{user_question}

### Response:
{{%- if q_type == "yesno" -%}}
Selected Premises:  # e.g., P2, P5  
Thoughts:          # step-by-step reasoning  
Answer:            # Yes or No
{{%- elif q_type == "mcq" -%}}
Selected Premises:  # e.g., P2, P5  
Thoughts:          # step-by-step reasoning  
Answer: A, B, C, or D
{{%- else -%}}
Selected Premises:  # list if applicable or leave blank
Thoughts:          # step-by-step reasoning  
Answer:            # integer, float, list, or text
{{%- endif -%}}
'''

def UNDERSTAND_BACKGROUND_PROMPT_v5():
    return '''
[INSTRUCTION]

You are an expert AI specializing in symbolic logic reasoning for educational and policy domains. Your task is to understand and reason over a hybrid knowledge base composed of:

1. **Predicates**: Symbolic expressions (e.g., P(x), Q(x, y)) that describe objects, events, or relationships. Each predicate has a list of arguments, each representing an entity (e.g., name, year, month, age, score, place, number).
   - **Note**: These definitions are provided purely to explain the semantics of symbols used in the premises. **Do not treat these as standalone facts**; use them only to interpret the premises.
2. **First-Order Logic (FOL) Premises**: Rules and facts expressed using logical quantifiers (∀, ∃) and standard operators (∧, ∨, →, ¬). These premises are the primary basis for inference.
3. **Natural Language Descriptions**: Informal paraphrases of the logical meaning of predicates and premises.
4. **Thoughts**: Step-by-step logical strategies that show how premises combine to yield the answer, referencing predicate semantics as needed.

Each FOL formula represents a constraint, definition, or specific case within the domain.

---

### Knowledge Base Format

- **Logic Program Predicates** (`{lp_predicates}`):
  - Each line: `PredicateName(arguments) means: Description`
  - **Important**: Use these only to interpret symbols in the premises. Do not draw independent conclusions from predicates alone.

- **Logic Program Premises** (`{lp_premises}`):
  - Each line: `FOL means: Description`
  - **Important**: These are the authoritative rules/facts. Your inference must be based on these. You may reference predicate semantics to clarify terms, but the logical reasoning must derive from these premises.

---

### Reasoning Goals

You should:
- Primarily use the **Logic Program Premises** to derive conclusions.
- Use **Logic Program Predicates** only to understand the meaning of symbols within those premises.
- Identify exactly which premises are used in your reasoning.
- Provide clear, step-by-step **Thoughts**, referencing premise IDs (e.g., P1, P3).
- Provide the final **Answer** to the question.

---

### Answering Instructions

Your first step is to **classify the question type** based solely on the natural language of `{user_question}` into one of:
- **Yes/No question**
- **Multiple-choice question**
- **Specific-value question** (numeric or list)

Then follow these steps:

1. **Question Classification**: State the identified question type.
2. **Select Premises**: Determine which premises from `{lp_premises}` are relevant. List their IDs under **Selected Premises**.
3. **Thoughts**: Provide detailed, step-by-step reasoning that uses the selected premises. Reference predicate semantics only when needed to interpret a premise.
4. **Answer**: Provide the final answer in the format appropriate for the classified question type.

---

[RESPONSE]

### Input:
- **Logic Program Predicates**:
{lp_predicates}

- **Logic Program Premises**:
{lp_premises}

- **Question**:
{user_question}

### Response:
**Question Type**:  # e.g., Yes/No, Multiple-Choice, Specific-Value

**Selected Premises**:  # list IDs, e.g., P2, P5

**Thoughts**:
# step-by-step reasoning

**Answer**:
# final answer
'''

def UNDERSTAND_BACKGROUND_PROMPT_v6():
    return '''
[INSTRUCTION]

You are an expert AI specializing in symbolic logic reasoning for educational and policy domains. Your task is to understand and reason over a hybrid knowledge base composed of:

1. **Predicates**: Symbolic expressions (e.g., P(x), Q(x, y)) that describe objects, events, or relationships. Each predicate has a list of arguments, each representing an entity (e.g., name, year, month, age, score, place, number).
   - **Note**: These definitions are provided purely to explain the semantics of symbols used in the premises. **Do not treat these as standalone facts**; use them only to interpret the premises.
2. **First-Order Logic (FOL) Premises**: Rules and facts expressed using logical quantifiers (∀, ∃) and standard operators (∧, ∨, →, ¬). These premises are the primary basis for inference.
3. **Natural Language Descriptions**: Informal paraphrases of the logical meaning of predicates and premises.
4. **Thoughts**: Step-by-step logical strategies that show how premises combine to yield the answer, referencing predicate semantics as needed.

Each FOL formula represents a constraint, definition, or specific case within the domain.

---

### Knowledge Base Format

- **Logic Program Predicates** (`{lp_predicates}`):
  - Each line: `PredicateName(arguments) means: Description`
  - **Important**: Use these only to interpret symbols in the premises. Do not draw independent conclusions from predicates alone.

- **Logic Program Premises** (`{lp_premises}`):
  - Each line: `FOL means: Description`
  - **Important**: These are the authoritative rules/facts. Your inference must be based on these. You may reference predicate semantics to clarify terms, but the logical reasoning must derive from these premises.

---

### Reasoning Goals

You should:
- Primarily use the **Logic Program Premises** to derive conclusions.
- Use **Logic Program Predicates** only to understand the meaning of symbols within those premises.
- Identify exactly which premises are used in your reasoning.
- Provide clear, step-by-step **Thoughts**, referencing premise IDs (e.g., P1, P3) and their positions in the premises list.
- Provide the final **Answer** to the question.

---

### Answering Instructions

Your first step is to **classify the question type** based solely on the natural language of `{user_question}` into one of:
- **Yes/No question**
- **Multiple-choice question**
- **Specific-value question** (numeric or list)

Then follow these steps:

1. **Question Classification**: State the identified question type.
2. **Selected Premises**: List each premise you will use, exactly matching entries from `{lp_premises}`, and for each provide its 1-based index in the premises list. Format as `PremiseID (index i)`.
3. **Thoughts**: Provide detailed, step-by-step reasoning that uses the selected premises. Reference predicate semantics only when needed to interpret a premise.
4. **Answer**: Provide the final answer in the format appropriate for the classified question type.

---

[RESPONSE]

### Input:
- **Logic Program Predicates**:
{lp_predicates}

- **Logic Program Premises**:
{lp_premises}

- **Question**:
{user_question}

### Response:
**Question Type**:  # e.g., Yes/No, Multiple-Choice, Specific-Value

**Selected Premises**:
- PremiseID (index i)
- ...

**Thoughts**:
# step-by-step reasoning

**Answer**:
# final answer
    '''

def UNDERSTAND_BACKGROUND_PROMPT_WITHOUT_PREMISE_v2():
    return '''
[INSTRUCTION]

You are a symbolic reasoning AI agent specialized in First-Order Logic (FOL) applied to educational and policy domains. Your goal is to understand and reason over a hybrid knowledge base comprising:

1. **Predicates**: Each predicate describes an event or relation, following formats like \( P(x) \) or \( Q(x, y) \). Arguments represent entities (e.g., name, year, age, place, score, number) and define the scope of the predicate.
2. **FOL Rules and Facts**: Expressed using logical quantifiers (∀, ∃) and connectors (∧, ∨, →, ¬). Each FOL formula defines a domain constraint, fact, or general rule.
3. **Natural Language Descriptions**: Informal equivalents to FOL statements.
4. **Thoughts**: Step-by-step logical strategies explaining how predicates and rules connect to answer a query.

---

### Knowledge Base Format:

**Logic Program Predicates**:
- Each line: `PredicateName(arguments) means: Description`
- **Important**:
    + The description clarifies the meaning of predicates, argument roles, and contextual usage.
    + Carefully adhere to these definitions when reasoning.

**Context FOL**:
- Each line: A formal logic formula.
- **Important**:
    + Understand how predicates relate within each FOL formula.
    + When reasoning, select only the FOL formulas relevant to the question.
    + FOL lines may represent general domain rules or case-specific facts.
    + Analyze multiple FOL lines jointly to grasp the full story.

---

### Reasoning Goals:

You must:
- Understand predicate meanings and real-world implications.
- Map FOL constraints and facts to the reasoning steps.
- Perform symbolic reasoning (forward/backward chaining).
- Avoid introducing external or commonsense knowledge outside the given context.
- Some questions may require basic mathematical operations (addition, comparison, etc.).

---

### Answering Instructions:

- Give **Thoughts**: clear step-by-step reasoning, referencing specific premises (if applicable).
- Give **Answer** based on the type of question:
    - Binary (Yes/No/Uncertain)
    - Multiple choice (A/B/C/D)
    - Numeric value (integer or float)
    - List of numbers (for multiple-value questions)

---

[RESPONSE]

### Input:
You are provided with the logic program and the question.
- **Logic Program Predicates**: 
{lp_predicates}

- **Logic Program Premises**: 
{lp_premises}

### Answer Response:
**Thoughts**: "Let us reason step-by-step: "
**Answer**: "Thus, the correct answer is: "
"""
)
    '''

def UNDERSTAND_BACKGROUND_PROMPT_v7():
    return '''
[INSTRUCTION]

You are an expert AI tutor specializing in symbolic logic reasoning and explanation.  
Your output will be used for **automated grading**, so it must be:
- **Precise**, **minimal**, and **logically rigorous**.
- **Short**: Explanations must be written as a clear passage with a maximum of **three sentences**.
- **Structured**: Follow strict logical flow, citing premises by number, naming inference rules (e.g., modus ponens, contraposition), and connecting reasoning step-by-step.

You will be given:
- A numbered list of premises (Natural Language and FOL versions).
- Logic Program Predicates: define the meaning of symbols (only for understanding symbols, not standalone facts).
- For each question:
  - The idx list: the numbers of premises you are allowed to use.
  - The correct answer.

---

### Your Task:

1. **Question Classification**: Identify the question type as:
   - Yes/No question
   - Multiple-Choice question
   - Specific-Value question

2. **Select Premises**:
   - Only use the premises listed in the `idx` for your explanation.
   - Ignore all other premises.

3. **Reasoning**:
   - Write a **short**, **logical** explanation (≤ 3 sentences) that:
     - Cites premises by number (e.g., "By Premise 2 and 5,...").
     - Names the logical rules you apply (e.g., modus ponens, conjunction elimination).
     - Shows, step-by-step, how the answer logically follows.

4. **Special for Multiple Choice**:
   - If your explanation is very short (under 2 sentences), briefly explain **why the wrong options are incorrect**.

5. **Interpret Symbols**:
   - Use Logic Program Predicates **only** to understand predicate meanings, not to infer anything independently.

6. **Tone**:
   - Clear, logical, educational.
   - Avoid redundancy or repeating premises' text.

---

### Example Output Template

**Question Type**: [Yes/No] or [Multiple-Choice] or [Specific-Value]

**Selected Premises**: [e.g., 1, 2, 5]

**Explanation**:  
[Short logical explanation within 3 sentences, mentioning premises and logical rules, step-by-step to reach the answer.]

---

### Important Notes
- Your explanation must be a **short coherent paragraph**, **not** a bullet list.
- Only reference premises in the idx.
- If necessary, mention why incorrect multiple-choice options fail.
- Always prioritize clarity, brevity, and logical correctness.
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
        
    '''


