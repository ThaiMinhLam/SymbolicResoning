def PARAPHRASE_PROMPT_TEMPLATE_FULL_GPT(): 
    '''
        GPT Prompt full
    '''
    return f'''
        {HEAD_INSTRUCTION()}

        Please simplify the following logic problem statements and convert each into a pair of outputs:
            - "Simplified Statement": A concise version of the original statement capturing the core idea.
            - "Formal Logical Expression": The formal logic expression (e.g., A → B) derived from the simplified statement.
        
        Each original statement is separated by a dot formula "."
        The number of "Simplified Statement" and "Formal Logical Expression" items must exactly match the number of input statements.
        Each "Formal Logical Expression" at index i must be inferred from the corresponding "Simplified Statement" at index i.
        
        You may also output the following optional sections:
            - "Important Information:": List key facts from the original text that do not fit neatly into logical expressions but are still relevant.
            - "Cross Relation:": List inferred logical relations that connect multiple statements together. These should be derived when the consequent of one statement connects to the antecedent of another.
        
        Cross Relations Guidelines:
            - A Cross Relation links the consequent (then-part) of one logical statement with the antecedent (if-part) of another.
            - Object names in Cross Relations must remain exactly the same as in the original statements (no renaming).
            - Cross Relations should reflect hidden or real-world dependencies, even if not explicitly stated.
            - The number of Cross Relations should not exceed (number of original statements - 1).
        
        Reminder:
            - In A → B, A is the antecedent, and B is the consequent.
            - Use negation (¬) when appropriate, especially when inferring relations like "If not A, then not B".
        
        Example:
        Statement 1: You can only buy a car in order to take your car home.
        Simplified Statement 1: You must buy a car to take it home.
        Formal Logical Expression 1: Buy(Car) → Take(Car)
        
        Statement 2: You must put fuel in your car in order to drive.
        Simplified Statement 2: You must fuel your car to drive.
        Formal Logical Expression 2: PutFuel(Car) → Drive(Car)
        
        Cross Relation:
        ¬Take(Car) → ¬PutFuel(Car)
        Explanation: If you don't buy the car, you can't take it home, so you can't fuel it to drive.
        
        Another Example:
        Statement 1: Register → FullRequirement
        Statement 2: Eligible → HasCertificate
        Cross Relation: ¬FullRequirement → ¬Eligible
        Explanation: Without registering, you don't meet the full requirement and can't become eligible.
        
        Now, analyze the input statements and generate the corresponding Simplified Statements, Formal Logical Expressions, and any possible Cross Relations.

        ### Answer Response: 
       Return only the Natural Language Statements, use the following output format
            "Simplified Statement 1" is:    || "List Objects 1" is:     || "List Actions 1" is:     || (Optional) "List Instances 1" is:
            "Simplified Statement 2" is:    || "List Objects 2" is:     || "List Actions 2" is:     || (Optional) "List Instances 2" is:
            "Simplified Statement 3" is:    || "List Objects 3" is:     || "List Actions 3" is:     || (Optional) "List Instances 3" is:
            ...
            "Simplified Statement n" is:    || "List Objects n" is:     || "List Actions n" is:     || (Optional) "List Instances n" is:

            ----
            "List Of Cross Relations" is: 
            "Explaination for Cross Relations" is:
    '''

def PARAPHRASE_PROMPT_TEMPLATE_GPT_ME(): 
    '''
        GPT Prompt + Me
    '''
    return f'''
        {HEAD_INSTRUCTION()}

        Please simplify the following logic problem statement and convert it into a "Simplified Statement" and "Formal Logical Expression". Each Statement is seperated by dot formula ".". Extract the core information from each statement and present its logical structure in a concise form. Ensure that the simplified information maintains all logical relationships of the original statement. The number of "Simplified Statement" and "Formal Logical Expression" must be equal to the number of questions statement. "Formal Logical Expression" number i is infered from corresponding "Simplified Statement" number i 
        You can add "Important Infomation" items if you think there is some information that is important but is not appropriate to parallel with the Logical Statements.
        You can add "Cross Relation" items if you think there is some relations that is appeared across different Statements, which can be inferred as cross relations when some Statements are the results of the another Statement.
        
        Cross Relations Guidelines:
            - A logical expression like A → B means: A is the antecedent (if-part), and B is the consequent (then-part).
            - "Cross Relation" is always inferred by combining two different logical statements.
            - A "Cross Relation" is created by linking the consequent of one statement to the antecedent of another statement.
            - Object names in the "Cross Relation" must stay the same as in the original statements. No renaming is allowed. This keeps logic and naming consistent.
            - The number of "Cross Relation" should not exceed (number of original statements - 1).
        Sometimes the relationship is not explicitly stated but can be understood from real-world context.
        
        For example:
            - Statement 1: "Register → FullRequirement"
            - Statement 2: "Eligible → HasCertificate"
        Even though there is no direct connection between these two statements, we know from real-world logic that if someone did **not** register, they cannot meet the full requirements, and they also **not** eligible. Therefore, the "Cross Relation" is: ¬FullRequirement → ¬Eligible
        
        Reminder:
            - In A → B, A is the antecedent, and B is the consequent.
            - "Cross Relation:" is inferred logical relations that connect two statements together. These should be derived when the consequent of one statement connects to the antecedent of another.
        Based on these instructions, read the provided statements and generate possible "Cross Relation" that can be logically inferred.
        
        $$$$
        ### Answer Response: 
        Return only the Natural Language Statements, use the following output format
            "Simplified Statement 1" is:    || "Formal Logical Expression 1" is:
            "Simplified Statement 2" is:    || "Formal Logical Expression 1" is:
            "Simplified Statement 3" is:    || "Formal Logical Expression 1" is:
            ...
            "Simplified Statement n" is:    || "Formal Logical Expression 1" is:

            ----
            "List Of Cross Relations" is: 
            "Explaination for Cross Relations" is:
    '''

def PARAPHRASE_PROMPT_TEMPLATE_instruct_cross_relation_not_optimzed(): 
    return f'''
        {HEAD_INSTRUCTION()}

        Please simplify the following logic problem statement and convert it into a "Simplified Statement" and "Formal Logical Expression". Each Statement is seperated by dot formula ".". Extract the core information from each statement and present its logical structure in a concise form. Ensure that the simplified information maintains all logical relationships of the original statement. The number of "Simplified Statement" and "Formal Logical Expression" must be equal to the number of questions statement. "Formal Logical Expression" number i is infered from corresponding "Simplified Statement" number i 
        You can add "Important Infomation:" items if you think there is some information that is important but is not appropriate to parallel with the Logical Statements.
        You can add "Cross Relation:" items if you think there is some relations that is appeared across different Statements, which can be inferred as cross relations when some Statements are the results of the another Statement, for example:
            - "Statement 1": You can only buy car in order to take your car home || Buy(Car) → Take(Car).
            - "Statement 2": You must put fuel to your car in order to drive || PutFuel(Car) → Drive(Car).
            - Cross Relations: If you can't take your car home, you can't put fuel to your car || ¬Take(Car) → ¬PutFuel(Car).
            - Explanation: If you don't buy car, you can't take your car home, therefore you can not put fuel to your car to drive car.
        Make sure that number of Cross Relation <= number of Logical Statement minus 1 
        Note that: For example, having the logical expression A → B. 
            - A is the antecedent, B is a consequent.
            - Cross Relations always implied by two different statements infered from the question's statements.
            - Cross Relations are created by linking the consequent (then-part) of one logical statement with the antecedent (if-part) of another logical statement. The object names in the Cross Relations are exactly the same as those used in the original statements—no renaming is done. This keeps the logic and naming consistent across statements.
        $$$$
        ### Answer Response: 
        Return only the Natural Language Statements, use the following output format
            "Simplified Statement 1" is:    || "Formal Logical Expression 1" is:
            "Simplified Statement 2" is:    || "Formal Logical Expression 1" is:
            "Simplified Statement 3" is:    || "Formal Logical Expression 1" is:
            ...
            "Simplified Statement n" is:    || "Formal Logical Expression 1" is:

            ----
            "List Of Cross Relations" is: 
            "Explaination for Cross Relations" is:
    '''
'''
    In this task, your goal is to find possible hidden or implied logical relations between statements, called Cross Relations.

- A logical expression like A → B means: A is the antecedent (if-part), and B is the consequent (then-part).
- Cross Relations are always inferred by combining two different logical statements.
- A Cross Relation is created by linking the consequent of one statement to the antecedent of another statement.
- Object names in the Cross Relation must stay the same as in the original statements. No renaming is allowed. This keeps logic and naming consistent.

Sometimes the relationship is not explicitly stated but can be understood from real-world context.
For example:
- Statement 1: "Register → FullRequirement"
- Statement 2: "Eligible → HasCertificate"
Even though there is no direct connection between these two statements, we know from real-world logic that if someone did **not** register, they cannot meet the full requirements.
Therefore, the hidden logic is: ¬FullRequirement → ¬Eligible

Based on these instructions, read the provided statements and generate possible Cross Relations that can be logically inferred.
'''

def PARAPHRASE_PROMPT_TEMPLATE_isoke(): 
    return '''
        Please paraphrase and simplify the following logic problem statement and convert it into a formal logical expression. Extract the core information from each statement and present its logical structure in a concise form. Ensure that the simplified information maintains all logical relationships of the original statement. Keep all the objects and relationship beetween that object in the statement.

        Use the following output format:
            "Simplified Statement 1" is:
            "Simplified Statement 2" is:
            "Simplified Statement 3" is:
            ...
        Note that: Only return formal logical expression with no explaination, avoid return logic statement.
        Additionally, you can add "Cross Relation:" items if you think there is some relations that is appeared across different Statements, which can be inferred as cross relations when some Statements are the results of the another Statement, for example:
            - "Statement 1": You can only buy car in order to take your car home || Buy(Car) → Take(Car).
            - "Statement 2": You must put fuel to your car in order to drive || PutFuel(Car) → Drive(Car).
            - Cross Relations: If you can't take your car home, you can't put fuel to your car || ¬Take(Car) → ¬PutFuel(Car).
            - Explanation: If you don't buy car, you can't take your car home, therefore you can not put fuel to your car to drive car.
        Make sure that number of Cross Relation <= number of Logical Statement minus 1 
    '''


def PARAPHRASE_PROMPT_TEMPLATE_3(): 
    return '''
        Please paraphrase and simplify the following logic problem statement and convert it into a formal logical expression. Extract the core information from each statement and present its logical structure in a concise form. Ensure that the simplified information maintains all logical relationships of the original statement. Keep all the objects and relationship beetween that object in the statement.

        Use the following output format:
            "Simplified Statement 1" is:
            "Simplified Statement 2" is:
            "Simplified Statement 3" is:
            ...
        Note that: 
            - Only return formal logical expression with no explaination, avoid return logic statement.
            - Try using ¬ for negative expression like no, did not, etc.
        
        ----
        Additionally, you can add "Cross Relation:" items, based on the original statement and formal logical expression recently extracted, if you think there is some relations that is appeared across different Statements, which can be inferred as cross relations when some Statements are the results of the another Statement, for example:
            - "Statement 1": You can only buy car in order to take your car home || "Logical Expression": BuyCar → TakeCar.
            - "Statement 2": You must put fuel to your car in order to drive || "Logical Expression": PutFuelCar → DriveCar.
            - Cross Relations: If you can't take your car home, you can't put fuel to your car || "Logical Expression": ¬TakeCar → ¬PutFuelCar.
        Note that: For example, having the logical expression A → B. 
            - A is the antecedent, B is a consequent.
            - Cross Relations always implied by two different statements infered from the question's statements.
            - Cross Relations are created by linking the consequent (then-part) of one logical statement with the antecedent (if-part) of another logical statement. The object names in the Cross Relations are exactly the same as those used in the original statements—no renaming is done. This keeps the logic and naming consistent across statements.
        Make sure that number of Cross Relation <= number of Logical Statement minus 1 
    '''
# - Object names in the cross relation are reused identically from the antecedent or consequent of the original expression.
