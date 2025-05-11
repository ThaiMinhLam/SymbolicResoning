from z3 import *
import re
import json

def create_dynamic_z3_solver(problem_json):
    """
    Dynamically create a Z3 solver for a first-order logic problem
    
    Args:
        problem_json: Problem definition with premises-FOL and choices-FOL
    
    Returns:
        The strongest valid conclusion or None if no valid conclusion found
    """
    # Parse the problem JSON if it's a string
    if isinstance(problem_json, str):
        problem = json.loads(problem_json)
    else:
        problem = problem_json
    
    # Extract the premises and choices
    premises_fol = problem.get("premises-FOL", [])
    choices_fol = problem.get("choices-FOL", [])
    choices_nl = problem.get("choices", [])
    
    # Find all predicate names and individuals (constants)
    all_statements = premises_fol + choices_fol
    
    # Extract predicates and individuals
    predicates = set()
    individuals = set()
    
    for stmt in all_statements:
        # Find predicate names
        pred_matches = re.findall(r'([a-zA-Z_]+)\(([^)]+)\)', stmt)
        for pred_name, args in pred_matches:
            predicates.add(pred_name)
            # Find individuals (assuming they start with uppercase)
            if args != 'x' and not args.startswith('('):
                for arg in args.split(','):
                    arg = arg.strip()
                    if arg[0].isupper():
                        individuals.add(arg)
    
    # Remove 'ForAll' from predicates as it's a quantifier, not a predicate
    while 'ForAll' in predicates:
        predicates.remove('ForAll')
    
    print(f"Found predicates: {predicates}")
    print(f"Found individuals: {individuals}")
    
    # Create Z3 Boolean variables for each predicate-individual combination
    vars_dict = {}
    for pred in predicates:
        for indiv in individuals:
            var_name = f"{pred}_{indiv}"
            vars_dict[var_name] = Bool(var_name)
    
    # Create a solver
    solver = Solver()
    
    # First, process the simple facts (no quantifiers)
    for premise in premises_fol:
        if "ForAll" not in premise:
            pred_match = re.match(r'([a-zA-Z_]+)\(([^)]+)\)', premise)
            if pred_match:
                pred_name = pred_match.group(1)
                indiv = pred_match.group(2)
                var_name = f"{pred_name}_{indiv}"
                
                if var_name in vars_dict:
                    solver.add(vars_dict[var_name])
                    print(f"Added fact: {var_name}")
                else:
                    print(f"Warning: Variable {var_name} not found in dictionary")
    
    # Process quantified statements by instantiating them for each individual
    for premise in premises_fol:
        if "ForAll" in premise:
            match = re.search(r'ForAll\(x, (.*)\)', premise)
            if match:
                body = match.group(1)
                
                # For each individual, instantiate the quantified formula
                for indiv in individuals:
                    # Create a Z3 expression for this instantiation
                    expr = None
                    
                    # Handle implication patterns: (A ∧ B) → C
                    impl_match = re.search(r'\((.*)\) → (.*)', body)
                    if impl_match:
                        antecedent = impl_match.group(1)
                        consequent = impl_match.group(2)
                        
                        # Handle conjunction in antecedent: (A ∧ B)
                        if '∧' in antecedent:
                            conj_parts = antecedent.split('∧')
                            ant_exprs = []
                            
                            for part in conj_parts:
                                part = part.strip()
                                if part.startswith('('):
                                    part = part[1:]
                                if part.endswith(')'):
                                    part = part[:-1]
                                
                                pred_match = re.match(r'([a-zA-Z_]+)\(x\)', part)
                                if pred_match:
                                    pred_name = pred_match.group(1)
                                    var_name = f"{pred_name}_{indiv}"
                                    if var_name in vars_dict:
                                        ant_exprs.append(vars_dict[var_name])
                            
                            # Handle consequent
                            cons_match = re.match(r'([a-zA-Z_]+)\(x\)', consequent)
                            if cons_match:
                                cons_pred = cons_match.group(1)
                                cons_var = f"{cons_pred}_{indiv}"
                                
                                if len(ant_exprs) > 0 and cons_var in vars_dict:
                                    expr = Implies(And(*ant_exprs), vars_dict[cons_var])
                                    # print(f"Added rule: {And(*ant_exprs)} → {vars_dict[cons_var]}")
                        
                        # Handle simple implication: A → B
                        else:
                            ant_match = re.match(r'([a-zA-Z_]+)\(x\)', antecedent)
                            cons_match = re.match(r'([a-zA-Z_]+)\(x\)', consequent)
                            
                            if ant_match and cons_match:
                                ant_pred = ant_match.group(1)
                                cons_pred = cons_match.group(1)
                                
                                ant_var = f"{ant_pred}_{indiv}"
                                cons_var = f"{cons_pred}_{indiv}"
                                
                                if ant_var in vars_dict and cons_var in vars_dict:
                                    expr = Implies(vars_dict[ant_var], vars_dict[cons_var])
                                    print(f"Added rule: {vars_dict[ant_var]} → {vars_dict[cons_var]}")
                    
                    # If we constructed a valid expression, add it to the solver
                    if expr is not None:
                        solver.add(expr)
    
    # Check for consistency of premises
    check_result = solver.check()
    if check_result != sat:
        # print("The premises are inconsistent!")
        return None
    else:
        print("Premises are consistent.")
    
    # Evaluate each choice
    valid_choices = []
    
    for i, choice in enumerate(choices_fol):
        letter = chr(65 + i)  # Convert to A, B, C, etc.
        # print(f"\nEvaluating choice {letter}: {choices_nl[i]}")
        # print(f"FOL: {choice}")
        
        # Simple predicate: pred(Indiv)
        simple_pred_match = re.match(r'([a-zA-Z_]+)\(([^)]+)\)', choice)
        if simple_pred_match:
            pred_name = simple_pred_match.group(1)
            indiv = simple_pred_match.group(2)
            var_name = f"{pred_name}_{indiv}"
            
            if var_name in vars_dict:
                # Create a new solver with all assertions
                choice_solver = Solver()
                for assertion in solver.assertions():
                    choice_solver.add(assertion)
                
                # Check if choice is entailed
                choice_solver.push()
                choice_solver.add(vars_dict[var_name])  # Thêm chính kết luận vào solver
                result = choice_solver.check()
                choice_solver.pop()

                if result == sat:
                    valid_choices.append((letter, choices_nl[i], True))
                    # print(f"{letter}. {choices_nl[i]} - POSSIBLE (satisfiable with premises)")
                else:
                    print(f"{letter}. {choices_nl[i]} - CONTRADICTS premises (unsatisfiable)")
            else:
                print(f"Warning: Variable {var_name} not found for choice {choice}")
            continue
        
        # Disjunction: ¬A ∨ B
        disj_match = re.match(r'¬([a-zA-Z_]+)\(([^)]+)\) ∨ ([a-zA-Z_]+)\(([^)]+)\)', choice)
        if disj_match:
            neg_pred = disj_match.group(1)
            neg_indiv = disj_match.group(2)
            pos_pred = disj_match.group(3)
            pos_indiv = disj_match.group(4)
            
            neg_var = f"{neg_pred}_{neg_indiv}"
            pos_var = f"{pos_pred}_{pos_indiv}"
            
            if neg_var in vars_dict and pos_var in vars_dict:
                # Create a new solver with all assertions
                choice_solver = Solver()
                for assertion in solver.assertions():
                    choice_solver.add(assertion)
                
                # Check if choice is entailed: ¬(¬A ∨ B) = A ∧ ¬B
                choice_solver.push()
                choice_solver.add(And(vars_dict[neg_var], Not(vars_dict[pos_var])))
                result = choice_solver.check()
                choice_solver.pop()
                
                if result == unsat:
                    valid_choices.append((letter, choices_nl[i], True))
                    # print(f"{letter}. {choices_nl[i]} - VALID (entailed by premises)")
                else:
                    print(f"{letter}. {choices_nl[i]} - NOT VALID (not entailed)")
            continue
            
        # Implication: ¬A → ¬B
        impl_match = re.match(r'¬([a-zA-Z_]+)\(([^)]+)\) → ¬([a-zA-Z_]+)\(([^)]+)\)', choice)
        if impl_match:
            ant_pred = impl_match.group(1)
            ant_indiv = impl_match.group(2)
            cons_pred = impl_match.group(3)
            cons_indiv = impl_match.group(4)
            
            ant_var = f"{ant_pred}_{ant_indiv}"
            cons_var = f"{cons_pred}_{cons_indiv}"
            
            if ant_var in vars_dict and cons_var in vars_dict:
                # Create a new solver with all assertions
                choice_solver = Solver()
                for assertion in solver.assertions():
                    choice_solver.add(assertion)
                
                # Check if implication is entailed: ¬(¬A → ¬B) = ¬A ∧ B
                choice_solver.push()
                choice_solver.add(And(Not(vars_dict[ant_var]), vars_dict[cons_var]))
                result = choice_solver.check()
                choice_solver.pop()
                
                if result == unsat:
                    valid_choices.append((letter, choices_nl[i], True))
                    print(f"{letter}. {choices_nl[i]} - VALID (entailed by premises)")
                else:
                    print(f"{letter}. {choices_nl[i]} - NOT VALID (not entailed)")
            continue
            
        print(f"Warning: Unrecognized choice format: {choice}")
    
    # Return the strongest valid conclusion (first one entailed by premises)
    entailed_choices = [(letter, choice) for letter, choice, is_entailed in valid_choices if is_entailed]
    
    if entailed_choices:
        strongest = entailed_choices[0]
        print(f"\nStrongest conclusion: {strongest[0]}. {strongest[1]}")
        return strongest[0]
    else:
        print("No valid conclusions found")
        return None

def load_json(filename):

    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Example usage
if __name__ == "__main__":
    data = load_json('/home/anhkhoa/SymbolicResoning/data/mutiple_choice_with_all_choices_fol.json')[1:10]
    for problem in data:
        print(problem['question'][0])
        strongest_conclusion = create_dynamic_z3_solver(problem)
        print("Right answer: ", problem["answers"][0])

