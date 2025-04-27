from prover9_solver import FOL_Prover9_Program
import json
from tqdm import tqdm

def load_json(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

import signal
# Custom exception
class TimeoutException(Exception):
    pass

# Signal handler for timeout
def timeout_handler(signum, frame):
    raise TimeoutException()

# Set signal
signal.signal(signal.SIGALRM, timeout_handler)

def load_folio(path):
    datas = load_json(path)

    total = 0
    correct = 0
    none = 0

    progress_bar = tqdm(datas, desc="Accuracy: 0.00% (0/0)")

    for data in progress_bar:
        premises_fol = data['premises-FOL']
        choice_fol = data['conclusion-FOL']
        premises_nl = data['premises']
        choice_nl = data['conclusion']

        premises_fol_string = ''
        for premise in premises_fol.split('\n'):
            premise_string = premise + ' ::: abc \n'
            premises_fol_string += premise_string

        choice_fol_string = choice_fol + ' ::: abc \n'

        logic_program = f"""Premises: 
        {premises_fol_string}
        Conclusion:
        {choice_fol_string}
        """

        prover9_program = FOL_Prover9_Program(logic_program)

        try:
            signal.alarm(5)  # set 5s timeout
            answer, error_message = prover9_program.execute_program()
            signal.alarm(0)  # reset alarm if success
        except TimeoutException:
            print("⏱️ Timeout! Skipping this example.")
            continue

        print("Pred:", answer)
        print("Gold:", data["label"])
        print("Used idx:", prover9_program.used_idx)
        print("---")

        total += 1
        if answer == data["label"]:
            correct += 1

        if str(answer) == "None" :
            none += 1 
        

        acc = correct / (total-none if total != none else 1) if total > 0 else 0
        progress_bar.set_description(f"Accuracy: {acc:.2%} ({correct}/{total-none if total != none else 1})")

    print(f"\n✅ Final Accuracy: {acc:.2%} ({correct}/{total-none if total != none else 1})")
    
load_folio('/data/npl/ICEK/News/SymbolicResoning/folio_v2_train.jsonl')
