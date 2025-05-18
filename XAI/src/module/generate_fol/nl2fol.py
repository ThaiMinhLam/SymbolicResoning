import re
import argparse
import yaml
from sentence_transformers import SentenceTransformer, util
import os 
import json
import time
import numpy as np
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import ParentedTree
import nltk
from icecream import ic

import subprocess
import os

def start_corenlp_server(port=9000):
    # Đảm bảo đang ở đúng thư mục chứa .jar
    os.chdir("/workspace/stanford-corenlp-4.5.6")

    # Nếu đã có server chạy thì không chạy lại
    if not os.path.exists("corenlp.pid"):
        # Tạo lệnh java
        cmd = [
            "java", "-mx4g", "-cp", "*",
            "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
            "-port", str(port),
            "-timeout", "15000"
        ]

        # Mở server ở chế độ nền
        with open("corenlp.log", "w") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            with open("corenlp.pid", "w") as f:
                f.write(str(process.pid))

        print(f"✅ CoreNLP Server started on port {port}")
        time.sleep(5)  # Chờ server khởi động
    else:
        print("⚠️ Server is already running or pid file exists.")

def stop_corenlp_server():
    if os.path.exists("corenlp.pid"):
        with open("corenlp.pid", "r") as f:
            pid = int(f.read())
        os.kill(pid, 9)
        os.remove("corenlp.pid")
        print("🛑 CoreNLP Server stopped.")
    else:
        print("⚠️ No running server found.")

# Danh sách các từ cần loại bỏ nếu đứng đầu
AUX_MODALS = [
    "am", "is", "are", "was", "were", "be", "do", "does", "did",
    "must", "should", "shall", "will", "would", "can", "could", "may", "might", "ought to"
]

NEGATIONS = [
    "am not", "is not", "isn't", "are not", "aren't", "was not", "wasn't", "were not", "weren't",
    "do not", "don't", "does not", "doesn't", "did not", "didn't",
    "must not", "mustn't", "should not", "shouldn't", "shall not", "shan't",
    "will not", "won't", "would not", "wouldn't", "cannot", "can't",
    "could not", "couldn't", "may not", "might not", "ought not to","have" ,"has", "had" ,"having","not"
]

# Gộp hai danh sách lại để xử lý chung
REMOVE_PATTERNS = NEGATIONS + AUX_MODALS

# Tạo regex để nhận diện nếu chuỗi bắt đầu bằng các mẫu trên
remove_regex = re.compile(rf"^({'|'.join(re.escape(p) for p in sorted(REMOVE_PATTERNS, key=len, reverse=True))})\s+", re.IGNORECASE)

# Redundance
redundant_list = ['then' ,'that']

def clean_predicate(phrase):
    return remove_regex.sub('', phrase.strip())

# Tách nếu có "and"/"or"
def split_compound(phrase):
    return [p.strip() for p in re.split(r'\band\b|\bor\b', phrase)]

# Tổng xử lý
def extract_clean_predicates(raw_phrase):
    predicates = []
    for part in split_compound(raw_phrase):
        cleaned = clean_predicate(part)
        if cleaned:
            predicates.append(cleaned)
    return predicates[0]

def check_multiple_choice(question: str):
    if len(question.split('\n1 ')) > 2:
        return True
    return False


def extract_subject (parse_tree):
    # Extract the first noun found in NP_subtree
    subject = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'NP'):
        for t in s.subtrees(lambda y: y.label().startswith('NN')):
            output = [t[0], extract_attr(t)]
            # Avoid empty or repeated values
            if output != [] and output not in subject:
                subject.append(output)
    if len(subject) != 0: return subject[0]
    else: return ['']

def extract_lowest_level_predicate(parse_tree):
    predicates = []

    for vp in parse_tree.subtrees(lambda t: t.label() == 'VP'):
        # Nếu subtree này KHÔNG chứa bất kỳ VP con nào khác (ngoài chính nó)
        has_descendant_vp = any(
            sub.label() == 'VP' and sub != vp
            for sub in vp.subtrees()
        )

        if not has_descendant_vp:
            predicate_phrase = ' '.join(vp.leaves())
            predicates.append(predicate_phrase)

    return predicates


def extract_object(parse_tree):
    # Extract the first noun or first adjective in NP, PP, ADP siblings of VP_subtree
    objects, output, word = [],[],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label() in ['NP','PP','ADP']):
            if t.label() in ['NP','PP']:
                for u in t.subtrees(lambda z: z.label().startswith('NN')):
                    word = u
            else:
                for u in t.subtrees(lambda z: z.label().startswith('JJ')):
                    word = u
            if len(word) != 0:
                output = [word[0], extract_attr(word)]
            if output != [] and output not in objects:
                objects.append(output)
    if len(objects) != 0: return objects[0]
    else: return ['']

def extract_attr(word):
    attrs = []
    # Search among the word's siblings
    if word.label().startswith('JJ'):
        for p in word.parent():
            if p.label() == 'RB':
                attrs.append(p[0])
    elif word.label().startswith('NN'):
        for p in word.parent():
            if p.label() in ['DT','PRP$','POS','JJ','CD','ADJP','QP','NP']:
                attrs.append(p[0])
    elif word.label().startswith('VB'):
        for p in word.parent():
            if p.label() == 'ADVP':
                attrs.append(p[0])
    # Search among the word's uncles
    if word.label().startswith('NN') or word.label().startswith('JJ'):
        for p in word.parent().parent():
            if p.label() == 'PP' and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    elif word.label().startswith('VB'):
        for p in word.parent().parent():
            if p.label().startswith('VB') and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    return attrs


def remove_substrings(lst):
    result = []
    for item in lst:
        if not any((item != other and item in other) for other in lst):
            result.append(item)
    return result


def remove_redundant(input):
    for word in redundant_list:
        input = input.replace(word, '')
    return input.strip()

def extract_result(output_text):
    output = []
    pattern = r"\w+ \:+ (\w+)"
    lines = output_text.split("\n")
    for text in lines:
        match = re.search(pattern, text)
        if match:
            output.append(match.group(1))
    return output

class chatAgent:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_response(self, prompt):
        response = self.pipeline(prompt)
        return response[0]['generated_text'][len(prompt):].strip()

class nl_to_fol_new(chatAgent):
    def __init__(self, pipeline):
        super().__init__(pipeline)

    def convert(self, list_predicates):
        dic = {}
        
        prompt = self.get_predicate_fol_prompt(list_predicates)
        response = extract_result(self.get_response(prompt))
        ic(response)
        for idx, pred in enumerate(list_predicates):
            dic[pred] = pred + ':::' + response[idx].strip()

        return dic

    def get_predicate_fol_prompt(self, predicate_nl):
        PROMPT_LP = """<s>[INST]
            ### Task: For each given **natural language predicate**, generate a corresponding **First-Order Logic (FOL) predicate** that faithfully captures its logical meaning.

            You are given:
            - A **premise** that provides the contextual background or knowledge base from which the logical predicates are derived.
            - Each sentence describes a logical relationship or property involving one or more entities.
            - Each definition is a **complete English sentence** and clearly specifies the **roles of all arguments** (e.g., who performs an action, who receives it).

            #### Your Goal:
            For every natural language predicate, produce a well-formed **FOL predicate expression** that:
            - Accurately reflects the **logical meaning** conveyed by the sentence.
            - Identifies the **correct number of arguments** (e.g., x, y, z) based on the entities involved.
            - Creates a **clear and self-explanatory predicate name** (in English) that summarizes the core relationship or property described.
            - Assigns each argument to its correct role (e.g., subject, object, action recipient).
            - Uses **logical arguments (e.g., x, y, z)** to represent the entities involved.
            - Reflects the **correct argument structure** (e.g., number of arguments, their logical roles).
            - Uses logical notation: `PredicateName(x)`, `PredicateName(x, y)`, etc.

            ### Instructions:
            1. **Analyze each natural language predicate carefully** to determine:
                - The number of **entities** involved (i.e., how many arguments the predicate should take).
                - The **core relationship or property** being described (used to form the predicate name).
                - The **roles** of the arguments (e.g., agent, theme, location, object).
                - If the predicate involves a **fixed object like a degree, title, award, job, etc.**, represent it as a **constant**, not part of the predicate name.

            2. **Rules for constructing the FOL predicate**:
                - The **predicate name** must summarize the core action/relation/property using meaningful English terms.
                - The number of arguments in the predicate must match the number of entities involved.
                - Maintain the **semantic roles** of the arguments
                - Avoid copying the sentence structure directly — instead, generate a compact and expressive predicate name.
                - Use consistent naming style: either `CamelCase` or `snake_case`.
                - **Do NOT merely paraphrase or copy** the sentence — you must **analyze** and convert it into a **logical representation**.

            3. **Output Format**:
                - Each output must be exactly **one line per definition**.
                - Each line must follow this format:
                    Natural Language Predicate ::: FOL-Predicate
                - Use **exactly three colons** (` ::: `) as the separator.
                - Example:
                    + is a Teacher ::: Teacher(x)
                    + do test on y subject ::: DoTestSubject(x, y)

            4. **Final Requirements**:
                - You must define **every single natural language predicate** provided in the list.
                - Each FOL predicate must be **logically sound, syntactically correct, and match the right number of arguments and roles**.
                - **Do not skip any natural language predicates**.
                - No extra commentary or explanation — only the list of converted predicates.
                - The number of output lines must **exactly match** the number of input definitions.

            ### Input:
            - List of Natural Language Predicate: {predicate_nl}
            [/INST]
            Output: </s>"""
        return PROMPT_LP.format(predicate_nl=predicate_nl)

    def construct_logic_program(self, lps):
        logic_program = [lp for lp in lps]
        return logic_program

    def premise_to_fol_prompt(self, logic_program, nl_premise, subject):
        prompt = f"""
        [INST]
        You are a **formal-logic expert**.

        TASK ▸ Convert the natural-language sentence (Premise-NL) into **one
        well-formed first-order-logic formula (Premise-FOL)**, suitable for a
        logic-program / Prolog-style rule base.

        ════════════════  OUTPUT RULES  ════════════════
        - Return **exactly ONE line** – the FOL formula only.  
        - No bullet numbers, no “:::”, no code-fences.  
        - Keep the NL ordering:  antecedent ⇒ consequent.  
        - Allowed symbols:  ∀  ∃  →  ∧  ∨  ¬  ( )  ,  x y z … constants.  
        - Do **NOT** invent new predicate names; use ONLY those listed in
        **Predicate Declarations** below.  
        - Express negation with **¬P(x)** – never create a “NotP” predicate.

        ══════════  PREDICATE DECLARATIONS  ══════════
        {logic_program}

        ════════════  FORMATTING GUIDELINES  ══════════
        1. **Quantifiers**  
        - Use **∀x** when the sentence states a rule applying to every
            instance.  
        - Use **∃x** only when NL asserts existence (“there is / some”).  
        2. **Arity**  
        - P(x) – intrinsic property or action with fixed object.  
        - R(x,y) – relation where the second argument can vary.  
        - More roles?  S(x,y,z,…)  
        3. **Named individuals** are constants (Alice, Server42,…).  
        Never quantify over a constant.  
        4. **Negation & Connectives**  
        - NOT → ¬P(x)  
        - AND → P(x) ∧ Q(x)  
        - OR  → P(x) ∨ Q(x)  
        - IF-THEN → wrap antecedent in ( ) if it contains ∧ / ∨.  
            Example:  ∀x (ReadScripts(x) ∨ PresentTooLong(x)) → GetFAssign(x)

        ══════════  STRICT STYLE RULES  ══════════
        - Use ONLY the predicate names supplied in  "Predicate Declarations".  
        + Never glue context or suffixes onto a name.  
        + Express context as an extra literal:  RedoAssignment(x) ∧ ReadScripts(x).

        - **Negation**  
        + Write ¬P(x) or ¬Q(x,y).  
        + Do NOT invent “NotP” / “NonP” / “No_P”.  

        - **Quantifiers**  
        + Universal rule → ∀.  
        + Existential only when NL says “there exists / some…”.  
        + Never quantify over a constant.

        - **Constants / Named individuals**  
        + If a sentence only states facts about a constant, output a pure
            conjunction of atomic predicates – no ∀/∃, no equality tricks.  
            Example:  “Alice is absent and got an A.”  
            FOL: Absent(Alice) ∧ GotGradeA(Alice)

        - **Contextual WHEN / WHILE clauses**  
        + Keep the main predicate names intact; prepend the context literal.  
            “Absent **when redoing**” ⇒  RedoAssignment(x) ∧ Absent(x)

        - **One-line rule** – collapse multi-line antecedents with spaces.  
        Example final form:  
        ∀x (RedoAssignment(x) ∧ ReadScripts(x) ∧ PresentTooLong(x) → FailCourse(x))


        ════════════  INPUT  ════════════
        Premise-NL : {premises_nl}
        Subject     : {subject}
        ═════════════════════════════════
        [/INST]
        [OUTPUT]
        """
        return prompt
    
    def _extract_fol_from_response(self, response):
        """
        response : List[List[Dict]] – raw output from self.pipeline
        return   : List[str]        – one FOL formula per premise (order preserved)
        """
        fol_list = []

        for batch in response:                  #   1st level: prompts in the batch
            for item in batch:                  #   2nd level: {"generated_text": ...}
                txt = item["generated_text"]

                # split by every [OUTPUT] tag; ignore the header part before the 1st tag
                segments = txt.split("[OUTPUT]")[1:]

                for seg in segments:
                    # keep only before the next [INPUT] (if any)
                    if "[INPUT]" in seg:
                        seg = seg.split("[INPUT]")[0]

                    # strip each line + discard empties
                    lines = [ln.strip() for ln in seg.splitlines() if ln.strip()]

                    # if nothing left → skip
                    if not lines:
                        continue

                    # some models indent first line with tabs/spaces → remove extra white space
                    joined = re.sub(r"\s+", " ", " ".join(lines))

                    fol_list.append(joined)

        return fol_list

    def post_process_fol(fol: str) -> str:
        fol = re.sub(
            r"\bnot\s+(∀|∃|\w+\([^)]*\))",
            lambda m: f"¬{m.group(1)}",
            fol,
            flags=re.I,
        )

        fol = re.sub(
            r"\bNot([A-Z]\w*)\s*\(([^)]*)\)",
            lambda m: f"¬{m.group(1)}({m.group(2)})",
            fol,
        )

        return fol

    def convert_premise_to_fol(
        self,
        premise_nl_list,
        premise_nl_pred_dict,
        dic_predicates,
        premise_nl_subject):

        # 1) Tạo prompt cho từng premise
        prompt_list = []
        for nl in premise_nl_list:
            logic_program = [dic_predicates[p] for p in premise_nl_pred_dict[nl]]
            prompt_list.append(
                self.premise_to_fol_prompt(
                    logic_program = logic_program,
                    nl_premise    = nl,
                    subject       = premise_nl_subject[nl]
                )
            )
        raw_response = self.pipeline(prompt_list, batch_size=len(prompt_list))
        fol_formulas = self._extract_fol_from_response(raw_response)
        fol_formulas = [post_process_fol(f) for f in fol_formulas]   
        return fol_formulas         


class predicate_nl_extractor(chatAgent):
    def __init__(self, pipeline_mistral, mapping_model, threshold=0.6, port = 8000):
        super().__init__(pipeline_mistral)
        self.model = pipeline_mistral
        self.mapping_model = mapping_model
        self.threshold = threshold
        self.dep_parser = CoreNLPDependencyParser(url='http://0.0.0.0:8000')
        self.pos_tagger = CoreNLPParser(url='http://0.0.0.0:8000', tagtype='pos')
    
    def triplet_extraction(self, input_sent, output=['parse_tree','spo','result']):
        input_sent = remove_redundant(input_sent)
        # Parse the input sentence with Stanford CoreNLP Parser
        pos_type = self.pos_tagger.tag(input_sent.split())
        parse_tree, = ParentedTree.convert(list(self.pos_tagger.parse(input_sent.split()))[0])
        dep_type, = ParentedTree.convert(self.dep_parser.parse(input_sent.split()))


        # pos_dict = {word: tag for word, tag in pos_type}
        # # input_sent = ' '.join([word for word, tag in pos_dict.items() if tag not in ['RB']])


        pos_dict = {word: tag for word, tag in pos_type}
        def extract_words_with_joined_nns(inputs, pos_type):
            inputs = inputs.replace(',', '')
            inputs = inputs.replace('.', '')
            input_split = inputs.split()

            result = []
            buffer = []

            def flush_buffer():
                if buffer:
                    result.append('_'.join(buffer))
                    buffer.clear()

            for word in input_split:
                if word in pos_type and 'NN' in pos_type[word]:
                    buffer.append(word)
                else:
                    flush_buffer()
                    result.append(word)
            flush_buffer()
            return result

        # print(' '.join(extract_words_with_joined_nns(input_sent.split(), pos_dict)))
        input_sent = ' '.join(extract_words_with_joined_nns(input_sent, pos_dict))
        # Parse the input sentence with Stanford CoreNLP Parser

        pos_type = self.pos_tagger.tag(input_sent.split())
        parse_tree, = ParentedTree.convert(list(self.pos_tagger.parse(input_sent.split()))[0])
        dep_type, = ParentedTree.convert(self.dep_parser.parse(input_sent.split()))


        # Extract subject, predicate and object
        subject = extract_subject(parse_tree)
        predicates = extract_lowest_level_predicate(parse_tree)
        objects = extract_object(parse_tree)

        # Tạo dict từ -> loại POS (từ pos_type)
        pos_dict = {word: tag for word, tag in pos_type}

        # In ra theo yêu cầu output
        if 'parse_tree' in output:
            print('---Parse Tree---')
            tree = parse_tree.pretty_print()

        subject_P = {word:tpe for word, tpe in pos_type if tpe == 'NNP'}
        # vbp_list = {word : tpe for word, tpe in pos_type if tpe in ['VBP'] }


        def remove_vbp(pred):
            for word, tpe in subject_P.items():
                pred = pred.replace(word, '')
            return pred.strip()

        def post_process(predicates):
            res = []
            for i in range(len(predicates)):
                res.append(remove_vbp(predicates[i]))
                if len(predicates[i].split(' ')) == 1 and pos_dict[predicates[i]] == 'VBZ':
                    res.pop()
            return res

        predicates = post_process(predicates)

        # Trả về kết quả (subject, predicate, object)
        return subject, predicates, pos_dict, subject_P


    def extract(self, premise_list, conclusion_list):
        premise_pred_dic = {}
        subject_pred_dic = {}

        pred_list = []
        sub_list = []
        nl_total = premise_list + conclusion_list
        nl_total_list = []

        for idx, premise_nl in enumerate(nl_total):
            split_items = []
            if idx >= len(premise_list) - 1 and check_multiple_choice(premise_nl):
                split_items = premise_nl.split('\n1 ')
            else:
                split_items = [premise_nl]

            for item in split_items:
                # Extract predicate
                try:
                    response_subject, extracted_preds, dic, subP = self.triplet_extraction(item)
                except Exception as e:
                    print(f"Error processing item: {item}")
                    print(f"Exception: {e}")
                    continue
                    
                extracted_preds = remove_substrings(extracted_preds)

                premise_pred_dic[item] = extracted_preds
                pred_list.extend(extracted_preds)

                
                subject_pred_dic[item] = response_subject[0]
                sub_list.append(response_subject[0])

                nl_total_list.append(item)

                print(item)
                print(extracted_preds)

        # Clustering predicates
        pred_mapping = {}
        pred_list_final = []
        mapping_matrix = self.clustering(pred_list)
        for group in mapping_matrix:
            for pred in group:
                pred_mapping[pred] = group[0]
            if len(group) > 0:
                pred_list_final.append(group[0])

        # Clustering subjects
        sub_mapping = {}
        sub_list_final = []
        mapping_matrix_sub = self.clustering(sub_list)
        for group in mapping_matrix_sub:
            for sub in group:
                sub_mapping[sub] = group[0]
            if len(group) > 0:
                sub_list_final.append(group[0])

        # Map predicates to canonical form
        for nl in nl_total_list:
            premise_pred_dic[nl] = [pred_mapping.get(pred, pred) for pred in premise_pred_dic[nl]]

        # Map subject to canonical form
        for nl in nl_total_list:
            raw_sub = subject_pred_dic[nl]
            subject_pred_dic[nl] = sub_mapping.get(raw_sub, raw_sub)
        
        # print(pred_mapping, flush = True)

        return premise_pred_dic, pred_list_final, subject_pred_dic, sub_list_final
        # map premise nl over predicates

    def get_predicate_from_nl(self, sentence):
        prompt = f"""
    You are a symbolic reasoning assistant.

    ### Task:
    Given a natural language statement (premise), extract all *positive predicate phrases* that describe main actions or properties. Focus only on actions or qualities expressed — remove anything that is negative or auxiliary.

    ---

    ### Instructions:

    1. **Extract only predicate phrases** — no subject (e.g., "students", "Python project", etc.).
    2. **Only keep positive statements**:
    - ⚠️ Remove all negations like: "not", "does not", "do not", "is not", "was not", "aren’t", "won’t", etc.
    - ✂ Example: "does not follow PEP 8 standards" → "follow PEP 8 standards"
    - ✂ Example: "is not optimized" → "optimized"
    3. **Remove auxiliaries/modals**: am, is, are, was, were, be, do, does, did, must, can, should, etc.
    4. **Split compound predicates**: Separate items joined by "and"/"or"
    5. **Preserve full meaning**: Keep key complements like "by the team", "for graduation", etc.
    6. **Do not** invent new predicates or add any extra text.
    ---

    ### Output Format (strict, no explanation, separate each predicate with a | ):
    | [Predicate 1]
    | [Predicate 2]
    ...

    ### Premise:
    {sentence}
    ### Output:
    """
        return prompt

    def identify_subject_prompt(self, nl_premise):
        return f"""
        ### Task: Identify the correct logical subject in the given natural language premise.

        You are given a premise sentence that may describe:
        - A general rule (e.g., "If a student submits an assignment...")
        - A specific fact about an individual (e.g., "John submitted the assignment.")

        Your job is to decide whether the subject of the premise should be:
        - `x` → if the sentence describes a general case about a generic person.
        - the actual **name mentioned** (e.g., "John", "Sophia") → if the sentence refers to a specific known individual.

        ### Rules:
        1. If the subject is a general noun phrase like "a student", "students", "anyone", or "employees", return: `x`
        2. If the subject is a named individual (e.g., "John", "Sophia", "Alice"), return their exact name as it appears.
        3. Return only the subject — no extra text, no explanation.

        ### Input:
        Premise: {nl_premise}

        ### Output:
        """
    
    def extract_predicates_from_table(self, table_str):
        lines = table_str.strip().split("\n")
        # Loại bỏ dấu | ở đầu và cuối, rồi tách lấy các predicate
        predicate_lines = [line.replace("|", "").strip() for line in lines if "|" in line and not "Predicate" in line and not "----" in line]
        # Đảm bảo rằng predicate không có dấu | và khoảng trắng thừa
        predicates = [p.strip().lower() for p in predicate_lines if p]
        return predicates


    def clustering(self, pred_nl_list):
        lps_list = list(pred_nl_list)  # convert set to list
        definitions = [lp.strip() for lp in lps_list]
        embeddings = self.mapping_model.encode(definitions, convert_to_tensor=True)
        list_cosine_scores = util.cos_sim(embeddings, embeddings)
        list_cosine_scores = [scores.detach().cpu() for scores in list_cosine_scores]
        list_idxs = [np.where(cosine_scores > self.threshold)[0] for cosine_scores in list_cosine_scores]
        select_lps = [list(np.array(lps_list)[idxs]) for idxs in list_idxs]
        unique_lps = list(map(list, set(tuple(x) for x in select_lps)))
        # unique_lps = [pred_group for pred_group in unique_lps if len(pred_group) > 1]
        return unique_lps