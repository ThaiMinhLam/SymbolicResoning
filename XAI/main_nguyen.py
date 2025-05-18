import argparse
import yaml
import time
import numpy as np
import copy
import re
from tqdm import tqdm
from icecream import ic
from sentence_transformers import SentenceTransformer, util

# Modules
from src.module import nl_to_fol, Extract_Logic_Progam, reducing, Prover9_K, FOL_Prover9_Program, make_conclusion, Extract_Hypothesis
from src.dataloader_v2.dataset import XAIDataset, load_dataloader
from utils.utils import load_llm, load_yml, load_finetune_model, save_json, map_to_fol, handle_missing_predicates_with_same_name

def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--file_path', type=str, required=True, help='Path to Reasoning Json File')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=int, required=True, default='cuda:0', help='Path to YAML config file')
    
    return parser.parse_args()


def clustering_lp(model, lps, threshold=0.6):
    definitions = [lp.split(":::")[0].strip() for lp in lps]
    embeddings = model.encode(definitions, convert_to_tensor=True)
    list_cosine_scores = util.cos_sim(embeddings, embeddings)
    list_cosine_scores = [scores.detach().cpu() for scores in list_cosine_scores]
    list_idxs = [np.where(cosine_scores > threshold)[0] for cosine_scores in list_cosine_scores]
    select_lps = [list(np.array(lps)[idxs]) for idxs in list_idxs] 
    unique_lps = list(map(list, set(tuple(x) for x in select_lps)))
    return unique_lps


def main():
    args = get_args()
    config = load_yml(args.config)
    config['file_path'] = args.file_path
    ic(config.keys())

    # Load LLM and Required Model
    print("Load base model")
    llm_model, llama_base, tokenizer = load_llm(
        model_id=config['model_id'],
        config=config['model_config'],
        model_type=config['model_type'],
        device=args.device,
    )

    print("logic llama model")
    logicllama = load_finetune_model(
        model_base=llama_base,
        peft_path=config['module_nl2fol']['peft_path'],
        device=args.device
    ) 

    print("Load embedding model")
    model_embedding = SentenceTransformer(config['model_embedding']).cuda(args.device)

    ic(llama_base.device)
    ic(logicllama.device)
    ic(model_embedding.device)
    # Load dataset
    dataset = XAIDataset(config['data']['train'], config['data']['num_samples'])
    dataloader = load_dataloader(dataset, batch_size=config['data']['batch_size'], shuffle=False)

    # -----------------------------------------MODULES-------------------------------------
    # 0. Create Hypothesis
    extract_hypothesis_another = Extract_Hypothesis(
        base_model=logicllama,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        max_output_len=config['module_nl2fol']['max_output_len'],
        tokenizer=tokenizer,
        load_in_8bit=config['module_nl2fol']['load_in_8bit']
    )
    
    # 1. NL2FOL
    module_1 = nl_to_fol(
        base_model=llama_base,
        finetune_model = logicllama,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        tokenizer=tokenizer,
        load_in_8bit = config['module_nl2fol']['load_in_8bit'],
        max_output_len = config['module_nl2fol']['max_output_len'],
        device=args.device,
    )

    # 2. FOL2FOL
    extract_logic_program = Extract_Logic_Progam(
        base_model=logicllama,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        max_output_len=config['module_nl2fol']['max_output_len'],
        tokenizer=tokenizer,
        load_in_8bit=config['module_nl2fol']['load_in_8bit'],
    )

    # 3. SOLVER
    prover9 = Prover9_K(solver = FOL_Prover9_Program)
    

    for step, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        if step > 0:
            break
        # if step != 3:
        #     continue
        premises = batch['premises-nl']
        fol_premises = batch['fol_premises']
        questions = batch['questions']
        reasonings = batch['reasonings']
        answers = batch['answers']

        # Solve tung sample 1 
        for i in range(len(premises)):
            data = {
                'premises-nl': premises[i],
                'fol_premises': fol_premises[i],
                'questions': questions[i],
                'reasonings': reasonings[i],
                'answers': answers[i],
                'LLM-FOL': None,
                'question-FOL': None,
                'logic_program': None,
                'new-fol': None, 
                'maps': None,
                'q_ids': None # Each sub question belongs to which original question
            }

# -----------------------------------------EXECUTE---------------------------------------------
            def check_multiple_choice(question: str):
                if re.findall(r"\n[A-D][\.\)]? (.*?)(?=\n[A-D][\.\)]? |\Z)", question):
                    return True
                return False
            
            new_questions = []
            for question in data["questions"]:
                if check_multiple_choice(question):
                    new_question = make_conclusion(
                        model=llm_model,
                        question=question,
                        config=config
                    )
                    new_questions.append(new_question)
                else: # Cho các câu hỏi loại khác
                    new_question = extract_hypothesis_another.generate_hypothesis(question)
                    new_questions.append(new_question)
            # data['questions'] = new_questions
            
            # 1. NL2FOL
            res_module_1 = module_1.generate_sample(data)
            
            # 2. EXTRACT LOGIC PROGRAM - FOR PREMISES
            logic_program_premise = extract_logic_program.generate_sample(res_module_1)
            # ic(logic_program_premise)
            # raise Exception("Stop Here")

            logic_program_premise = handle_missing_predicates_with_same_name(logic_program_premise, res_module_1['LLM-FOL'])
            res_module_1['logic_program_premise'] = logic_program_premise
            
            # 2. EXTRACT LOGIC PROGRAM - FOR QUESTION (Extract lp của premise + fol) --> Concat
            res_module_1_question = copy.deepcopy(res_module_1)
            res_module_1_question['premises-nl'] = [nl.split('\n')[1:] for nl in res_module_1['questions'] if len(nl.split('\n')) > 1]
            res_module_1_question['premises-nl'] = np.array(res_module_1_question['premises-nl']).flatten().tolist()
            res_module_1_question['LLM-FOL'] = [fol.split('\n')[1:] for fol in res_module_1['question-FOL'] if len(fol.split('\n')) > 1]
            res_module_1_question['LLM-FOL'] = np.array(res_module_1_question['LLM-FOL']).flatten().tolist()

            logic_program_question = extract_logic_program.generate_sample(res_module_1_question)
            # ic(logic_program_question)
            ic(logic_program_question)
            logic_program_question = handle_missing_predicates_with_same_name(logic_program_question, res_module_1_question['LLM-FOL'])
            res_module_1['logic_program_question'] = logic_program_question

            #--- Combine
            lp_predicate_premises = [lp.split(':::')[0].strip() for lp in logic_program_premise]
            ic(logic_program_premise)
            ic(logic_program_question)
            logic_program = logic_program_premise + [lp for lp in logic_program_question if lp.split(':::')[0].strip() not in lp_predicate_premises]
            res_module_1['logic_program'] = logic_program
            ic(logic_program)

            # 3. MAPPING
            clusters = clustering_lp(model_embedding, logic_program, 0.6)
            ic(clusters)
            save_json(
                {"cluster":clusters}, 
                f"/data/npl/ViInfographicCaps/Contest/final_contest/final_code/save/clusters/cluster_{step}.json"
            )
            raise
            # ic(logic_program)
            # ic(clusters)
            all_map_dict = []
            for cluster in clusters:
                if len(cluster) == 1:
                    continue
                ic(cluster)
                tmp_res_module_1 = copy.deepcopy(res_module_1)
                tmp_res_module_1['logic_program'] = cluster
                # tmp_res_module_1 = tmp_res_module_1
                
                mapping_dict = reducing(llm_model, tmp_res_module_1, config)
                all_map_dict.append(mapping_dict)
            maps = {}
            for map_dict in all_map_dict:
                for k, v in map_dict.items():
                    maps[k] = v

            res_module_1['new-fol'] = map_to_fol(
                maps=maps,
                logic_program=logic_program,
                fol=res_module_1['LLM-FOL']
            )
            res_module_1['new-question-FOL'] = map_to_fol(
                maps=maps,
                logic_program=logic_program,
                fol=res_module_1['question-FOL']
            )
            res_module_1['maps'] = all_map_dict
            res_module_1['clusters'] = clusters

            # 4. SOLVER
             # 4. SOLVER
            res_module_1['solver'] = []
            list_premises = res_module_1['new-fol']
            list_questions = res_module_1['new-question-FOL']
            for question in list_questions:
                question = [question]

                result_solver = prover9.solving_questions(list_premises, question)
                res_module_1['solver'].append(result_solver)
        
            path = f"/data/npl/ViInfographicCaps/Contest/final_contest/final_code/save/all_files/log_{step}_{i}.json"
            save_json(res_module_1, path)


if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    time_execute = end - begin
    ic(time_execute)
