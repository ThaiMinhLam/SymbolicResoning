import argparse
import yaml
import time
import numpy as np
import copy
from tqdm import tqdm
from icecream import ic
from sentence_transformers import SentenceTransformer, util

# Modules
from src.module import nl_to_fol, Extract_Logic_Progam, reducing, extract_lp, Prover9_K, FOL_Prover9_Program
from src.dataloader_v2.dataset import XAIDataset, load_dataloader
from utils.utils import load_llm, load_yml, load_finetune_model, save_json, map_to_fol

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


def remove_intro_phrase_if_common(sentences: list[str]):
    def extract_intro_phrase(sentence: str):
        doc = nlp(sentence)
        comma_idx = None
        for token in doc:
            if token.text == ',':
                comma_idx = token.i
                break
        if comma_idx is None:
            return None
        for token in doc[:comma_idx]:
            if token.dep_ in ['prep', 'advmod', 'npadvmod']:
                return sentence[:doc[comma_idx + 1].idx].strip()
        return None
    
    intro_phrases = [extract_intro_phrase(sen) for sen in sentences]
    phrase_counts = Counter(intro_phrases)
    most_common_phrase, freq = phrase_counts.most_common(1)[0]
    
    if most_common_phrase and freq == len(sentences):
        return [s.replace(most_common_phrase, "").lstrip().capitalize() for s in sentences]
    else:
        return sentences


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
    model_embedding = SentenceTransformer(config['model_embedding'])

    # Load dataset
    dataset = XAIDataset(config['data']['train'], config['data']['num_samples'])
    dataloader = load_dataloader(dataset, batch_size=config['data']['batch_size'], shuffle=False)

    # -----------------------------------------MODULES-------------------------------------
    # 1. NL2FOL
    module_1 = nl_to_fol(
        base_model=llama_base,
        finetune_model = logicllama,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        tokenizer=tokenizer,
        load_in_8bit = config['module_nl2fol']['load_in_8bit'],
        max_output_len = config['module_nl2fol']['max_output_len'],
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
        if step >= 9:
            break
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
            }
# -----------------------------------------EXECUTE---------------------------------------------
            # 1. NL2FOL
            res_module_1 = module_1.generate_sample(data)
            
            # 2. EXTRACT LOGIC PROGRAM
            logic_program = extract_logic_program.generate_sample(res_module_1)
            logic_program = handle_missing_predicates(logic_program, res_module_1['LLM-FOL'])
            res_module_1['logic_program'] = logic_program
            ic(res_module_1['question-FOL'])
            # raise
            # 3. MAPPING
            clusters = clustering_lp(model_embedding, logic_program, 0.6)
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
            ic(res_module_1['question-FOL'])
            # raise
            maps = {}
            for map_dict in all_map_dict:
                for k, v in map_dict.items():
                    maps[k] = v

            res_module_1['new-fol'] = map_to_fol(
                maps=maps,
                logic_program=logic_program,
                fol=res_module_1['LLM-FOL']
            )
            ic(res_module_1['question-FOL'])
            # raise
            res_module_1['new-question-FOL'] = map_to_fol(
                maps=maps,
                logic_program=logic_program,
                fol=res_module_1['question-FOL']
            )
            res_module_1['maps'] = all_map_dict
            res_module_1['clusters'] = clusters

            # 4. SOLVER
            res_module_1['solver'] = []
            list_premises = res_module_1['new-fol']
            list_questions = res_module_1['new-question-FOL']
            for question in list_questions:
                question = [question]

                result_solver = prover9.solving_questions(list_premises, question)
                res_module_1['solver'].append(result_solver)
            # res_module_1['solver'] = result_solver
            
        
        path = f"/data/npl/ViInfographicCaps/Contest/final_contest/final_code/save/all_files/log_{step}.json"
        save_json(res_module_1, path)

        
        



if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    time_execute = end - begin
    ic(time_execute)
