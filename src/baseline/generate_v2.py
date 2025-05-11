from transformers import GenerationConfig, LlamaForCausalLM
import torch
from utils import DataPreparer, all_exists
from typing import Dict, Optional, Callable, Union, List, Tuple
from functools import partial
from torch.nn.utils.rnn import pad_sequence

def llama_batch_generate(
    llama_model: LlamaForCausalLM,
    data_preparer: DataPreparer,
    input_str: Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]],
    max_new_tokens: int,
    generation_config: GenerationConfig,
    prompt_keys: Optional[Dict[str, str]] = None,
    prepare_input: Optional[Callable] = None,
    return_tensors: bool = False,
    rlhf_mode: bool = False
) -> Union[Tuple[str, Dict], Tuple[List[str], List[Dict]]]:
    """
    Generate responses from the LLM, supporting single or batch inputs.

    Args:
        input_str: a single NL (str or dict) or a list of NLs (each a str or dict).
        Other args as before.

    Returns:
        If single input: full_resp_str, resp_parts
        If batch input: list of full_resp_str, list of resp_parts
    """
    assert not all_exists(prompt_keys, prepare_input), \
        'either give me the prompt_keys or the pre-compiled prepare input func'

    # Build the prepare function
    if all_exists(prompt_keys):
        prepare_fn = partial(data_preparer.prepare_input, **prompt_keys)
    elif all_exists(prepare_input):
        prepare_fn = prepare_input
    else:
        raise ValueError('either give me the prompt_keys or the pre-compiled prepare input func')

    # Normalize to list of examples
    if isinstance(input_str, list):
        examples = input_str
        is_batch = True
    else:
        examples = [input_str]
        is_batch = False

    # Prepare inputs per example
    all_inputs = [prepare_fn(ex) for ex in examples]
    # Instead of torch.cat, pad all inputs to the same length:
    ids = [inp['input_ids'].view(-1) for inp in all_inputs]
    input_ids = pad_sequence(
        ids,
        batch_first=True,
        padding_value=data_preparer.tokenizer.pad_token_id
    ).to('cuda')

    model_kwargs = {'input_ids': input_ids}

    if 'attention_mask' in all_inputs[0]:
        masks = [inp['attention_mask'].view(-1) for inp in all_inputs]
        attention_mask = pad_sequence(
            masks,
            batch_first=True,
            padding_value=0
        ).to('cuda')
        model_kwargs['attention_mask'] = attention_mask

    # Generation
    llama_model.eval()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            gen_out = llama_model.generate(
                **model_kwargs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens
            )
    llama_model.train()

    # Decode sequences
    outputs: List[str] = []
    resp_parts_list: List[Dict] = []
    for s in gen_out.sequences:
        if rlhf_mode:
            text = data_preparer.tokenizer.decode(s)
            parts = text.split(data_preparer.tokenizer.eos_token)
            main_str = parts[0]
            _, resp_parts = data_preparer.get_response(main_str)
            full_resp_str, _ = data_preparer.get_response(text)
        else:
            text = data_preparer.tokenizer.decode(s, skip_special_tokens=True)
            full_resp_str, resp_parts = data_preparer.get_response(text)
        outputs.append(full_resp_str)
        resp_parts_list.append(resp_parts)

    # Return according to input type
    if return_tensors:
        tensor_results = data_preparer.tokenizer(outputs, return_tensors='pt', padding=True, truncation=True)
        return outputs, resp_parts_list, input_ids, tensor_results
    if is_batch:
        return outputs, resp_parts_list
    return outputs[0], resp_parts_list[0]