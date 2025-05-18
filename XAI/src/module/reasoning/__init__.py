from .reason import create_template_explain, generate_explain, create_template_reasoning_easy_sample_v2,extract_llm_output
from .reasoning_hard import reasoning_hard
from .template import (
    PARSE_QUESTION, 
    UNDERSTAND_BACKGROUND_PROMPT, 
    UNDERSTAND_BACKGROUND_PROMPT_WITHOUT_PREMISE, 
    REDUCE_AND_MATCHING_PREDICATE_PROMPT, 
    EXTRACT_MAIN_PREDICATE,
    HEAD_INSTRUCTION,
    LOGIC_PROGRAM_EXTRACTION_PROMPTING,
    LOGIC_PROGRAM_EXTRACTION_PROMPTING_NEW,
    MAKE_CONCLUSION_FROM_OPTION_QUESTION,
    CONVERT_INDIVIDUAL_TO_PARAM,
    OPEN_QUESTION_PROMPT_EN,
)