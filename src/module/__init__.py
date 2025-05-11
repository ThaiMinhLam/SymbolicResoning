# from .lina.lina_module import IE, SR
from .nl2fol.nl2fol_module import nl_to_fol
from .fol2fol.extract_lp import Extract_Logic_Progam
from .fol2fol.remove_redundant_predicate import reducing
from .fol2fol.extract_lp_nguyen import extract_lp
from .solver.run_solver import Prover9_K
from .solver.prover9_solver import FOL_Prover9_Program
from .nl2fol.make_conclusion_multiple_choice import make_conclusion
from .nl2fol.make_conclusion_another import Extract_Hypothesis
from .nl2fol.preprocessing_premise import Preprocessing_PremiseNL
from .fol2fol.convert_entity_in_predicate_to_para import convert_entity