import re
from nltk.inference.prover9 import *
from nltk.sem.logic import Expression, NegatedExpression
from fol_prover9_parser import Prover9_FOL_Formula
from Formula import FOL_Formula
import os

# Set Prover9 binary path
os.environ['PROVER9'] = '/data/npl/ICEK/News/Qwen_evaluate/LADR-2009-11A/bin'

class FOL_Prover9_Program:
    def __init__(self, logic_program: str, dataset_name='FOLIO') -> None:
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        self.used_idx = []
        self.dic_premises = {}
        self.flag = self.parse_logic_program()

    def parse_logic_program(self):
        try:
            premises_string = self.logic_program.split("Conclusion:")[0].split("Premises:")[1].strip()
            conclusion_string = self.logic_program.split("Conclusion:")[1].strip()

            premises = premises_string.strip().split('\n')
            conclusion = conclusion_string.strip().split('\n')

            self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]
            self.logic_conclusion = conclusion[0].split(':::')[0].strip()

            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if not fol_rule.is_valid:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            for idx, assumption in enumerate(self.prover9_premises):
                expr_str = str(Expression.fromstring(assumption))
                self.dic_premises[expr_str] = idx + 1

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if not fol_conclusion.is_valid:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True
        except Exception as e:
            print("Parsing Error:", e)
            return False

    def find_original_idx(self, premise_from_proof):
        expr_from_proof = Expression.fromstring(premise_from_proof)
        for idx, p in enumerate(self.prover9_premises):
            expr_candidate = Expression.fromstring(p)
            if expr_candidate == expr_from_proof:
                return idx + 1
        return None

    def get_used_idx(self, prover):
        # print(prover.proof())
        check = prover.proof().split('\n')
        for line in check:
            if '[assumption]' in line:
                line_parts = line.strip().split(None, 1)
                if len(line_parts) < 2:
                    continue
                expr_str = line_parts[1].split('[')[0].strip()
                if expr_str.endswith('.'):
                    expr_str = expr_str[:-1].strip()
                matched_idx = self.find_original_idx(expr_str)
                if matched_idx:
                    self.used_idx.append(matched_idx)


    def get_used_premises(self,return_idx: bool = False, unique: bool = True):
       
        if not self.used_idx:  
            return []
        idxs = self.used_idx.copy()
        if unique:
            seen = set()
            idxs = [i for i in idxs if not (i in seen or seen.add(i))]
     
        if return_idx:
            return [(i, self.logic_premises[i - 1]) for i in idxs]
        else:
            return [self.logic_premises[i - 1] for i in idxs]

    def execute_program(self):
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 10

            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()

            if result:
                self.get_used_idx(prover)
                return 'True', ''
            else:
                negated_goal = NegatedExpression(goal)
                prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
                negation_result = prover.prove()
                if negation_result:
                    self.get_used_idx(prover)
                    return 'False', ''
                else:
                    self.get_used_idx(prover)
                    return 'Uncertain', ''
        except Exception as e:
            return None, str(e)

    def answer_mapping(self, answer, World="OWA"):
        if answer == 'True':
            return 'A'
        elif answer == 'False':
            return 'B'
        elif answer == 'Unknown':
            return 'B' if World == "CWA" else 'C'
        else:
            raise Exception("Answer not recognized")


if __name__ == "__main__":
    ## ¬∀x (Movie(x) → HappyEnding(x))
    ## ∃x (Movie(x) → ¬HappyEnding(x))
    # ground-truth: True
    logic_program1 = """Premises:
    ¬∀x (Movie(x) → HappyEnding(x)) ::: Not all movie has a happy ending.
    Movie(titanic) ::: Titanic is a movie.
    ¬HappyEnding(titanic) ::: Titanic does not have a happy ending.
    Movie(lionKing) ::: Lion King is a movie.
    HappyEnding(lionKing) ::: Lion King has a happy ending.
    Conclusion:
    ∃x (Movie(x) ∧ ¬HappyEnding(x)) ::: Some movie does not have a happy ending.
    """
    logic_program_test = '''Predicates:
Quiet(x) ::: x is quiet.
Furry(x) ::: x is furry.
Green(x) ::: x is green.
Red(x) ::: x is red.
Rough(x) ::: x is rough.
White(x) ::: x is white.
Young(x) ::: x is young.
Premises:
Quiet(Anne) ::: Anne is quiet.
Furry(Erin) ::: Erin is furry.
Green(Erin) ::: Erin is green.
Furry(Fiona) ::: Fiona is furry.
Quiet(Fiona) ::: Fiona is quiet.
Red(Fiona) ::: Fiona is red.
Rough(Fiona) ::: Fiona is rough.
White(Fiona) ::: Fiona is white.
Furry(Harry) ::: Harry is furry.
Quiet(Harry) ::: Harry is quiet.
White(Harry) ::: Harry is white.
∀x (Young(x) → Furry(x)) ::: Young people are furry.
∀x (Quiet(Anne) → Red(Anne)) ::: If Anne is quiet then Anne is red. 
∀x ((Young(x) ∧ Green(x)) → Rough(x)) ::: Young, green people are rough.
∀x (Green(x) → White(x)) ::: If someone is green then they are white.
∀x ((Furry(x) ∧ Quiet(x)) → White(x)) ::: If someone is furry and quiet then they are white. 
∀x ((Young(x) ∧ White(x)) → Rough(x)) ::: If someone is young and white then they are rough.
∀x (Red(x) → Young(x)) ::: All red people are young.
Conclusion:
White(Anne) ::: Anne is white'''

    # ground-truth: True
    logic_program_D2 =    "Predicates:\nRough(x) ::: x is rough.\nWhite(x) ::: x is white.\nBlue(x) ::: x is blue.\nKind(x) ::: x is kind.\nYoung(x) ::: x is young.\nCold(x) ::: x is cold.\nPremises:\nRough(Bob) ::: Bob is rough.\n∀x (White(x) → Blue(x)) ::: All white people are blue.\n∀x (Rough(x) → Kind(x)) ::: If someone is rough then they are kind.\n∀x ((Young(x) ∧ White(x)) → Cold(x)) ::: If Bob is young and Bob is white then Bob is cold.\n∀x ((Cold(x) ∧ White(x)) → Rough(x)) ::: Cold, white people are rough.\n∀x (Kind(x) → Rough(x)) ::: All kind people are rough.\n∀x ((White(x) ∧ ¬Blue(x)) → Young(x)) ::: If someone is white and not blue then they are young.\n∀x ((Rough(x) ∧ Kind(x)) → Young(x)) ::: If someone is rough and kind then they are young.\n∀x (Young(Bob) → Rough(Bob)) ::: If Bob is young then Bob is rough.\nConclusion:\nRough(Bob) ::: Bob is rough."

    # ground-truth: True
    logic_program = """Premises:
    ∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
    ∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
    ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Conclusion:
    ((Jokes(rina) ∧ Unaware(rina)) ⊕ ¬(Jokes(rina) ∨ Unaware(rina))) → (Jokes(rina) ∧ Drinks(rina)) ::: If Rina is either a person who jokes about being addicted to caffeine and a person who is unaware that caffeine is a drug, or neither a person who jokes about being addicted to caffeine nor a person who is unaware that caffeine is a drug, then Rina jokes about being addicted to caffeine and regularly drinks coffee.
    """

    # ground-truth: Unknown
    logic_program = """Premises:
    Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
    ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    Love(miroslav, music) ::: Miroslav Venhoda loved music.
    """

    # ground-truth: True
    logic_program2 = """Premises:
    Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
    ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    ∃y ∃x (Czech(x) ∧ Author(x, y) ∧ Book(y) ∧ Publish(y, year1946)) ::: A Czech person wrote a book in 1946.
    """

    # ground-truth: False
    logic_program3 = """Premises:
    Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
    ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    ¬∃x (ChoralConductor(x) ∧ Specialize(x, renaissance)) ::: No choral conductor specialized in the performance of Renaissance.
    """

    # ground-truth: Unknown
    # Premises:\nall x.(perform_in_school_talent_shows_often(x) -> (attend_school_events(x) & very_engaged_with_school_events(x))) ::: If people perform in school talent shows often, then they attend and are very engaged with school events.\nall x.(perform_in_school_talent_shows_often(x) ^ (inactive_member(x) & disinterested_member(x))) ::: People either perform in school talent shows often or are inactive and disinterested members of their community.\nall x.(chaperone_high_school_dances(x) -> not student_attend_school(x)) ::: If people chaperone high school dances, then they are not students who attend the school.\nall x.((inactive_member(x) & disinterested_member(x)) -> chaperone_high_school_dances(x)) ::: All people who are inactive and disinterested members of their community chaperone high school dances.\nall x.((young_child(x) | teenager(x)) & wish_to_further_academic_careers(x) & wish_to_further_educational_opportunities(x) -> student_attend_school(x)) ::: All young children and teenagers who wish to further their academic careers and educational opportunities are students who attend the school.\n(attend_school_events(bonnie) & very_engaged_with_school_events(bonnie) & student_attend_school(bonnie)) ^ (not attend_school_events(bonnie) & not very_engaged_with_school_events(bonnie) & not student_attend_school(bonnie)) ::: Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school.\nConclusion:\nperform_in_school_talent_shows_often(bonnie) ::: Bonnie performs in school talent shows often."
    logic_program = """Premises:
    ∀x (TalentShows(x) → Engaged(x)) ::: If people perform in school talent shows often, then they attend and are very engaged with school events.
    ∀x (TalentShows(x) ∨ Inactive(x)) ::: People either perform in school talent shows often or are inactive and disinterested members of their community.
    ∀x (Chaperone(x) → ¬Students(x)) ::: If people chaperone high school dances, then they are not students who attend the school.
    ∀x (Inactive(x) → Chaperone(x)) ::: All people who are inactive and disinterested members of their community chaperone high school dances.
    ∀x (AcademicCareer(x) → Students(x)) ::: All young children and teenagers who wish to further their academic careers and educational opportunities are students who attend the school.
    Conclusion:
    TalentShows(bonnie) ::: Bonnie performs in school talent shows often.
    """

    # ground-truth: False
    logic_program = """Premises:
    MusicPiece(symphonyNo9) ::: Symphony No. 9 is a music piece.
    ∀x ∃z (¬Composer(x) ∨ (Write(x,z) ∧ MusicPiece(z))) ::: Composers write music pieces.
    Write(beethoven, symphonyNo9) ::: Beethoven wrote Symphony No. 9.
    Lead(beethoven, viennaMusicSociety) ∧ Orchestra(viennaMusicSociety) ::: Vienna Music Society is an orchestra and Beethoven leads the Vienna Music Society.
    ∀x ∃z (¬Orchestra(x) ∨ (Lead(z,x) ∧ Conductor(z))) ::: Orchestras are led by conductors.
    Conclusion:
    ¬Conductor(beethoven) ::: Beethoven is not a conductor."""
    
    # ground-truth: True
    logic_program4 = """Predicates:
    JapaneseCompany(x) ::: x is a Japanese game company.
    Create(x, y) ::: x created the game y.
    Top10(x) ::: x is in the Top 10 list.
    Sell(x, y) ::: x sold more than y copies.
    Premises:
    ∃x (JapaneseCompany(x) ∧ Create(x, legendOfZelda)) ::: A Japanese game company created the game the Legend of Zelda.
    ∀x ∃z (¬Top10(x) ∨ (JapaneseCompany(z) ∧ Create(z,x))) ::: All games in the Top 10 list are made by Japanese game companies.
    ∀x (Sell(x, oneMillion) → Top10(x)) ::: If a game sells more than one million copies, then it will be selected into the Top 10 list.
    Sell(legendOfZelda, oneMillion) ::: The Legend of Zelda sold more than one million copies.
    Conclusion:
    Top10(legendOfZelda) ::: The Legend of Zelda is in the Top 10 list."""

    logic_program = """Premises:
    (Like(x, c) → Love(x, c)) ::: If someone likes someone else, then they love them.
    (Love(x, c) → Like(x, c)) ::: If someone loves someone else, then they like them.
    Like(a, b) ::: Person a likes person b.

    Conclusion:
    Love(a, b) ::: Therefore, person a loves person b."""

    prover9_program = FOL_Prover9_Program(logic_program4)
    answer, error_message = prover9_program.execute_program()
    print(answer)
    print(error_message)
    # print(logic_program4)
    print(prover9_program.used_idx)