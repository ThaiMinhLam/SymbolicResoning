import logging
import os
import subprocess
import tempfile
import argparse
import asyncio
from typing import Any, List, Dict, Optional
from pathlib import Path


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('mcp_logic')

class LogicEngine:

    def __init__(self, prover_path: str):
       """Initialize connection to Prover9"""
       self.prover_path = Path(prover_path)
       # Try both prover9.exe (Windows) and prover9 (Linux/Mac)
       self.prover_exe = self.prover_path / "prover9.exe"
       if not self.prover_exe.exists():
           # Try without .exe extension for Linux/Mac
           self.prover_exe = self.prover_path / "prover9"
           if not self.prover_exe.exists():
               raise FileNotFoundError(f"Prover9 not found at {self.prover_exe} or with .exe extension")
       
       logger.debug(f"Initialized Logic Engine with Prover9 at {self.prover_exe}")   

    def _create_input_file(self, premises: List[str], goal: str) -> Path:
        """Create a Prover9 input file"""
        content = [
            "formulas(assumptions).",
            *[p if p.endswith(".") else p + "." for p in premises],
            "end_of_list.",
            "",
            "formulas(goals).",
            goal if goal.endswith(".") else goal + ".",
            "end_of_list."
        ]
        
        input_content = '\n'.join(content)
        logger.debug(f"Created input file content:\n{input_content}")
        
        fd, path = tempfile.mkstemp(suffix='.in', text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(input_content)
        return Path(path)

    def _run_prover(self, input_path: Path, timeout: int = 60) -> Dict[str, Any]:
        """Run Prover9 directly"""
        try:
            logger.debug(f"Running Prover9 with input file: {input_path}")
            
            # Set working directory to Prover9 directory
            cwd = str(self.prover_exe.parent)
            result = subprocess.run(
                [str(self.prover_exe), "-f", str(input_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                errors='replace'   # <== thêm dòng này
            )
            
            logger.debug(f"Prover9 stdout:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"Prover9 stderr:\n{result.stderr}")
            
            if "THEOREM PROVED" in result.stdout:
                proof = result.stdout.split("PROOF =")[1].split("====")[0].strip()
                return {
                    "result": "proved",
                    "proof": proof,
                    "complete_output": result.stdout
                }
            elif "SEARCH FAILED" in result.stdout:
                return {
                    "result": "unprovable",
                    "reason": "Proof search failed",
                    "complete_output": result.stdout
                }
            elif "Fatal error" in result.stderr:
                return {
                    "result": "error",
                    "reason": "Syntax error",
                    "error": result.stderr
                }
            else:
                return {
                    "result": "error",
                    "reason": "Unexpected output",
                    "output": result.stdout,
                    "error": result.stderr
                }
        except subprocess.TimeoutExpired:
            logger.error(f"Proof search timed out after {timeout} seconds")
            return {
                "result": "timeout",
                "reason": f"Proof search exceeded {timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Prover error: {e}")
            return {
                "result": "error",
                "reason": str(e)
            }
        finally:
            try:
                input_path.unlink()  # Clean up temp file
            except:
                pass


import os
from pathlib import Path

# Set Prover9 path
os.environ['PROVER9'] = '/data/npl/ICEK/News/Qwen_evaluate/LADR-2009-11A/bin'



# Initialize
prover_path = os.environ['PROVER9']
engine = LogicEngine(prover_path=prover_path)

# Premises and goal
premises = [
        "∀x (Student(x) ∧ CompletedCoreCurriculum(x) ∧ PassedScienceAssessment(Sophia) → QualifiedForHighAchievingCourses(x))",
        "∀x (Student(x) ∧ QualifiedForHighAchievingCourses(x) ∧ CompletedResearchMethodology(x) → EligibleForGlobalExchangeProgram(x))",
        "∀x (Student(x) ∧ PassedLanguageProficiencyExam(x) → EligibleForGlobalExchangeProgram(x))",
        "∀x (Student(x) ∧ EligibleForGlobalExchangeProgram(x) ∧ CompletedCapstoneProject(x) → AwardedHonorsDiploma(x))",
        "∀x (Student(x) ∧ AwardedHonorsDiploma(x) ∧ CompletedCommunityService(x) → QualifiesForUniversityScholarship(x))",
        "∀x (Student(x) ∧ AwardedHonorsDiploma(x) ∧ ReceivedFacultyRecommendation(x) → QualifiesForUniversityScholarship(x))",
        "∀x (Student(x) ∧ AtSophiaUniversity(x) → CompletedCoreCurriculum(x))",
        "PassedScienceAssessment(Sophia)",
        "HasCompleted(Sophia, ResearchMethodologyCourse)",
        "HasCompletedCapstoneProject(Sophia)",
        "∀x (Person(x) ∧ NameSophia(x) → CompletedCommunityServiceHours(x))"
    ]

goal = "QualifiesForUniversityScholarship(Sophia)"


# Create input file and run prover
input_file = engine._create_input_file(premises, goal)
result = engine._run_prover(input_file)

# Print result
print(result)