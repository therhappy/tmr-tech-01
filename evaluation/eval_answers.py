import json
import tqdm

# When run as a script, ensure the modules from app can be imported
import sys
import pathlib
# Add the parent directory to sys.path
sys.path.append(str(pathlib.Path(__file__).parent.parent))
    
from evaluation.eval_utils import (extract_components,
                                   ajust_number_magnitude,
                                   contains_uncertainty_marker,
                                   get_arbiter_answer)


def evaluate_answer(given_answer, expected_answer):
    """
    Evaluate whether a given answer matches the expected answer using local code.
    The evaluation checks iteratively for:
    1. Empty answer
    2. Exact match between expected answer and given answer
    3. Numbers match (reasonably good answer)
    The residuals from this approach are marked to be evaluated by an arbiter.
    
    Args:
        given_answer (str): The answer provided by the model
        expected_answer (str): The expected correct answer
    
    Returns:
        dict: A dictionary containing evaluation results with keys:
            'is_empty': True if given answer is empty
            'is_exact_match': True if answers match exactly
            'is_number_match': True if the numerical values match
            'needs_arbiter': True if arbiter judgment is needed
    """
    # Initialize result dictionary
    result = {
        'is_empty': False,
        'is_exact_match': False,
        'is_number_match': False,
        'is_number_close': False,
        'needs_arbiter': False,
        'arbiter_ruling': None
    }
    
    # Convert to lowercase for case-insensitive comparison
    given = given_answer.lower()
    expected = expected_answer.lower()
    
    # Check for empty answers
    if len(given) == 0:
        result['is_empty'] = True
        return result
    
    # Check for exact match
    if given == expected:
        result['is_exact_match'] = True
        result['is_number_match'] = True
        return result
    
    # Extract the number and other parts
    giv_number, giv_other = extract_components(given)
    exp_number, exp_other = extract_components(expected)
    
    # Check for number match
    if giv_number and exp_number:
        if giv_number == exp_number:
            result['is_number_match'] = True
            return result
        
        # Adjust for magnitude words
        giv_number_adjusted = ajust_number_magnitude(giv_number, giv_other)
        exp_number_adjusted = ajust_number_magnitude(exp_number, exp_other)
        
        # Check if the numbers are close enough (within 1%) to account for a true match
        if abs(giv_number_adjusted - exp_number_adjusted) < 0.01 * exp_number_adjusted:
            result['is_number_match'] = True
            return result
        # A 10% difference is considered an approximate match
        if abs(giv_number_adjusted - exp_number_adjusted) < 0.1 * exp_number_adjusted:
            result['is_number_close'] = True
            return result
    
    # If we got here, it's neither an exact match nor a number match
    # Need arbiter to determine if it's acceptable
    result['needs_arbiter'] = True
    return result



def main():
    
    # Path to find the questions-answers pairs
    qas_filepath = "../data/convfinqa_qa_eval_answers.json"

    qas = json.load(open(qas_filepath, "r"))
    
    # Prepare an output file and a report
    output_filepath = "../data/convfinqa_qa_eval_answers_rated.json"
    report_filepath = "../data/convfinqa_qa_eval_answers_report.json"
    rated_qas = []
    report = {
        "total questions": len(qas),
        "questions not answered": 0,
        "perfect answers": 0,
        "acceptable answers": 0,
        "wrong answers": 0,
        "admits uncertainty": 0,
    }

    for _, record in tqdm.tqdm(enumerate(qas)):
        expected_answer = record["metadata"]["answer"].lower()
        given_answer = record["answer"].lower()
        
        # Evaluate the answer locally
        evals = evaluate_answer(given_answer, expected_answer)

        # If answer is not empty and not a clear match, use a LLM arbiter
        # with the same toolkit as the app to get a final say on the answer
        # but provide the needed context"       
        if evals["needs_arbiter"]:
            # Get the arbiter to rule on the answer correctness
            question = record["metadata"]["question"]
            context = record["metadata"]["gold_content"]
            arbiter_answer = get_arbiter_answer(
                question=question,
                context=context,
                expected_answer=expected_answer,
                given_answer=given_answer,
            )
            evals["arbiter_ruling"] = arbiter_answer

            # Note if there was an admission of uncertainty
            if contains_uncertainty_marker(given_answer):
                record["admits uncertainty"] +=1

        # Update the report based on the evaluation
        record["evaluation"] = evals
        rated_qas.append(record)
        if evals["is_empty"]:
            report["questions not answered"] += 1
        elif evals["is_exact_match"] or evals["is_number_match"]:
            report["perfect answers"] += 1
        elif evals["is_number_close"] or evals["arbiter_ruling"]:
            report["acceptable answers"] += 1
        else:
            report["wrong answers"] += 1
        
    # Save the rated questions and the report
    with open(output_filepath, "w") as f:
        json.dump(rated_qas, f, indent=4)
    with open(report_filepath, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Evaluation report saved to {report_filepath}")


if __name__ == "__main__":
    main()
