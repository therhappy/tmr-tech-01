import json
import requests
import os
import tqdm
from typing import Any


def load_questions(file_path: str) -> list[dict[str, Any]]:
    """Load questions from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_answer_from_api(question: str,
                        api_url: str = "http://localhost:8000/conversation",
                        fail_fast: bool =True) -> str:
    """
    Send a question to the API and get the answer.
    Args:
        question (str): The question to ask.
        api_url (str): The URL of the API endpoint.
        fail_fast (bool): Whether to raise an exception on failure.
    """
    # Start a new conversation for each question
    response = requests.post(
        api_url,
        json={"message": question, "conversation_id": "new"},
    )
    
    if response.status_code != 200:
        msg = f"Error: {response.status_code} - {response.text}"
        if fail_fast:
            raise Exception(msg)
        else:
            return msg
    
    # Extract the answer from the response
    return response.json().get("response", "No answer provided")


def main():
    # Define file paths
    input_file = "../data/convfinqa_qa_eval.json"
    output_file = "../data/convfinqa_qa_eval_answers.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return
    
    # Load questions
    questions = load_questions(input_file)
    results = []
    
    # Process each question
    for i, item in tqdm.tqdm(enumerate(questions, 1), total=len(questions)):
        question = item.get("metadata").get("question")
        if not question:
            print(f"Warning: Question {i} has no 'question' field")
            continue
        
        # Get answer from API
        answer = get_answer_from_api(question)
        
        # Store result
        result = item.copy()  # Keep all original data
        result["answer"] = answer
        results.append(result)
        
        print(f"Answer received: {answer[:50]}...")
        return
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()