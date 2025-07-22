"""
LLM as a judge assessment
"""

import json

import pandas as pd
from tqdm import tqdm

from llmzmcp.data import load_llm_eval_dataframes
from llmzmcp.shared import oaiclient as client

# Load the dataframe of the LLM answers generated via api calls on the ground truth questions
gpt_4o_mini = load_llm_eval_dataframes("gpt4o-mini")


def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


###########################################################################################
# Helper prompts for evaluating the model
###########################################################################################
# Requires a user question and answer which is only available in offline mode
offline_prompt_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer compared to the original answer provided.
Based on the relevance and similarity of the generated answer to the original answer, you will classify
it as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Original Answer: {answer_orig}
Generated Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the original
answer and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()

# Works if the user answer is not available
online_prompt_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# Randomly select 5 samples just to test the pipeline. Tokens aint cheap
samples = gpt_4o_mini.sample(n=5, random_state=42).to_dict(orient="records")

offline_evaluations = []
online_evaluations = []
for record in tqdm(samples):
    # Change prompts for openai completions api
    # Offline prompt evaluation
    offl_prompt = offline_prompt_template.format(**record)
    offl_eval = llm(offl_prompt, model='gpt-4o-mini')
    json_fmt = json.loads(offl_eval)
    offline_evaluations.append(json_fmt)
    # Online prompt evaluation
    onl_prompt = online_prompt_template.format(**record)
    onl_eval = llm(onl_prompt, model='gpt-4o-mini')
    json_fmt = json.loads(onl_eval)
    online_evaluations.append(json_fmt)

df_offl_eval = pd.DataFrame(offline_evaluations)
df_onl_eval = pd.DataFrame(online_evaluations)

print("\nOffline evaluation example results")
print(df_offl_eval.Relevance.value_counts().to_frame())

print("\n\nOnline evaluation example results")
print(df_onl_eval.Relevance.value_counts().to_frame())