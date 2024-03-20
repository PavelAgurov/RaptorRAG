"""
    List of prompt templates
"""

SUMMARY_PROMPT_TEMPLATE = """You are an assistant to create a detailed summary of the text input prodived.
Text:
{text}
"""

QUERY_PROMPT_TEMPLATE = """
Answer the question based ONLY on the following context:
<context>
{context}
</context>

Answer the following question:
<question>
{question}
</question>

Do not make up answer, use only provided information!
If there is no answer in the provided document - just tell "No answer".

Provide answer in JSON format:
{{
  "answer": "put an answer here",
  "score" : "score of relevance (0..1)"
}}
"""