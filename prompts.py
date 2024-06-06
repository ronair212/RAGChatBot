
question_summarizer_prompt =''' 
[GENERATE QUESTION]

***Conversation Summary***
Chat History:
{chat_history}

User's New Question:
{question}

Given the Chat History and a User's New Question, rephrase the new question to be a standalone question. '''



answer_generator_prompt = '''
[INST]\n
***Instructions***
- Respond with factual information only from the provided context.
- Do not give any details apart from what is given in the context.
- If there's insufficient context, mention you don't have enough information.
- Always include a "SOURCES" section from the context in the response. 

***Details***
Context:
---------------------
{summaries}
---------------------

Chat History:
{chat_history}

Question: {question}
\n[\INST]\n\n

Answer:
'''
