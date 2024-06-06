import openai

openai.api_key = "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"

def generate_three_qa_from_chunk(chunk):
    # Generate three questions based on the chunk
    questions_response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"Given the context of research computing at Northeastern University, and based on the following text: \"{chunk}\", generate three distinct questions related to research computing:",
        max_tokens=300,  # Adjust token count if needed
        temperature=0.7
    )
    questions = questions_response.choices[0].text.strip().split("\n")[:3]

    # Generate answers for the three questions
    answers = []
    for question in questions:
        answer_response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Answer the following question based on the text: \"{chunk}\". Question: {question}",
            max_tokens=500,
            temperature=0.7
        )
        answers.append(answer_response.choices[0].text.strip())

    return questions, answers

def generate_qa_new(chunk):
    # Single API call for both question and answer generation
    qa_response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"Given the context of research computing at Northeastern University, which includes topics like high-performance computing, cloud computing, data analysis, software development, etc., and the following text: \"{chunk}\", generate a relevant question and provide an answer to that question. Please focus only on the parts of the text that relate directly to research computing and ignore any off-topic or unrelated sections.",
        max_tokens=800, # Increased to accommodate both Q&A
        temperature=0.3  # Adjust based on desired randomness
    )
    

    #print(qa_response)
    response_text = qa_response.choices[0].text.strip()

    if "Question:" in response_text and "Answer:" in response_text:
        question = response_text.split("Answer:")[0].replace("Question:", "").strip()
        answer = response_text.split("Answer:")[1].strip()
    elif "Q:" in response_text and "A:" in response_text:
        question = response_text.split("A:")[0].replace("Q:", "").strip()
        answer = response_text.split("A:")[1].strip()
    else:
        print("Unexpected Q&A format in the response.")
        question = "Error"
        answer = "Error" 

    return question, answer

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter

urls = ["https://rc-docs.northeastern.edu/en/latest/welcome/index.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/welcome.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/services.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/gettinghelp.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/introtocluster.html",
"https://rc-docs.northeastern.edu/en/latest/welcome/casestudiesandtestimonials.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/index.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/get_access.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/accountmanager.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/connectingtocluster/index.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/connectingtocluster/mac.html",
"https://rc-docs.northeastern.edu/en/latest/gettingstarted/connectingtocluster/windows.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/index.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/passwordlessssh.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/shellenvironment.html",
"https://rc-docs.northeastern.edu/en/latest/first_steps/usingbash.html",
"https://rc-docs.northeastern.edu/en/latest/hardware/index.html",
"https://rc-docs.northeastern.edu/en/latest/hardware/hardware_overview.html",
"https://rc-docs.northeastern.edu/en/latest/hardware/partitions.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/index.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/introduction.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/accessingood.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/index.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/desktopood.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/fileexplore.html",
"https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/jupyterlab.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/index.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/workingwithgpus.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/recurringjobs.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/debuggingjobs.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/index.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/discovery_storage.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/transferringdata.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/globus.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/databackup.html",
"https://rc-docs.northeastern.edu/en/latest/datamanagement/securityandcompliance.html",
"https://rc-docs.northeastern.edu/en/latest/software/index.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/index.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/modules.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/mpi.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/r.html",
"https://rc-docs.northeastern.edu/en/latest/software/systemwide/matlab.html",
"https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/index.html",
"https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/conda.html",
"https://rc-docs.northeastern.edu/en/latest/software/packagemanagers/spack.html",
"https://rc-docs.northeastern.edu/en/latest/software/fromsource/index.html",
"https://rc-docs.northeastern.edu/en/latest/software/fromsource/makefile.html",
"https://rc-docs.northeastern.edu/en/latest/software/fromsource/cmake.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/index.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/introductiontoslurm.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmcommands.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmrunningjobs.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmmonitoringandmanaging.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmscripts.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmarray.html",
"https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmbestpractices.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/index.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/class_use.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/cps_ood.html",
"https://rc-docs.northeastern.edu/en/latest/classroom/classroomexamples.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/index.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/homequota.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/checkpointing.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/optimizingperformance.html",
"https://rc-docs.northeastern.edu/en/latest/best-practices/software.html",
"https://rc-docs.northeastern.edu/en/latest/tutorialsandtraining/index.html",
"https://rc-docs.northeastern.edu/en/latest/tutorialsandtraining/canvasandgithub.html",
"https://rc-docs.northeastern.edu/en/latest/faq.html",
"https://rc-docs.northeastern.edu/en/latest/glossary.html",
]
loader = WebBaseLoader(urls)
data = loader.load()



text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=25)
docs = text_splitter.split_documents(data)

import re
for i in range(len(docs)):
    docs[i] = re.sub(r'(\\n)+', ' ', str(docs[i]))


    
import re

def preprocess_text(text):
    # Remove URL and other metadata
    text = re.sub(r"metadata=\{.*?\}", "", text)
    
    # Remove navigation links and page control commands
    text = re.sub(r"Toggle (child pages in navigation|Light \/ Dark \/ Auto color theme|table of contents sidebar)", "", text)
    
    # Remove special characters or unwanted sequences
    text = text.replace("\\xa0", " ")
    
    # Remove additional spaces
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"page_content='(.*?)'", r"\1", text)
    return text






processed_docs = [preprocess_text(text) for text in docs]




def generate_three_qa_from_chunk(chunk):
    # Generate three questions based on the chunk
    questions_response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"Given the context of research computing at Northeastern University, and based on the following text: \"{chunk}\", generate three distinct questions related to research computing:",
        max_tokens=300,  # Adjust token count if needed
        temperature=0.7
    )
    questions = questions_response.choices[0].text.strip().split("\n")[:3]

    # For each question, generate an answer and yield the pair
    for question in questions:
        answer_response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Answer the following question based on the text: \"{chunk}\". Question: {question}",
            max_tokens=500,
            temperature=0.7
        )
        answer = answer_response.choices[0].text.strip()
        #print(question , "\n"  , answer)
        yield question, answer

        
        
import pandas as pd

def initialize_dataframe(excel_path):
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Question', 'Answer'])
    return df

def save_to_excel(df, question, answer, excel_path):
    new_row = pd.DataFrame({'Question': [question], 'Answer': [answer]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(excel_path, index=False)
    return df

excel_path = 'qa_pairs_from_RCDocs.xlsx'
df = initialize_dataframe(excel_path)


for i, chunk in enumerate(processed_docs[:255]):
    print(i)
    if chunk.strip() == "":
        continue
    for question, answer in generate_three_qa_from_chunk(chunk):
        df = save_to_excel(df, question, answer, excel_path)
        #print(f"Question: {question}")
        #print(f"Answer: {answer}\n")





        
        


