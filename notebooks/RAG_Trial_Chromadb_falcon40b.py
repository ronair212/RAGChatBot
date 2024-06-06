#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pinecone-client


# In[2]:


#!pip install langchain


# In[3]:


#!pip install tiktoken


# In[4]:


#!pip install cohere


# In[5]:


#!pip install openai


# In[6]:


#!pip install chromadb


# In[7]:


import torch


# In[8]:


import langchain


# In[9]:


from langchain.embeddings import HuggingFaceEmbeddings


# In[10]:


from langchain.document_loaders import WebBaseLoader


urls = ["https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/interactiveandbatch.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/workingwithgpus.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/recurringjobs.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/debuggingjobs.html",
"https://rc-docs.northeastern.edu/en/latest/runningjobs/../datamanagement/index.html",
]
loader = WebBaseLoader(urls)
data = loader.load()


# In[11]:


import tiktoken
encoding_name = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# In[12]:


from langchain.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=25)
docs = text_splitter.split_documents(data)

'''
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 70,
    length_function = len,
    add_start_index = True,
)
docs = text_splitter.create_documents([data])

for idx, text in enumerate(docs):
    docs[idx].metadata['source'] = "RCDocs"
'''


# In[13]:


type(docs[0])


# In[14]:


docs[0]


# In[15]:


EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"


# In[16]:


LLM_FLAN_T5_XXL = "google/flan-t5-xxl"
LLM_FLAN_T5_XL = "google/flan-t5-xl"
LLM_FASTCHAT_T5_XL = "lmsys/fastchat-t5-3b-v1.0"
LLM_FLAN_T5_SMALL = "google/flan-t5-small"
LLM_FLAN_T5_BASE = "google/flan-t5-base"
LLM_FLAN_T5_LARGE = "google/flan-t5-large"
LLM_FALCON_7B = "tiiuae/falcon-7b-instruct"
LLM_FALCON_40b = "tiiuae/falcon-40b-instruct"


# In[17]:


cache_dir='/work/rc/projects/chatbot/models'


# In[18]:


config = {"persist_directory":None,
          "load_in_8bit":False,
          "embedding" : EMB_SBERT_MPNET_BASE,
          "llm":LLM_FALCON_40b,
          }


# In[19]:



import os
os.environ['TRANSFORMERS_CACHE'] = '/work/rc/projects/chatbot/models'
#cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/work/rc/projects/chatbot/models'


# In[20]:


'''
def create_sbert_mpnet():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})

'''

def create_sbert_mpnet():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, cache_folder=cache_dir, model_kwargs={"device": device})




#tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir="new_cache_dir/")

#model = AutoModelForMaskedLM.from_pretrained("roberta-base", cache_dir="new_cache_dir/")


# In[21]:


from transformers import AutoTokenizer
from transformers import pipeline


# In[22]:


def create_falcon_40b_instruct(load_in_8bit=False):
        model = LLM_FALCON_40b

        tokenizer = AutoTokenizer.from_pretrained(model , cache_dir=cache_dir)
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                do_sample=True,
                tokenizer = tokenizer,
                #trust_remote_code = True,
                max_new_tokens=100,
                #cache_dir=cache_dir,
                model_kwargs={
                    "device_map": "auto", 
                    "load_in_8bit": load_in_8bit, 
                    "max_length": 512, 
                    "temperature": 0.01,
                    
                    "torch_dtype":torch.bfloat16,
                    }
            )
        return hf_pipeline


# In[23]:



def create_flan_t5_base(load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
        
'''
 WARNING: You are currently loading Falcon using legacy code contained in the model repository. 
 Falcon has now been fully ported into the Hugging Face transformers library. 
 For the most up-to-date and high-performance version of the Falcon model code, 
 please update to the latest version of transformers and then load the model without the trust_remote_code=True argument.
'''


# In[24]:


if config["embedding"] == EMB_SBERT_MPNET_BASE:
    embedding = create_sbert_mpnet()


# In[25]:


load_in_8bit = config["load_in_8bit"]
if config["llm"] == LLM_FLAN_T5_BASE:
    llm = create_flan_t5_base(load_in_8bit=load_in_8bit)


# In[26]:


load_in_8bit = config["load_in_8bit"]

if config["llm"] == LLM_FALCON_40b:
    llm = create_falcon_40b_instruct(load_in_8bit=load_in_8bit)
    


# In[27]:


data = """

Introduction to OOD#
Open OnDemand (OOD) is a web portal to the Discovery cluster. A Discovery account is necessary for you to access OOD. If you need an account, see Request an account. If you already have an account, in a web browser go to http://ood.discovery.neu.edu and sign in with your Northeastern username and password.
OOD provides you with several resources for interacting with the Discovery cluster:

Launch a terminal within your web browser without needing a separate terminal program. This is an advantage if you use Windows, as otherwise, you need to download and use a separately installed program, such as MobaXterm.
Use software applications like SAS Studio that run in your browser without further configuration. See Interactive Open OnDemand Applications for more information.
View, download, copy, and delete files using the OOD File Explorer feature.

Note
OOD is a web-based application. You access it by using a web browser. Like many web-based applications, it has compatibility issues with specific web browsers. Use OOD with newer Chrome, Firefox, or Internet Explorer versions for optimal results. OOD does not currently support Safari or mobile devices (phones and tablets).

https://rc-docs.northeastern.edu/en/latest/using-ood/accessingood.html

Accessing Open OnDemand#
Open OnDemand (OOD) is a web portal to the HPC cluster.
This topic is for connecting to the HPC cluster through the browser application Open OnDemand. If you want to access the HPC directly on your system rather than through a browser, please see Connecting To Cluster, whether Mac or Windows.
A cluster account is necessary for you to access OOD. If you need an account, see Getting Access. After you have created a cluster account, access the cluster through Open OnDemand (OOD) via the following steps:

In a web browser, go to http://ood.discovery.neu.edu.
At the prompt, enter your Northeastern username and password. Note that your username is the first part of your email without the @northeastern, such as j.smith.
Press Enter or click Sign in.

Watch the following video for a short tutorial. If you do not see any controls on the video,
right-click on the video to see viewing options.

  Your browser does not support the video tag.

https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/index.html

Interactive Open OnDemand Applications#

Desktop App

OOD File Explorer

JupyterLab

The OOD web portal provides a range of applications. Upon clicking launch, the Slurm scheduler assigns a compute node with a specified number of cores and memory. By default, applications run for one hour. If you require more than an hour, you may have to wait for Slurm to allocate resources for the duration of your request.

Applications on OOD#

The Open OnDemand interface offers several applications, which as of June 2023, include:

JupyterLab
RStudio (Rocker)
Matlab
Schrodinger (Maestro)
Desktop
Gaussian (GaussView)
KNIME
TensorBoard
SAS

These applications can be accessed from the OOD web interface’s Interactive Apps drop-down menu.

Note
Specific applications in the Interactive Apps section, particularly those with graphical user interfaces (GUIs), may require X11 forwarding and the setup of passwordless SSH. For tips and troubleshooting information on X11 forwarding setup and usage, please look at the [Using X11] section of our documentation.

Additionally, we offer a selection of modified standard applications intended to support specific coursework. These applications are under the Courses menu on the OOD web interface. Please note that these course-specific applications are only accessible to students enrolled in the respective courses.

Note
Certain apps are reserved for specific research groups and are not publicly accessible, as indicated by the “Restricted” label next to the application name. If you receive an access error when attempting to open a restricted app, and you believe you should have access to it, please email rchelp@northeastern.edu with the following information: your username, research group, the app you are trying to access, and a screenshot of the error message. We will investigate and address the issue.

Go to [Open On Demand] in a web browser. If prompted, enter your MyNortheastern username and password.
Select Interactive Apps, then select the application you want to use.
Keep the default options for most apps, then click Launch. You might have to wait a minute or two for a compute node to be available for your requested time and resource.

https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/desktopood.html

Desktop App#
Open OnDemand provides a containerized desktop to run on the HPC cluster.
The following tools and programs are accessible on our Desktop App:

Slurm (for running Slurm commands via the terminal in the desktop and interacting on compute nodes)
Module command (for loading and running HPC-ready modules)
File explorer (able to traverse and view files that you have access to on the HPC)
Firefox web browser
VLC media player
LibreOffice suite of applications (word, spreadsheet, and presentation processing)

Note
The desktop application is a Singularity container; a Singularity container cannot run inside the desktop application. It fails if users run a container-based module or program via the desktop application.

https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/fileexplore.html

OOD File Explorer#
When working with the resources in OOD, your files are stored in your home directory on the storage space on the Discovery cluster. Like any file navigation system, you can work with your files and directories through the OOD Files feature, as detailed below. For example, you can download a Jupyter Notebook file in OOD that you have been working on to your local hard drive, rename a file, or delete a file you no longer need.

Note
Your home directory has a file size limit of 75GB. Please check your home directory regularly, and remove any files you do not need to make sure you have enough space.

In a web browser, go to ood.discovery.neu.edu. If prompted, enter your MyNortheastern username and password.
Select Files > Home Directory. The contents of your home directory display in a new tab.
To download a file to your hard drive, navigate to the file you want to download,
select the file, and click Download. If prompted by your browser,
click OK to save your file to your hard drive.
To navigate to another folder on the Discovery file system, click Go To,
enter the path to the folder you want to access and click OK.

Note
From the Files > Home Directory view, the Edit button will not launch your .ipynb file in a Jupyter Notebook. It will open the file in a text editor. You must be in Jupyter Notebook to launch a .ipynb file from your /home directory. See Interactive Open OnDemand Applications to access a Jupyter Notebook through OOD.

https://rc-docs.northeastern.edu/en/latest/using-ood/interactiveapps/jupyterlab.html

JupyterLab#
JupyterLab Notebook is one of the interactive apps on OOD. This section will provide a walk through of setting up and using this app. The general workflow is to create a virtual Python environment, ensure that JupyterLab Notebook uses your virtual environment, and reference this environment when you start the JupyterLab Notebook OOD interactive app.
To find the JupyterLab Notebook on OOD, follow these steps:

Go to [Open On Demand].
Click on Interactive Apps.
Select JupyterLab Notebook from the drop-down list.

The OOD form for launching JupyterLab Notebook will appear.

Conda virtual environment
You can import Python packages in your JupyterLab Notebook session by creating a conda virtual environment and activating that environment when starting a JupyterLab Notebook instance.

First, set up a virtual Python environment. See Creating Environments for how to set up a virtual Python environment on the HPC using the terminal.
Type source activate <yourenvironmentname> where <yourenvironmentname> is the name of your custom environment.
Type conda install jupyterlab -y to install JupyterLab in your environment.

Using OOD to launch JupyterLab Notebook#

Go to [Open On Demand].
Click Interactive Apps, then select JupyterLab Notebook.
Enter your Working Directory (e.g., /home/<username> or /work/<project>) that you want JupyterLab Notebook to launch in.
Select from the Partition drop-down menu the partition you want to use for your session. Refer to Partitions for the resource restrictions for the different partitions. If you need a GPU, select the gpu partition.
Select the compute node features for the job:

In the Time field, enter the number of hour(s) needed for the job.
Enter the memory you need for the job in the Memory (in Gb) field.
If you selected the gpu partition from the drop-down menu, select the GPU you would like to use and the version of CUDA that you would like to use for your session under the respective drop-down menus.

Select the Anaconda version you used to create your virtual Python environment in the System-wide Conda Module field.
Check the Custom Anaconda Environment box, and enter the name of your custom virtual Python environment in the Name of Custom Conda Environment field.
Click Launch to join the queue for a compute node. This might take a few minutes, depending on what you asked for.
When allocated a compute node, click Connect to Jupyter.

When your JupyterLab Notebook is running and open, type conda list in a cell and run the cell to confirm that the environment is your custom conda environment (you should see this on the first line). This command will also list all of your available packages.


Understanding the Queuing System#
The queuing system in a high-performance computing (HPC) environment manages and schedules computing tasks. Our HPC cluster uses the Slurm Workload Manager as our queuing system. This section aims to help you understand how the queuing system works and how to interact effectively.

Introduction to Queuing Systems#
The Slurm scheduler manages jobs in the queue. When you submit a job, it gets placed in the queue. The scheduler then assigns resources to the job when they become available, according to the job’s priority and the available resources.

Job Submission and Scheduling#
Jobs are submitted to the queue via a script specifying the resources required (e.g., number of CPUs, memory, and GPUs) and the commands to be executed. Once submitted, the queuing system schedules the job based on the resources requested, the current system load, and scheduling policies.

Scheduling Policies**#
Our cluster uses a fair-share scheduling policy. This means that usage is tracked for each user or group, and the system attempts to balance resource allocation over time. If a user or group has been using many resources, their job priority may be temporarily reduced to allow others to use the system. Conversely, users or groups that have used fewer resources will have their jobs prioritized.
The following policies ensure fair use of the cluster resources:

Single job size: The maximum number of nodes a single job depends on the partition (see Partitions).
Run time limit: The maximum run time for a job depends on the partition (see Partitions).
Priority decay: If a job remains in the queue without running for an extended period, its priority may slowly decrease.

Job Priority**#
Several factors determine job priority:

Fair-share: This is based on the historical resource usage of your group. The more resources your group has used, the lower your job’s priority becomes, and vice versa.
Job size: Smaller jobs (regarding requested nodes) typically have higher priority.
Queue wait time: The longer a job has been in the queue, the higher its priority becomes.

Job States#
Each job in the queue has a state. The main job states are:

Pending (PD): The job is waiting for resources to become available.
Running (R): The job is currently running.
Completed (CG): The job has been completed successfully.

A complete list of job states can be found in the Slurm documentation.

Monitoring the Queue**#
You can use the following commands to interact with the queue:

squeue: Displays the state of jobs or job steps. It has a wide variety of filtering, sorting, and formatting options. For example, to display your jobs:

squeue -u your_username

scontrol: Used to view and modify Slurm configuration and state. For example, to show the details of a specific job:

scontrol show job your_job_id

Tips for Efficient Queue Usage**#

Request only the resources you need: Overestimating your job’s requirements can result in longer queue times.
Break up large jobs: Large jobs tend to wait in the queue longer than small jobs. Break up large jobs into smaller ones.
Use idle resources: Sometimes, idle resources can be used. If your job is flexible regarding start time and duration, you can use the --begin and --time options to take advantage of these idle resources.

https://rc-docs.northeastern.edu/en/latest/runningjobs/jobscheduling.html

Job Scheduling Policies and Priorities#
In an HPC environment, efficient job scheduling is crucial for allocating computing resources and ensuring optimal cluster utilization. Job scheduling policies and priorities determine the order in which jobs are executed and the resources they receive. Understanding these policies is essential for maximizing job efficiency and minimizing wait times.

Scheduling Policies#

FIFO (First-In-First-Out)#
Jobs are executed in the order they are submitted. Although simple, this policy may lead to long wait times for large, resource-intensive jobs if smaller jobs are constantly being submitted.

Fair Share#
This policy ensures that all users receive a fair share of cluster resources over time. Users with high resource usage may experience reduced priority, allowing others to access resources more regularly.

Priority-Based#
Jobs are assigned priorities based on user-defined criteria or system-wide rules. Higher-priority jobs are executed before lower-priority ones, allowing for resource allocation based on user requirements.

Job Priorities#

User Priority#
Users can assign priority values to their jobs. Higher values result in increased job priority and faster access to resources.

Resource Requirements#
Jobs with larger resource requirements may be assigned higher priority, as they require more significant resources to execute efficiently.

Walltime Limit#
Jobs with shorter estimated execution times may receive higher priority, ensuring they are executed promptly and freeing up resources for other jobs.

Balancing Policies#

Backfilling#
This policy allows smaller jobs to “backfill” into available resources ahead of larger jobs, optimizing resource utilization and reducing wait times.

Preemption#
Higher-priority jobs can preempt lower-priority ones, temporarily pausing the lower-priority job’s execution to make resources available for the higher-priority job.

Best Practices#

Set Realistic Priorities: Assign accurate priorities to your jobs to reflect their importance and resource requirements.
Use Resource Quotas: Be mindful of the resources you request to prevent over- or underutilization.
Leverage Backfilling: Submit smaller, shorter jobs that can backfill into available resources while waiting for larger jobs to start.

Understanding these scheduling policies and priorities empowers you to make informed decisions when submitting jobs, ensuring that your computational tasks are executed efficiently and promptly. If you need further guidance on selecting the right scheduling policy for your job or optimizing your resource usage, our support team is available at rchelp@northeastern.edu or consult our Frequently Asked Questions (FAQs).
Optimize your job execution by maximizing our cluster’s scheduling capabilities. Happy computing!

https://rc-docs.northeastern.edu/en/latest/runningjobs/interactiveandbatch.html

Interactive and Batch Mode#
In our High-Performance Computing (HPC) environment, users can run jobs in two primary modes: Interactive and Batch. This page provides an in-depth guide to both, assisting users in selecting the appropriate mode for their specific tasks.

Interactive Mode#
Interactive mode allows users to run jobs that need immediate execution and feedback.

Getting Started with Interactive Mode#
To launch an interactive session, use the following command:
# Request an interactive session
srun --pty /bin/bash

This command allocates resources and gives you a shell prompt on the allocated node.

Interactive Mode Use Cases#

Development and Testing: Ideal for code development and testing.
Short Tasks: Best for tasks that require less time and immediate results.

See also
ADD LINK for More Examples and Guides for Interactive Mode

Batch Mode#
Batch mode enables users to write scripts that manage job execution, making it suitable for more complex or longer-running jobs.

Creating Batch Scripts#
A typical batch script includes directives for resource allocation, job names, and commands. Here is an example:
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00

# Commands to execute
module load my_program
srun my_program.exe

Save this script with a .sh extension, e.g., my_script.sh.

Submitting Batch Jobs#
You can submit your batch script using the sbatch command.
sbatch my_script.sh

Monitoring Batch Jobs#
You can monitor the status of your batch job using the squeue command.
squeue -u username

Where username is your actual username.

Use Cases#

Long-Running Jobs: Suitable for extensive simulations or calculations.
Scheduled Tasks: Execute jobs at specific times or under certain conditions.
Automated Workflows: Manage complex workflows using multiple scripts.



Transfer Data#
The HPC has a dedicated transfer node that you must use to transfer data to and from the cluster. You cannot transfer data from any other node or the HPC to your local machine. The node name is <username>@xfer.discovery.neu.edu: where <username> is your Northeastern username to login into the transfer node.
You can also transfer files using Globus. This is highly recommended if you need to transfer large amounts of data. See Using Globus for more information.
If you are transferring data from different directories on the HPC, you need to use a compute node (see Interactive Jobs: srun Command or Batch Jobs: sbatch) with SCP, rsync, or the copy command to complete these tasks. You should use the --constraint=ib flag (see Hardware Overview) to ensure the fastest data transfer rate.

Caution
The /scratch space is for temporary file storage only. It is not backed up. If you have directed your output files to /scratch, you should transfer your data from /scratch to another location as soon as possible. See Data Storage Options for more information.

Transfer via Terminal#

SCP
You can use scp to transfer files/directories to and from your local machine and the HPC. As an example, you can use this command to transfer a file to your /scratch space on the HPC from your local machine:
scp <filename> <username>@xfer.discovery.neu.edu:/scratch/<username>

where <filename> is the name of the file in your current directory you want to transfer, and <username> is your Northeastern username. So that you know, this command is run on your local machine.
If you want to transfer a directory in your /scratch called test-data from the HPC to your local machine’s current working directory, an example of that command would be:
scp -r <username>@xfer.discovery.neu.edu:/scratch/<username>/test-data .

where -r flag is for the recursive transfer because it is a directory. So that you know, this command is run on your local machine.

Rsync
You can use the rsync command to transfer data to and from the HPC and local machine. You can also use rsync to transfer data from different directories on the cluster.
The syntex of rsync is
rsync [options] <source> <destination>

An example of using rsync to transfer a directory called test-data in your current working directory on your local machine to your /scratch on the HPC is
rsync -av test-data/ <username>@xfer.discovery.neu.edu:/scratch/<username>

where this command is run on your local machine in the directory that contains test-data.
Similarly, rsync can be used to copy from the current working directory on the HPC to your current working directory on your local machine:
rsync -av <username>@xfer.discovery.neu.edu:/scratch/<username>/test-data .

where this command is run on your local machine in the current directory that you want to save the directory test-data.
You can also use rsync to copy data from different directories on the HPC:
srun --partition=short --nodes=1 --ntasks=1 --time=01:05:00 --constraint=ib --pty /bin/bash
rsync -av /scratch/<username>/source_folder /home/<username>/destination_folder

sbatch
You can use a sbatch job to complete data transfers by submitting the job to the HPC queue. An example of using rsync through a sbatch script is as follows:
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=0:05:00
#SBATCH --job-name=DataTransfer
#SBATCH --mem=2G
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH -o %j.out
#SBATCH -e %j.err

rsync -av /scratch/<username>/source_folder /home/<username>/destination_folder

where we are transferring the data from source_folder to the destination_folder.

SSHFS
If you want to use sshfs, use it with the dedicated transfer node xfer.discovery.neu.edu. It will not work on the login or compute nodes. On a Mac, you will also have to install macFUSE and sshfs (please refer to macFUSE) to use the sshfs command.
Use this syntax to perform file transfers with sshfs:
sshfs <username>@xfer.discovery.neu.edu:</your/remote/path> <your/local/path> -<options>

For example, this will mount a directory in your /scratch named test-data to a local directory on your machine ~/mount_point:
sshfs <username>@xfer.discovery.neu.edu:/scratch/<username>/test-data ~/mount_point

You can interact with the directory from your GUI or use the terminal to perform tasks on it.

Transfer via GUI Application#

OOD’s File Explorer
You can use OOD’s File Explorer application to transfer data from different directories on the HPC and also to transfer data to and from your local machine to the HPC. For more information to complete this please see OOD File Explorer.

MobaXterm
You can use MobaXterm to transfer data to and from the HPC. Please check out MobaXterm to download MobaXterm.

Open MobaXterm.
Click Session, then select SFTP.
In the Remote host field, type xfer.discovery.neu.edu
In the Username field, type your Northeastern username.
In the Port field, type 22.
In the Password box, type your Northeastern password and click OK. Click No if prompted to save your password.

You will now be connected to the transfer node and can transfer files through MobaXterm. Please refer to MobaXterm for further information.

FileZilla
You can use FileZilla to transfer data to and from the HPC. Please check out FileZilla to download.

Open FileZilla.
In the Host field, type sftp://xfer.discovery.neu.edu
In the Username field, type your Northeastern username.
In the Password field, type your Northeastern password.
In the Port field, type 22.

You will now be connected to the transfer node and can transfer files through FileZilla. Please refer to FileZilla for further information.

https://rc-docs.northeastern.edu/en/latest/datamanagement/globus.html

Using Globus#
Globus is a data management system that you can use to transfer and share files. Northeastern has a subscription to Globus, and you can set up a Globus account with your Northeastern credentials. You can link your accounts if you have another account, either personal or through another institution.
To use Globus, you will need to set up an account, as detailed below. Then, as detailed below, you will need to install Globus Connect to create an endpoint on your local computer. After completing these two initial setup procedures, you can use the Globus web app to perform file transfers. See Using the Northeastern endpoint for a walkthrough of using the Northeastern endpoint on Globus.

Globus Account Set Up#
You can use the following instructions to set up an account with Globus using your Northeastern credentials.

Go to Globus.
Click Log In.
From the Use your existing organizational login, select Northeastern University, and then click Continue.
Enter your Northeastern username and password.
If you do not have a previous Globus account, click Continue. If you have a previous account, click the Link to an existing account.
Check the agreement checkbox, and then click Continue.
Click Allow to permit Globus to access your files.

You can then access the Globus File Manager app.

Tip
If you received an account identity that includes your NUID number (for example, 000123456@northeastern.edu), you can follow the “Creating and linking a new account identity” instructions below to get a different account identity if you want a more user-friendly account identity. You can then link the two accounts together.

Creating and linking a new account identity (Optional)#
If you created an account through Northeastern University’s existing organizational login and received a username that included your NUID, you can create a new identity with a different username and link the two accounts together. A username you select instead of one with your NUID can make it easier to remember your login credentials.

Go to Globus.
Click Log In.
Click Globus ID to sign in.
Click Need a Globus ID? Sign up.
Enter your Globus ID information.
Enter the verification code that Globus sends to your email.
Click Link to an existing account to link this new account with your primary account.
Select Northeastern University from the drop-down box and click Continue to be taken to the Northeastern University single sign-on page.
Enter your Northeastern username and password.

You should now see your two accounts linked in the Account section on the Globus web app.

Install Globus Connect Personal (GCP)#
Use Globus Connect Personal (GCP) as an endpoint for your laptop. You first need to install GCP using the following procedure and be logged in to Globus before you can install GCP.

Go to Globus File Manager.
Enter a name for your endpoint in the Endpoint Display Name field.
Click Generate Setup Key to generate a setup key for your endpoint.
Click the Copy icon next to the generated setup key to copy the key to your clipboard. You will need this key during the installation of GCP in step 6.
Click the appropriate OS icon for your computer to download the installation file.
After downloading the installation file to your computer, double-click on the file to launch the installer.

Accept the defaults on the install wizard. After the installation, you can use your laptop as an endpoint within Globus.

Note
You cannot modify an endpoint after you have created it. If you need an endpoint with different options, you must delete and recreate it. Follow the instructions on the Globus website for deleting and recreating an endpoint.

Working with Globus#
After you have an account and set up a personal endpoint using Globus Connect personal, you can perform basic file management tasks using the Globus File Manager interface, such as transferring files, renaming files, and creating new folders. You can also download and use the Globus Command Line Interface (CLI) tool. Globus also has extensive documentation and training files for you to practice with.

Using the Northeastern endpoint#
To access the Northeastern endpoint on Globus, on the Globus web app, click File Manager, then in the Collection text box, type Northeastern. The endpoints owned by Northeastern University are displayed in the collection area. The general Northeastern endpoint is northeastern#discovery. Using the File Manager interface, you can easily change directories, switch the direction of transferring to and from, and specify options such as transferring only new or changed files. Below is a procedure for transferring files from Discovery to your personal computer, but with the flexibility of the File Manager interface, you can adjust the endpoints, file view, direction of the transfer, and many other options.
To transfer files from Discovery to your personal computer, do the following

Create an endpoint on your computer using the procedure above “Install Globus Connect,” if you have not done so already.
In the File Manager on the Globus web app, in the Collections textbox, type Northeastern, then in the collection list, click the northeastern#discovery endpoint.
click Transfer or Sync to in the right-pane menu.
Click in the Search text box, and then click the name of your endpoint on the Your Collections tab. You can now see the list of your files on Discovery on the left and on your personal computer on the right.
Select the file or files from the right-side list of Discovery files that you want to transfer to your personal computer.
Select the destination folder from the left-side list of the files on your computer.
(Optional) Click Transfer & Sync Options and select the transfer options you need.
Click Start.

Connecting to Google Drive#
The version of Globus currently on Discovery allows you to connect to Google Drive by first setting up the connection in GCP. This will add your Google Drive to your current personal endpoint.
Just so you know, you will first need a personal endpoint, as outlined in the procedure above. This procedure is slightly different from using the Google Drive Connector with
Globus version 5.5. You will need your Google Drive downloaded to your local computer.
To add Google Drive to your endpoint, do the following

Open the GCP app. Right-click the G icon in your taskbar on Windows and select Options. Click the G icon in the menu bar on Mac and select Preferences.
On the Access tab, click the + button to open the Choose a directory dialog box.
Navigate to your Google Drive on your computer and click Choose.
Click the Shareable checkbox to make this a shareable folder in Globus File Manager, and then click Save.

You can now go to Globus File Manager and see that your Google Drive is available as a folder on your endpoint.

Command Line Interface (CLI)#
The Globus Command Line Interface (CLI) tool allows you to access Globus from the command line. It is a stand-alone app that requires a separate download
and installation. Please refer to the Globus CLI documentation for working with this app.

Globus documentation and test files#
Globus provides detailed instructions on using Globus and has test files for you to practice with. These are free for you to access and use. We would like to encourage you to use the test files to become familiar with the Globus interface. You can access the Globus documentation and training files on the Globus How To website.



"""


# In[28]:


import tiktoken
encoding_name = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# In[29]:


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 70,
    length_function = len,
    add_start_index = True,
    #metadata={"source": "RCDocs"},
)
texts = text_splitter.create_documents([data])

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#texts = text_splitter.split_documents(text_documents)
for idx, text in enumerate(docs):
    docs[idx].metadata['source'] = "RCDocs"


# In[30]:


from langchain.vectorstores import Chroma


# In[31]:


persist_directory = config["persist_directory"]
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)


# In[32]:


from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)


# In[33]:


hf_llm = HuggingFacePipeline(pipeline=llm)
retriever = vectordb.as_retriever(search_kwargs={"k":4})
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",retriever=retriever)


# In[34]:



# Defining a default prompt for flan models
if config["llm"] == LLM_FLAN_T5_SMALL or config["llm"] == LLM_FLAN_T5_BASE or config["llm"] == LLM_FLAN_T5_LARGE or config["llm"] == LLM_FALCON_40b:
    question_t5_template = """
    context: {context}
    question: {question}
    answer: 
    """
    QUESTION_T5_PROMPT = PromptTemplate(
        template=question_t5_template, input_variables=["context", "question"]
    )
    qa.combine_documents_chain.llm_chain.prompt = QUESTION_T5_PROMPT


# In[35]:


question = "What are the stpes to install GCP?"
qa.combine_documents_chain.verbose = True
qa.return_source_documents = True


# In[36]:


import time
start_time = time.time()

results = qa({"query":question,})

end_time = time.time()
elapsed_time = end_time - start_time
print("Time take : " , elapsed_time)


# In[37]:


print("Time take : " , elapsed_time)


# In[38]:


results.keys()


# In[39]:


results['result']


# ### result from falcon 40b (time taken 837 sec)
# - Go to Globus File Manager.
# - Enter a name for your endpoint in the Endpoint Display Name field.
# - Click Generate Setup Key to generate a setup key for your endpoint.
# - Click the Copy icon next to the generated setup key to copy the key to your clipboard. You will need this key during the installation of GCP in step 6.
# - Click the appropriate OS icon for your computer to download the installation file.
# - After downloading the installation file to your'

# ### result from falcon 7b (time taken 5 sec)
# - Go to Globus File Manager.
# - Click on the G icon in the taskbar.
# - Click on the G icon in the menu bar.
# - Click on the + icon in the Choose a directory dialog box.
# - Navigate to your Google Drive on your computer and click Choose.
# - Click the Shareable checkbox to make this a shareable folder in Globus File Manager.
# - Click Save.
# 
# Now you can go to Globus File Manager and see'

# ### Install Globus Connect Personal (GCP) https://rc-docs.northeastern.edu/en/latest/datamanagement/globus.html
# 
# Use Globus Connect Personal (GCP) as an endpoint for your laptop. You first need to install GCP using the following procedure and be logged in to Globus before you can install GCP.
# 
# - Go to Globus File Manager.
# - Enter a name for your endpoint in the Endpoint Display Name field.
# - Click Generate Setup Key to generate a setup key for your endpoint.
# - Click the Copy icon next to the generated setup key to copy the key to your clipboard. You will need this key during the installation of GCP in step 6.
# - Click the appropriate OS icon for your computer to download the installation file.
# - After downloading the installation file to your computer, double-click on the file to launch the installer.

# In[ ]:




