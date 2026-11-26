## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Develop a multidocument retrieval agent using LlamaIndex that can search multiple research papers, retrieve relevant information, and generate accurate, concise, and meaningful answers to user queries. The agent should also be able to compare and combine information from different documents.

### DESIGN STEPS:

#### STEP 1:

Load the research papers using LlamaIndex.
Create vector and summary tools for each paper.

#### STEP 2:

Combine the document tools.
Initialize the OpenAI language model.

#### STEP 3:
Create the FunctionCallingAgentWorker.
Create the AgentRunner.


#### step 4:
Give natural-language queries to the agent.
Retrieve relevant information from the papers.
Generate concise answers using the retrieved information.
Test the agent with different queries.
Evaluate the responses based on accuracy, relevance, and conciseness.




### PROGRAM:
```

from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio

nest_asyncio.apply()
from utils import get_doc_tools
from pathlib import Path

papers = [
    "lion.pdf",
    "CGD.pdf",
    "study.pdf"
]

paper_to_tools_dict = {}

for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(
        paper,
        Path(paper).stem
    )
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [
    t for paper in papers
    for t in paper_to_tools_dict[paper]
]

print("Number of tools:", len(initial_tools))
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools,
    llm=llm,
    verbose=True
)

agent = AgentRunner(agent_worker)
response = agent.query(
    "What is the main objective and proposed approach of the Lion research paper?"
)
print(str(response))
response = agent.query(
    "What problem does CGD address, what method is proposed, and what are its main results?"
)
print(str(response))
response = agent.query(
    "What is the main objective, methodology, and conclusion of the study paper?"
)
print(str(response))
response = agent.query(
    "Compare the objectives and approaches of the Lion and CGD papers."
)
print(str(response))
response = agent.query(
    "Compare the objectives, methods, and results of Lion, CGD, and Study."
)
print(str(response))
```

### OUTPUT:
<img width="820" height="595" alt="image" src="https://github.com/user-attachments/assets/efae2ec9-b565-4d49-a13d-24cb5a7c99df" />


<img width="807" height="555" alt="image" src="https://github.com/user-attachments/assets/ee9c4b92-1f12-4346-beff-414bc0b5e7f0" />

<img width="815" height="492" alt="image" src="https://github.com/user-attachments/assets/853829cf-6ad7-46b3-9f2f-6280ed429cbe" />

<img width="777" height="612" alt="image" src="https://github.com/user-attachments/assets/70fb4f99-0ec9-4803-8c81-3045784f62de" />


<img width="787" height="635" alt="image" src="https://github.com/user-attachments/assets/bb6b419a-b7f8-4375-a03c-58a637a70ed5" />

<img width="790" height="622" alt="image" src="https://github.com/user-attachments/assets/f1c3af03-ee57-47ea-b9df-02b4d5b7261a" />




### RESULT:

The multidocument retrieval agent was successfully implemented using LlamaIndex. It retrieved relevant information from the research papers and generated concise and relevant responses to different queries. The system demonstrated effective document retrieval and information synthesis.
