from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor,create_openai_tools_agent
from langchain.tools import tool,Tool
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,FunctionMessage
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import PythonREPL,WikipediaAPIWrapper
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, Sequence
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

import operator

load_dotenv()



#tools for the MULTIAGENT

search=DuckDuckGoSearchRun()
search_tool=Tool(name="search",
                    func=search.run,
	            description="A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.")


exe=PythonREPL()
executer_tool=Tool(name="python_code_runner",
                func=exe.run,
                description="useful for when we to run python code and remember convert entire code to string and give as input when using the tool ")

wikipedia = WikipediaAPIWrapper()
wikipedia_tool=Tool(name="Wikipedia",
                func=wikipedia.run,
	            description="A useful tool for searching the Internet to find information on stocks,world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")

@tool
def stock_price(ticker:str):
    '''This function is used to get latest price of stock using the ticker symbol.This function takes only ticker symbol as string'''
    out=yf.Ticker(ticker)
    price=out.history(period="1d")["Close"].values[0]
    return price


#search_tool,executer_tool,wikipedia_tool,stock_price

tools=[wikipedia_tool,stock_price]


llm=ChatOpenAI(temperature=0,model="gpt-4-0125-preview")


def relevant_output(information):
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(information)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    if(len(information)==0):
        return [AIMessage(content="")]
    

    output=[]
    for agent_action, observation in information:
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=0)
        documents=text_splitter.create_documents([observation])
        split_docs=text_splitter.split_documents(documents)
        db=FAISS.from_documents(split_docs,OpenAIEmbeddings())

        docs = db.similarity_search(agent_action.tool_input)
        
        for i in range(len(docs)):
            output.append(docs[i].page_content)

        output=" ".join(output)
        print("***********************************************************")
        print(output)
        print("***********************************************************")
        
        return [FunctionMessage(name=agent_action.tool,content=output)]





class AgentState(TypedDict):
    goal:str
    subtasks:list     #sub_tasks
    task_output:str   #output of execution
    task_input:str
    execution_state:str
    evaluate_state:str  #next task or rexcecute task with different context  
    iterations:int    # number of iterations executed
    max_iterations:int   #max iterations to execute if score is less than threshold
    task_num:int
    score:float
    out_e:any
    error:any



def task_planner(state):

    prompt_p=ChatPromptTemplate.from_messages([
    ("system","""You are an perfect planner. your task is to detaily understand the goal
    ,understand the functionality of tools with its description.Think step by step about the goal
    and its requirements to divide into smaller subtasks which can accomplish the goal with tools given.
    Finally give only the answer in list of substasks without numbering format """),
    ("human","The goal:{goal} and dictionary of tools with tool name and discription:{tools}")    
    ])

    tools_detail={}
    for t in tools:
        tools_detail[f"{t.name}"]=t.description

    plan_chain=LLMChain(prompt=prompt_p,llm=llm)
    response=plan_chain.predict(goal=state["goal"],tools=tools_detail)

    state["subtasks"]=response.split("\n")

    return state  


execute_memory=[]

def task_execution_agent(state):

    if(state["execution_state"]=="feedback" or state["evaluate_state"]=="rexecute"):
        task=state["task_input"]        
    else:
        n=state["task_num"]        
        task=state["subtasks"][n]
        state["task_input"]=task

    prompt_e=ChatPromptTemplate.from_messages([
    ("system","""you are an perfect Analyst who can analyse detailly and understand the task ,previous history
    and select the tool with necessary input that can accomplish the task """),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human","task is:{task}") ,
    MessagesPlaceholder(variable_name="agent_scratchpad")
    ])



    #agent=create_openai_tools_agent(llm=llm,tools=tools,prompt=prompt_e)

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])


    agent = (
            RunnablePassthrough.assign( agent_scratchpad=lambda x: relevant_output(x["intermediate_steps"]) )
            | prompt_e
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )


    agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

    try:
        out=agent_executor.invoke({"task":task,"history":execute_memory})
        state["task_output"]=out["output"]
        state["out_e"]=out
        state["execution_state"]="evaluate"
    except Exception as e:
        state["error"]=e
        state["execution_state"]="error"

    return state

#score the output based on some objectives

score_memory=[]

def evaluate_output(state):

    system_prompt="""you are an expert critic and analyzer who analyzes the output of the task and scores the output between 1 and 10
    based on the objectives.
    give only the final answer in integer format. 
    """

    prompt_s=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","task:{task} and output:{output}")])
    
    chain_s=LLMChain(prompt=prompt_s,llm=llm)
    n=state["task_num"] 
    output=chain_s.predict(task=state["subtasks"][n],output=state["task_output"])

    #parse the output for score..........
    score=int(output)
    state["score"]=score
    threshold_score=5
    
    if((state["iterations"]>=state["max_iterations"]) or (score>threshold_score)):

        state["task_num"]+=1
        state["iterations"]=0
        state["evaluate_state"]="next"

        #append max score to the memory but below code is not max
        execute_memory.append(AIMessage(content=state["task_input"]))#ai
        execute_memory.append(HumanMessage(content=state["task_output"]))#human
        score_memory.clear()
        if((state["task_num"])==len(state["subtasks"])):
            state["evaluate_state"]="end"
            
    else:    
        state["evaluate_state"]="rexecute"
        state["iterations"]+=1

        human=state["task_input"]
        ai=state["task_output"]
        score_memory.append((human,ai,score))   
            

    return state



def feedback_agent(state):
    
    system_error="you are an expert tester who analyze the task and error and modify the task to eliminate the error"
    system_threshold="you are an expert analyzer who analyze the task ,output and score to the task and output, modify the context of the task to get desired output. finally give only modifed task"
    system_prompt=""
    user_input=""
    if(state["evaluate_state"]=="rexecute"):
        
        print("thresholddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")        
        system_prompt=system_threshold
        i=1
        for human ,ai,score in score_memory:
            user_input+=f"Trail:{i} \n ai:{ai} \n human:{human} \n score:{score} \n \n"
            i=i+1
            
    else:
        print("errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        system_prompt=system_error
        user_input=state["error"]   

    prompt_f=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","{input}") ])
    
    chain_f=LLMChain(prompt=prompt_f,llm=llm)
    output=chain_f.predict(input=user_input)

    #parse the output to subtask with context to get desired output
    state["task_input"]=output

    return state

# task planner,task_execution_agent,evaluate_output,feedback_agent



graph=StateGraph(AgentState)

graph.add_node("planner",task_planner)
graph.add_node("executor",task_execution_agent)
graph.add_node("evaluator",evaluate_output)
graph.add_node("feedback",feedback_agent)

graph.set_entry_point("planner")

graph.add_edge("planner","executor")
graph.add_edge("feedback","executor")


def execution_to_go(state):
    if state["execution_state"]=="error":        
        return "feedback"    
    else:
        return "evaluator"

def evaluate_to_go(state):
    if state["evaluate_state"]=="next":
        return "executor"
    elif state["evaluate_state"]=="end":
        return "end"
    else:
        return "feedback"


graph.add_conditional_edges("executor",execution_to_go,{"feedback":"feedback","evaluator":"evaluator"})
graph.add_conditional_edges("evaluator",evaluate_to_go,{"executor":"executor","feedback":"feedback","end":END})


multiagent=graph.compile()

'''
for output in multiagent.stream(input):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")

'''
#print(input["subtasks"])

'''


    
'''