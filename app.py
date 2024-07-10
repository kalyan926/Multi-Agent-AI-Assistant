import streamlit as st
from multiagent import multiagent

input={
    "goal":"",
    "subtasks":list ,   #sub_tasks
   #"task_output":str   #output of execution
   #"task_input":str
   #"execution_state":str
   #"evaluate_state":str  #next task or rexcecute task with different context  
    "iterations":0  ,  # number of iterations executed
    "max_iterations":1 , #max iterations to execute if score is less than threshold
    "task_num":0,
    "error":any
}


st.header(":violet[Welcome to AI Assistant]")

task=st.text_input("Give me an task to perform ")

if task!=None and task!="":
    with st.spinner("Performing....."):
        input["goal"]=task
        answer=multiagent.invoke(input)
        st.write("Subtasks")
        st.write(answer["subtasks"])
        st.write(answer["task_output"])
