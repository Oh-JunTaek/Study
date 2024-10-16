from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

def create_qa_chain(ensemble_retriever):
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        Consider the intent behind the question to provide the most relevant and accurate response. 
        #Question: 
        {question} 
        #Context: 
        {context} 
        #Previous Chat History:
        {chat_history} 
        #Answer:"""
    )
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, streaming=True)
    
    multiquery_retriever = MultiQueryRetriever.from_llm(retriever=ensemble_retriever, llm=llm)
    
    rag_chain = (
        {"context": multiquery_retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
