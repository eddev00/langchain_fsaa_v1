from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import time

def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0.4",
        # verbose=True
    )
    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="CNPN-Licence-Fondamentale_Professionnelle",
        embedding_function=embedding,
        persist_directory="app/model/src/data/chroma",
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        # verbose=True,
    )


def gen_answer(user_input,chat_history_input=None):
    load_dotenv()

    chain = make_chain()
    chat_history = []

    if chat_history_input:
       for chat in chat_history_input['chat history']:
              if(list(chat_history_input['chat history']).index(chat)%2==0):
               print(chat['content'])  
               chat_history.append(HumanMessage(content=chat['content']))
              else:
               chat_history.append(AIMessage(content=chat['content']))
           
    else:
       chat_history = []

    print('okey',chat_history)
    question = user_input

    # Generate answer
    response = chain({"question": question, "chat_history": chat_history})

    # Retrieve answer
    answer = response["answer"]
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    return answer,chat_history
   


