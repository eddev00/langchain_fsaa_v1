from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv


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
    
    if chat_history_input:
       chat_history= chat_history_input
    else:
       chat_history = []

    
    
    question = user_input

    # Generate answer
    response = chain({"question": question, "chat_history": chat_history})

    # Retrieve answer
    answer = response["answer"]
    source = response["source_documents"]
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
     
    # Display answer
    #print("\n\nSources:\n")
    #for document in source:
    #    print(f"Page: {document.metadata['page_number']}")
    #    print(f"Text chunk: {document.page_content[:160]}...\n")
    #print(f"Answer: {answer}")
    return response,chat_history
   