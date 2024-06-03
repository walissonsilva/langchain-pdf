from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader

load_dotenv()

loader = PDFMinerLoader(file_path="/Users/walisson/Downloads/edital-b3.pdf")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

template = """""
    Você é um assistente virtual de uma escola de programação. Sua função será responder dúvidas que nós recebemos que estão relacionados
    aos programas que oferecemos. Esses programas se referenrem a processos seletivos que oferecem bolsas de estudo para os candidatos que se inscreverem
    e forem aprovados em todas as etapas.

    Os candidatos interessados nesses programas podem entrar em contato para tirar dúvidas e você deve respondê-los com base nas informações que estão contidas no edital do programa e te serão fornecidas logo abaixo.

    {knowledge}

    Aqui está uma pergunta enviada por um dos nossos candidatos: {question}

    Escreva a melhor resposta que você acha que seria adequada para responder a pergunta do candidato.
"""

prompt = PromptTemplate(
    input_variables=["knowledge", "question"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)

# message = "Quais são os requisitos para participar do programa de bolsas de estudo?"
# message = "Como eu faço para me inscrever nesse processo seletivo? Quando se encerram as inscrições?"
# message = (
#     "Atualmente eu moro em João Pessoa. Eu posso participar desse processo seletivo?"
# )
# message = "O que acontece se eu passar nesse processo seletivo?"


def generate_response(message):
    knowledge = retrieve_info(message)
    response = chain.run(knowledge=knowledge, question=message)
    return response


print("\n\n\n")
print(generate_response(message))
