import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()
chave_api = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=chave_api)
PASTA_DOCS = os.getenv("PASTA_DOCS")
docs = []
for nome in os.listdir(PASTA_DOCS):
    if nome.endswith(".txt"):
        caminho = os.path.join(PASTA_DOCS, nome)
        loader = TextLoader(caminho, encoding="utf-8")
        docs.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_divididos = splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=chave_api,
    model="models/embedding-001"
)
db = FAISS.from_documents(docs_divididos, embeddings)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=chave_api
)
llm_juiz = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=chave_api
)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)
rag_chain_juiz = RetrievalQA.from_chain_type(
    llm=llm_juiz,
    retriever=db.as_retriever(),
    return_source_documents=True
)


def responder_pergunta(pergunta: str) -> str:
    resposta = rag_chain(pergunta)
    return resposta["result"]


def mandar_pro_tribunal(pergunta: str, resposta: str) -> str:
    prompt = f"""
    Você é um avaliador imparcial. Sua tarefa é revisar a resposta de um tutor de IA para uma pergunta de aluno.

    Critérios:
    - A resposta está tecnicamente correta?
    - Está clara para o nível médio técnico?
    - O próximo passo sugerido está bem formulado?
    
    Se a resposta for boa, diga “✅ Aprovado” e explique por quê.
    Se tiver problemas, diga “⚠️ Reprovado” e proponha uma versão melhorada.
    
    "Pergunta do aluno: {pergunta}\n\nResposta do tutor: {resposta}"
    """
    resposta = rag_chain_juiz(prompt)
    return resposta["result"]
