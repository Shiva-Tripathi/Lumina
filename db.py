from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
import getpass
import os
if not os.environ.get("MISTRALAI_API_KEY"):
  os.environ["MISTRALAI_API_KEY"] = getpass.getpass("lxOMZS25srGzQgZNqkSDqytjRKyGBc6T")

from langchain_mistralai import MistralAIEmbeddings

embeddings = MistralAIEmbeddings(model="mistral-embed")

from langchain_core.documents import Document
docs = [
  Document(page_content="Python is widely used in Artificial Intelligence.",metadata={"source": "AI_book"}),
  Document(page_content="Pandas is used for data analysis in python.",metadata={"source": "DataScience_book"}),
  Document(page_content="Neural networks are used in deep learning.",metadata={"source": "DL_book"}),
]


embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
  documents=docs,
  embedding= embedding_model,
  persist_directory= "chroma-db"
)


result = vectorstore.similarity_search("what is used for data analysis?",k=2)

for r in result:
  print(r.page_content)
  print(r.metadata)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
  print(d.page_content)



