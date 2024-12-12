from langchain_community.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer, Html2TextTransformer
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint




load_dotenv()
# Load HTML

nest_asyncio.apply()
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")


schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}


def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)



# async def main():
#     # Load HTML
#     loader = AsyncChromiumLoader(["https://lenta.ru/"])
#     html = loader.load()

#     # Transform
#     bs_transformer = BeautifulSoupTransformer()
#     docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["span"])

#     print(docs_transformed[0].page_content[0:500])

def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span"]
    )
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    # Process the first split
    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content

async def main():
    urls = ["https://www.zoopla.co.uk/to-rent/property/london/?price_frequency=per_month&q=London&search_source=to-rent"]
    extracted_content = scrape_with_playwright(urls, schema=schema)



# async def main():
#     # Load HTML

#     urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]

#     loader = AsyncHtmlLoader(urls)

#     docs = loader.load()

#     html2text = Html2TextTransformer()
#     docs_transformed = html2text.transform_documents(docs)
#     print(docs_transformed[0].page_content[0:500])



if __name__ == "__main__":
    # Ensure compatibility for environments with an active event loop
    asyncio.run(main())