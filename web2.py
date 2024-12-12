from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint

# Load environment variables
load_dotenv()
nest_asyncio.apply()

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Define the schema for extraction
schema = {
    "properties": {
        "property_price": {
            "type": "string",
            "description": "The price of the property, usually in the format £x,xxx pcm or £xxx pw."
        },
        "property_location": {
            "type": "string",
            "description": "The location of the property, including area or address details."
        },
    },
    "required": ["property_price", "property_location"],
}


# Define extraction chain
def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).invoke(content)

# Define asynchronous main function
async def main():
    urls = [
        "https://www.zoopla.co.uk/to-rent/property/london/?price_frequency=per_month&q=London&search_source=to-rent#listing_68955045"
    ]

    # Load HTML content from URLs
    loader = AsyncChromiumLoader(urls)
    docs =  loader.load()

    # Transform HTML content using BeautifulSoupTransformer
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span","div", "p"]
    )
    print("Extracting content with LLM")

    # Split documents into manageable chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    splits = splitter.split_documents(docs_transformed)

    # Process each split (for demonstration, process the first split)
    extracted_content = extract(content=splits[0].page_content, schema=schema)

    print("number of splits",len(splits))

    # Print the extracted content
    pprint.pprint(extracted_content)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
