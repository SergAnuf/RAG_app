import json
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncChromiumLoader
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()

nest_asyncio.apply()


async def extract_json_data(url: str):
    # Load the page content
    loader = AsyncChromiumLoader([url])
    docs = loader.load()

    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(docs[0].page_content, "html.parser")

    # Locate the <script> tag containing the JSON data
    script_tag = soup.find("script", id="__ZAD_TARGETING__", type="application/json")

    if not script_tag:
        print("No relevant script tag found.")
        return []

    # Parse the JSON content
    data = json.loads(script_tag.string)

    # Extract desired fields
    listings = []
    listing = {
        "property_price": data.get("price_actual", "N/A"),
        "property_location": data.get("display_address", "N/A"),
        "num_beds": data.get("num_beds", "N/A"),
        "num_baths": data.get("num_baths", "N/A"),
        "property_type": data.get("property_type", "N/A"),
        "furnished_state": data.get("furnished_state", "N/A"),
    }
    listings.append(listing)

    return listings

async def main():
    url = "https://www.zoopla.co.uk/to-rent/property/london/?price_frequency=per_month&q=London&search_source=to-rent"
    listings = await extract_json_data(url)
    
    # Print extracted listings
    for listing in listings:
        print(listing)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
