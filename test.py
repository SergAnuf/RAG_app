from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncChromiumLoader
import nest_asyncio
import asyncio

nest_asyncio.apply()

async def extract_html_data(url: str):
    # Load the page content
    loader = AsyncChromiumLoader([url])
    docs = await loader.aload()  # Use `aload()` instead of `load()` for async loading

    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(docs[0].page_content, "html.parser")

    # Extract prices and locations
    listings = []

    # Extract price from <p> tags with the class "_64if862 _194zg6t6"
    price_tags = soup.find_all("p", class_="_64if862 _194zg6t6", attrs={"data-testid": "listing-price"})
    
    # Extract address from <address> tags with the class "m6hnz62 _194zg6t9"
    address_tags = soup.find_all("address", class_="m6hnz62 _194zg6t9")

    # Pair price and address tags together
    for price_tag, address_tag in zip(price_tags, address_tags):
        price = price_tag.get_text(strip=True)
        address = address_tag.get_text(strip=True)

        listings.append({
            "property_price": price,
            "property_location": address,
        })

    return listings

async def main():
    url = "https://www.zoopla.co.uk/to-rent/property/london/?price_frequency=per_month&q=London&search_source=to-rent"
    listings = await extract_html_data(url)

    # Print the extracted listings
    for listing in listings:
        print(listing)

if __name__ == "__main__":
    asyncio.run(main())

