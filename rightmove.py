import csv
from langchain_community.document_loaders import AsyncChromiumLoader
from bs4 import BeautifulSoup
import nest_asyncio
import asyncio

nest_asyncio.apply()

async def extract_html_data(url: str):
    # Load the page content
    loader = AsyncChromiumLoader([url])
    docs = loader.load()  # This returns a list, so no `await` is needed

    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(docs[0].page_content, "html.parser")
    print(docs[0].page_content)  # For debugging

    # Extract prices and addresses from the specific tags
    listings = []

    # Extract price from <span> with class "propertyCard-priceValue"
    price_tags = soup.find_all("span", class_="propertyCard-priceValue")

    # Extract address from <address> tags with class "propertyCard-address"
    address_tags = soup.find_all("address", class_="propertyCard-address")

    # Extract property information (beds, baths) from <div> with class "property-information"
    property_info_tags = soup.find_all("div", class_="property-information")

    # Pair price, address, and property information together
    for price_tag, address_tag, info_tag in zip(price_tags, address_tags, property_info_tags):
        price = price_tag.get_text(strip=True)
        address = address_tag.get_text(strip=True)

        # Extract individual details (beds, baths)
        details = info_tag.find_all("span", class_="text")
        
        # Assign values to beds and baths with default 'NA' if missing
        beds = details[0].get_text(strip=True) if len(details) > 0 else 'NA'
        baths = details[1].get_text(strip=True) if len(details) > 1 else 'NA'

        listings.append({
            "property_price": price,
            "property_location": address,
            "beds": beds,
            "baths": baths,
        })

    return listings

async def scrape_all_pages(base_url: str, max_pages: int):
    all_listings = []

    for page_num in range(1, max_pages + 1):
        url = f"{base_url}&index={page_num * 24}"  # Update the index for pagination
        print(f"Scraping page {page_num}: {url}")

        # Extract data from the current page
        page_listings = await extract_html_data(url)

        if not page_listings:
            print("No more listings found.")
            break

        all_listings.extend(page_listings)

        # Add a polite delay between requests to avoid being flagged
        await asyncio.sleep(3)

    return all_listings

async def main():
    base_url = "https://www.rightmove.co.uk/property-to-rent/find.html?locationIdentifier=REGION%5E87490&maxPrice=25000&propertyTypes=flat&includeLetAgreed=false&mustHave=&dontShow=&furnishTypes=&keywords="
    max_pages = 10  # Set the maximum number of pages to scrape (adjust based on your needs)

    listings = await scrape_all_pages(base_url, max_pages)

    # Save the data to a CSV file
    output_file = "rightmove_listings.csv"
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["property_price", "property_location", "beds", "baths"])
        writer.writeheader()
        writer.writerows(listings)

    print(f"Data saved to {output_file}")
    print(f"Total listings scraped: {len(listings)}")

if __name__ == "__main__":
    asyncio.run(main())

