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
    
    # Extract price from <p> tags with the class "_64if862 _194zg6t6"
    price_tags = soup.find_all("p", class_="_64if862 _194zg6t6", attrs={"data-testid": "listing-price"})
    
    # Extract address from <address> tags with the class "m6hnz62 _194zg6t9"
    address_tags = soup.find_all("address", class_="m6hnz62 _194zg6t9")

    # Extract details (beds, baths, receptions, area) from <p> tags with the class "_1wickv3 _194zg6t9"
    detail_tags = soup.find_all("p", class_="_1wickv3 _194zg6t9")

    # Pair price, address, and detail tags together
    for price_tag, address_tag, detail_tag in zip(price_tags, address_tags, detail_tags):
        price = price_tag.get_text(strip=True)
        address = address_tag.get_text(strip=True)
        
        # Extract individual details
        details = detail_tag.find_all("span", class_="_1wickv4")
        
        # Assign values to beds, baths, receptions, and area with default 'NA' if missing
        beds = details[0].get_text(strip=True) if len(details) > 0 else 'NA'
        baths = details[1].get_text(strip=True) if len(details) > 1 else 'NA'
        receptions = details[2].get_text(strip=True) if len(details) > 2 else 'NA'
        area = details[3].get_text(strip=True) if len(details) > 3 else 'NA'

        listings.append({
            "property_price": price,
            "property_location": address,
            "beds": beds,
            "baths": baths,
            "receptions": receptions,
            "area_sq_ft": area,
        })

    return listings

async def scrape_all_pages(base_url: str, max_pages: int):
    all_listings = []

    for page_num in range(1, max_pages + 1):
        url = f"{base_url}&pn={page_num}"
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
    base_url = "https://www.zoopla.co.uk/to-rent/property/london/?price_frequency=per_month&q=London&search_source=to-rent"
    max_pages = 40  # Set the maximum number of pages to scrape (adjust based on your needs)

    listings = await scrape_all_pages(base_url, max_pages)

    # Save the data to a CSV file
    output_file = "zoopla_listings5.csv"
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["property_price", "property_location", "beds", "baths", "receptions", "area_sq_ft"])
        writer.writeheader()
        writer.writerows(listings)

    print(f"Data saved to {output_file}")
    print(f"Total listings scraped: {len(listings)}")

if __name__ == "__main__":
    asyncio.run(main())
