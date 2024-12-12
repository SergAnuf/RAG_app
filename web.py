from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# Start Playwright to fetch dynamic content
def fetch_zoopla_data(url):
    with sync_playwright() as p:
        # Launch a browser instance
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to the webpage
        page.goto(url)
        page.wait_for_timeout(5000)  # Wait for 5 seconds to allow the page to load
        
        # Extract the page content
        html_content = page.content()
        browser.close()
        
    return html_content

# Parse the data using BeautifulSoup
def parse_properties(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    properties = []
    
    # Locate all property listings
    for listing in soup.find_all("div", class_="css-1itf2he-ListingCardContainer e2uk8e23"):
        # Extract price
        price_tag = listing.find("p", class_="css-18tfumg-Text eczcs4p0")
        price = price_tag.get_text(strip=True) if price_tag else "N/A"
        
        # Extract location
        location_tag = listing.find("p", class_="css-q7ifb8-Text eczcs4p0")
        location = location_tag.get_text(strip=True) if location_tag else "N/A"
        
        properties.append({"price": price, "location": location})
    
    return properties

# Main function
if __name__ == "__main__":
    url = "https://www.zoopla.co.uk/to-rent/property/london/?price_frequency=per_month&q=London&search_source=to-rent"
    
    # Fetch and parse data
    html_content = fetch_zoopla_data(url)
    properties = parse_properties(html_content)
    
    # Print extracted properties
    for prop in properties[:10]:  # Show first 10 properties
        print(f"Price: {prop['price']}, Location: {prop['location']}")
