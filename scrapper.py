from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import time

def scrape_dynamic_intershop():
    base_url = "https://www.intershop.com"
    customers_page = f"{base_url}/en/customers"
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page()
        
        print(f"Opening {customers_page}...")
        page.goto(customers_page)

        # Handle the "View More" button loop
        while True:
            try:
                # Target the 'View more' button specifically
                view_more = page.locator("text='View more'") 
                
                if view_more.is_visible():
                    print("Clicking 'View More' to reveal more customers...")
                    view_more.click()
                    # Short sleep to let the AJAX content load
                    page.wait_for_timeout(2000) 
                else:
                    break
            except Exception:
                break 

        # Capture the fully rendered HTML
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')
        browser.close()

        customer_data = []
        # Finding the text containers
        containers = soup.find_all('div', class_='customer-list-text')

        for container in containers:
            # 1. Extract Customer Name from the sibling/parent image
            parent = container.find_parent()
            img_tag = parent.find('img') if parent else None
            customer_name = img_tag.get('alt', 'Unknown') if img_tag else "Unknown"

            # 2. Extract Challenge (Header) and Description
            header = container.find('span', class_='customer-list-header').get_text(strip=True)
            description = container.find('div', class_='customer-list-description').get_text(strip=True)

            # 3. Extract the 'Read More' Link
            # Usually found in 'customer-list-more' or wrapped around the whole card
            link_element = container.find('a') 
            if not link_element and parent.name == 'a':
                link_element = parent
            elif not link_element:
                link_element = container.find_next('a')

            relative_link = link_element.get('href', '') if link_element else ""
            
            # Clean up the link
            if relative_link.startswith('/'):
                full_link = f"{base_url}{relative_link}"
            else:
                full_link = relative_link

            customer_data.append({
                "customer": customer_name,
                "challenge": header,
                "description": description,
                "case_study_url": full_link
            })

        # Step 2: Save to the structured JSON file
        with open('references.json', 'w', encoding='utf-8') as f:
            json.dump(customer_data, f, indent=4, ensure_ascii=False)
        
        print(f"Success! {len(customer_data)} customers saved to references.json")

if __name__ == "__main__":
    scrape_dynamic_intershop()