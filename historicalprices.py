import requests
from bs4 import BeautifulSoup
import re

def get_historical_prices(product_url):
    headers = {
        'User-Agent': 'Your User Agent'
    }
    response = requests.get(product_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        price_data = {}
        
        # Extract historical price data
        price_blocks = soup.find_all('div', class_='price-history__row')
        
        for block in price_blocks:
            date = block.find('span', class_='price-history__date').text.strip()
            price = block.find('span', class_='price-history__price').text.strip()
            price_data[date] = price
            
        return price_data
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

# Example usage
if __name__ == '__main__':
    product_url = 'https://camelcamelcamel.com/product/ASIN'
    historical_prices = get_historical_prices(product_url)
    
    if historical_prices:
        print("Historical Prices:")
        for date, price in historical_prices.items():
            print(f"{date}: {price}")
