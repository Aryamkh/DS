A lightweight Python web scraper that extracts car listing data for "Samand" vehicles manufactured after 1385 (2006) from the Bama.ir API. 

The extracted data is automatically saved to a `.csv` file with Excel-friendly UTF-8 encoding so Persian/Farsi characters render perfectly.

## Extracted Fields
This script extracts the following data points for each car:
- Price
- Mileage
- Color
- Production Year
- Transmission Type
- Description

## Prerequisites
Make sure you have Python installed on your system. You will also need the `requests` library to handle the API calls.

⚙️ How to Scrape 100 (or a Custom Number) of Cars
By default, the script is set to scrape exactly 50 cars.

To change this to 100 (or any other number), simply open bama_scraper.py in your text editor and look near the top of the file for this line (around line 14):

Python
TARGET_COUNT = 50
Change the 50 to your desired number of cars. For example, to scrape 100 cars, update it to:

Python
TARGET_COUNT = 100
Save the file and run the script again. The scraper will automatically paginate through Bama.ir until it hits your new target!

Note on Rate Limiting: The script includes a polite delay (time.sleep(1.5)) between pages so it doesn't overwhelm the Bama.ir servers. If you set a very high custom number, the script will take a little bit of time to complete.
