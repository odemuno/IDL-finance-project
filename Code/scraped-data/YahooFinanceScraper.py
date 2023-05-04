import yfinance as yf
import os

class YahooFinanceScraper:
    def __init__(self, ticker):
        self.ticker = ticker
    
    def get_current_price(self):
        stock = yf.Ticker(self.ticker)
        return stock.info['regularMarketPrice']
    
    def get_historical_data(self, start_date, end_date):
        stock = yf.Ticker(self.ticker)
        data = stock.history(start=start_date, end=end_date)
        return data
    
    def get_price_diff(self, start_date, end_date):
        data = self.get_historical_data(start_date, end_date)
        price_diff =  data['Close'].shift(-1) - data['Close']
        return price_diff
    
    # def save_to_csv(self, data):
    #     subfolder_path = os.path.join(self.folder_path, self.ticker)
    #     if not os.path.exists(subfolder_path):
    #         os.makedirs(subfolder_path)
        
    #     file_path = os.path.join(subfolder_path, f"{self.ticker}_data.csv")
    #     data.to_csv(file_path, index=False)

# scraper = YahooFinanceScraper('ZOOM')
# current_price = scraper.get_current_price()
# print(f"Current price of {scraper.ticker}: {current_price}")

# historical_data = scraper.get_historical_data(start_date='2020-01-01', end_date='2022-12-31')
# print(historical_data.head(10))

# price_diff = scraper.get_price_diff(start_date='2020-01-01', end_date='2022-12-31')
# print(price_diff.head(10))

TICKER_LIST = ['ZM', 'AMC']
for ticker in TICKER_LIST:
    scraper = YahooFinanceScraper(ticker)

    # make subfolder 
    subfolder_path = os.path.join(os.getcwd() + "/data/YahooFinance")
    if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    # historical price 
    historical_data = scraper.get_historical_data(start_date='2020-01-01', end_date='2022-12-31')
    historical_data.to_csv(os.path.join(subfolder_path, f"{ticker}_data.csv"))
    print(f"Extracted historical data of {ticker}:")
    print(historical_data.head(10))

    # price difference
    price_diff = scraper.get_price_diff(start_date='2020-01-01', end_date='2022-12-31')
    price_diff.to_csv(os.path.join(subfolder_path, f"{ticker}_price_diff.csv"))
    print(f"Extracted price difference data of {ticker}:")
    print(price_diff.head(10))