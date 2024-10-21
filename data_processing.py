import requests
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
load_dotenv()
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

def get_stock_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

def calculate_moving_average(df, window=20):
    df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    return df

def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def fetch_news(symbol):
    today = datetime.now().strftime('%Y-%m-%d')
    last_month = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={last_month}&to={today}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching news: {response.status_code} - {response.text}")
        return []

def get_relevant_news(symbol):
    company_aliases = {
        'AAPL': 'Apple',
        'AMZN': 'Amazon',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'GOOG': 'Google',
        'META': 'Meta Platforms',
        'TSLA': 'Tesla',
        'NFLX': 'Netflix',
        'NVDA': 'Nvidia',
        'BABA': 'Alibaba',
        'V': 'Visa',
        'JNJ': 'Johnson & Johnson',
        'WMT': 'Walmart',
        'JPM': 'JPMorgan Chase',
        'PG': 'Procter & Gamble',
        'DIS': 'Disney',
        'MA': 'Mastercard',
        'HD': 'Home Depot',
        'PFE': 'Pfizer',
        'VZ': 'Verizon',
        'KO': 'Coca-Cola',
        'PEP': 'PepsiCo',
        'CSCO': 'Cisco',
        'INTC': 'Intel',
        'MRK': 'Merck',
        'NKE': 'Nike',
        'ORCL': 'Oracle',
        'T': 'AT&T',
        'UNH': 'UnitedHealth',
        'PYPL': 'PayPal',
        'COST': 'Costco',
        'ADBE': 'Adobe',
        'CRM': 'Salesforce',
        'ABT': 'Abbott Laboratories',
        'CMCSA': 'Comcast',
        'MCD': 'McDonald\'s',
        'UPS': 'UPS',
        'QCOM': 'Qualcomm',
        'AMD': 'Advanced Micro Devices',
        'IBM': 'IBM',
        'SBUX': 'Starbucks',
        'CAT': 'Caterpillar',
        'GE': 'General Electric',
        'GS': 'Goldman Sachs',
        'BA': 'Boeing',
        'LMT': 'Lockheed Martin',
        'MMM': '3M',
        'AXP': 'American Express',
        'FDX': 'FedEx',
        'SPGI': 'S&P Global',
        'RTX': 'Raytheon Technologies',
        'MDT': 'Medtronic',
        'TMO': 'Thermo Fisher Scientific',
        'DHR': 'Danaher',
        'CVX': 'Chevron',
        'XOM': 'Exxon Mobil',
        'AMAT': 'Applied Materials',
        'BKNG': 'Booking Holdings',
        'BLK': 'BlackRock',
        'SCHW': 'Charles Schwab',
        'HON': 'Honeywell',
        'DE': 'Deere & Company',
        'MO': 'Altria Group',
        'LYFT': 'Lyft',
        'UBER': 'Uber',
        'SNAP': 'Snap Inc.',
        'SQ': 'Block (Square)',
        'TWTR': 'Twitter',
        'ZM': 'Zoom Video Communications',
        'SPOT': 'Spotify',
        'ROKU': 'Roku',
        'F': 'Ford',
        'GM': 'General Motors',
        'PLTR': 'Palantir',
        'DOCU': 'DocuSign',
        'EBAY': 'eBay',
        'ATVI': 'Activision Blizzard',
        'DKNG': 'DraftKings',
        'PINS': 'Pinterest',
        'TDOC': 'Teladoc',
        'RBLX': 'Roblox',
        'ETSY': 'Etsy',
        'PTON': 'Peloton',
        'NVAX': 'Novavax',
        'CVS': 'CVS Health',
        'JD': 'JD.com',
        'BIDU': 'Baidu',
        'SPLK': 'Splunk',
        'WDC': 'Western Digital',
        'MRNA': 'Moderna'
    }

    articles = fetch_news(symbol)
    
    if not articles and symbol in company_aliases:
        articles = fetch_news(company_aliases[symbol])
    
    return [article for article in articles if 'title' in article]
