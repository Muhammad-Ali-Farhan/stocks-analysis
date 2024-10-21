import requests
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
from flask import Flask, render_template
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

FINNHUB_API_KEY = "cs8mhopr01qu0vk4fci0cs8mhopr01qu0vk4fcig"





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
    }
    
    articles = fetch_news(symbol)
    
    if not articles and symbol in company_aliases:
        articles = fetch_news(company_aliases[symbol])
    
    return [{'headline': article['headline'], 'datetime': article['datetime'], 'url': article['url']} for article in articles if 'headline' in article]


def get_volume_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df['Volume']


def calculate_bollinger_bands(df, window=20, num_std=2):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['Upper Band'] = df['SMA'] + (df['Close'].rolling(window=window).std() * num_std)
    df['Lower Band'] = df['SMA'] - (df['Close'].rolling(window=window).std() * num_std)
    return df

def calculate_price_change_percentage(df):
    df['Price Change %'] = df['Close'].pct_change() * 100
    return df


def calculate_rsi(df, window=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df


server = Flask(__name__)
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')


app.layout = html.Div(
    style={
        'background-image': 'url(image.jpg)',  
        'background-size': 'cover',
        'background-position': 'center',
        'background-attachment': 'fixed',
        'min-height': '100vh',
        'padding': '20px',
    },
    children=[
        html.Div(
            style={
                'background-color': 'rgba(255, 255, 255, 0.9)',  
                'padding': '20px',
                'border-radius': '10px',
                'max-width': '1200px',
                'margin': '0 auto',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
            },
            children=[
                html.H1(
                    "Stock Market Analysis Dashboard",
                    style={'text-align': 'center', 'margin-bottom': '20px'}
                ),
                dcc.Input(
                    id='stock-input',
                    type='text',
                    placeholder='Enter stock symbol (e.g., AAPL)',
                    style={'margin-right': '10px'}
                ),
                html.Button('Fetch News', id='fetch-button', n_clicks=0),
                html.Div(id='news-results'),
                dcc.Graph(id='stock-graph'),
                dcc.Graph(id='sentiment-graph'),
                dcc.Graph(id='volume-graph'),
                dcc.Graph(id='bollinger-bands-graph'),
                dcc.Graph(id='price-change-graph'),
                dcc.Graph(id='rsi-graph'),




            ]
        )
    ]
)


app.layout = html.Div(children=[
    html.H1("Stock Market Analysis Dashboard"),
    dcc.Input(id='stock-input', value='AAPL', type='text'),
    dcc.Dropdown(
        id='moving-average',
        options=[
            {'label': '20-day SMA', 'value': 20},
            {'label': '50-day SMA', 'value': 50},
            {'label': '100-day SMA', 'value': 100}
        ],
        value=20
    ),
    dcc.Graph(id='stock-graph'),
    dcc.Graph(id='sentiment-graph'),
    dcc.Graph(id='volume-graph'),
    dcc.Graph(id='bollinger-bands-graph'),
    dcc.Graph(id='price-change-graph'),
    dcc.Graph(id='rsi-graph'),




])


@app.callback(
    Output('stock-graph', 'figure'),
    [Input('stock-input', 'value'), Input('moving-average', 'value')]
)
def update_stock_graph(symbol, window):
    if not symbol:
        return {
            'data': [],
            'layout': go.Layout(title="Please enter a stock symbol.")
        }
    
    df = get_stock_data(symbol)
    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(title="No data found for the given stock symbol.")
        }
    
    df = calculate_moving_average(df, window)

    trace_close = go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price')
    trace_sma = go.Scatter(x=df.index, y=df[f'SMA_{window}'], mode='lines', name=f'{window}-day SMA')

    return {
        'data': [trace_close, trace_sma],
        'layout': go.Layout(title=f"{symbol} Stock Price and {window}-day SMA", xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
    }


@app.callback(
    Output('sentiment-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_sentiment_graph(symbol):
    if not symbol:
        return {
            'data': [],
            'layout': go.Layout(title="Please enter a stock symbol.")
        }
    
    news_articles = get_relevant_news(symbol)
    
    if not news_articles:
        return {
            'data': [],
            'layout': go.Layout(title="No relevant news articles found")
        }

    sentiments = [perform_sentiment_analysis(article['headline']) for article in news_articles]
    short_labels = [article['headline'].split()[0] for article in news_articles]

    trace_sentiment = go.Bar(
        x=short_labels,
        y=sentiments,
        name="Sentiment Score"
    )

    return {
        'data': [trace_sentiment],
        'layout': go.Layout(
            title=f"{symbol} Sentiment Analysis",
            xaxis={'title': 'News Item', 'tickangle': -45},
            yaxis={'title': 'Sentiment Score'},
            width=1000,
            height=600
        )
    }

# Moving Average Callback
@app.callback(
    Output('volume-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_volume_graph(symbol):
    if not symbol:
        return {
            'data': [],
            'layout': go.Layout(title="Please enter a stock symbol.")
        }
    
    df = get_volume_data(symbol)
    
    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(title="No data found for the given stock symbol.")
        }
    
    trace_volume = go.Bar(
        x=df.index,
        y=df.values,
        name='Volume'
    )

    return {
        'data': [trace_volume],
        'layout': go.Layout(title=f"{symbol} Volume", xaxis={'title': 'Date'}, yaxis={'title': 'Volume'})
    }


# Bollinger Bands Callback
@app.callback(
    Output('bollinger-bands-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_bollinger_bands_graph(symbol):
    df = get_stock_data(symbol)
    df = calculate_bollinger_bands(df)
    trace_upper = go.Scatter(x=df.index, y=df['Upper Band'], mode='lines', name='Upper Band')
    trace_lower = go.Scatter(x=df.index, y=df['Lower Band'], mode='lines', name='Lower Band')
    return {'data': [trace_upper, trace_lower], 'layout': go.Layout(title='Bollinger Bands')}

import requests
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
from flask import Flask, render_template
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

FINNHUB_API_KEY = "cs8mhopr01qu0vk4fci0cs8mhopr01qu0vk4fcig"





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
    }
    
    articles = fetch_news(symbol)
    
    if not articles and symbol in company_aliases:
        articles = fetch_news(company_aliases[symbol])
    
    return [{'headline': article['headline'], 'datetime': article['datetime'], 'url': article['url']} for article in articles if 'headline' in article]


def get_volume_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df['Volume']


def calculate_bollinger_bands(df, window=20, num_std=2):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['Upper Band'] = df['SMA'] + (df['Close'].rolling(window=window).std() * num_std)
    df['Lower Band'] = df['SMA'] - (df['Close'].rolling(window=window).std() * num_std)
    return df

def calculate_price_change_percentage(df):
    df['Price Change %'] = df['Close'].pct_change() * 100
    return df


def calculate_rsi(df, window=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df


server = Flask(__name__)
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')


app.layout = html.Div(
    style={
        'background-image': 'url(image.jpg)',  
        'background-size': 'cover',
        'background-position': 'center',
        'background-attachment': 'fixed',
        'min-height': '100vh',
        'padding': '20px',
    },
    children=[
        html.Div(
            style={
                'background-color': 'rgba(255, 255, 255, 0.9)',  
                'padding': '20px',
                'border-radius': '10px',
                'max-width': '1200px',
                'margin': '0 auto',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
            },
            children=[
                html.H1(
                    "Stock Market Analysis Dashboard",
                    style={'text-align': 'center', 'margin-bottom': '20px'}
                ),
                dcc.Input(
                    id='stock-input',
                    type='text',
                    placeholder='Enter stock symbol (e.g., AAPL)',
                    style={'margin-right': '10px'}
                ),
                html.Button('Fetch News', id='fetch-button', n_clicks=0),
                html.Div(id='news-results'),
                dcc.Graph(id='stock-graph'),
                dcc.Graph(id='sentiment-graph'),
                dcc.Graph(id='volume-graph'),
                dcc.Graph(id='bollinger-bands-graph'),
                dcc.Graph(id='price-change-graph'),
                dcc.Graph(id='rsi-graph'),




            ]
        )
    ]
)


app.layout = html.Div(children=[
    html.H1("Stock Market Analysis Dashboard"),
    dcc.Input(id='stock-input', value='AAPL', type='text'),
    dcc.Dropdown(
        id='moving-average',
        options=[
            {'label': '20-day SMA', 'value': 20},
            {'label': '50-day SMA', 'value': 50},
            {'label': '100-day SMA', 'value': 100}
        ],
        value=20
    ),
    dcc.Graph(id='stock-graph'),
    dcc.Graph(id='sentiment-graph'),
    dcc.Graph(id='volume-graph'),
    dcc.Graph(id='bollinger-bands-graph'),
    dcc.Graph(id='price-change-graph'),
    dcc.Graph(id='rsi-graph'),




])


@app.callback(
    Output('stock-graph', 'figure'),
    [Input('stock-input', 'value'), Input('moving-average', 'value')]
)
def update_stock_graph(symbol, window):
    if not symbol:
        return {
            'data': [],
            'layout': go.Layout(title="Please enter a stock symbol.")
        }
    
    df = get_stock_data(symbol)
    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(title="No data found for the given stock symbol.")
        }
    
    df = calculate_moving_average(df, window)

    trace_close = go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price')
    trace_sma = go.Scatter(x=df.index, y=df[f'SMA_{window}'], mode='lines', name=f'{window}-day SMA')

    return {
        'data': [trace_close, trace_sma],
        'layout': go.Layout(title=f"{symbol} Stock Price and {window}-day SMA", xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
    }


@app.callback(
    Output('sentiment-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_sentiment_graph(symbol):
    if not symbol:
        return {
            'data': [],
            'layout': go.Layout(title="Please enter a stock symbol.")
        }
    
    news_articles = get_relevant_news(symbol)
    
    if not news_articles:
        return {
            'data': [],
            'layout': go.Layout(title="No relevant news articles found")
        }

    sentiments = [perform_sentiment_analysis(article['headline']) for article in news_articles]
    short_labels = [article['headline'].split()[0] for article in news_articles]

    trace_sentiment = go.Bar(
        x=short_labels,
        y=sentiments,
        name="Sentiment Score"
    )

    return {
        'data': [trace_sentiment],
        'layout': go.Layout(
            title=f"{symbol} Sentiment Analysis",
            xaxis={'title': 'News Item', 'tickangle': -45},
            yaxis={'title': 'Sentiment Score'},
            width=1000,
            height=600
        )
    }

# Moving Average Callback
@app.callback(
    Output('volume-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_volume_graph(symbol):
    if not symbol:
        return {
            'data': [],
            'layout': go.Layout(title="Please enter a stock symbol.")
        }
    
    df = get_volume_data(symbol)
    
    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(title="No data found for the given stock symbol.")
        }
    
    trace_volume = go.Bar(
        x=df.index,
        y=df.values,
        name='Volume'
    )

    return {
        'data': [trace_volume],
        'layout': go.Layout(title=f"{symbol} Volume", xaxis={'title': 'Date'}, yaxis={'title': 'Volume'})
    }


# Bollinger Bands Callback
@app.callback(
    Output('bollinger-bands-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_bollinger_bands_graph(symbol):
    df = get_stock_data(symbol)
    df = calculate_bollinger_bands(df)
    trace_upper = go.Scatter(x=df.index, y=df['Upper Band'], mode='lines', name='Upper Band')
    trace_lower = go.Scatter(x=df.index, y=df['Lower Band'], mode='lines', name='Lower Band')
    return {'data': [trace_upper, trace_lower], 'layout': go.Layout(title='Bollinger Bands')}


@app.callback(
    Output('price-change-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_price_change_graph(symbol):
    if not symbol:
        return {
            'data': [],
            'layout': go.Layout(title="Please enter a stock symbol.")
        }
    
    df = get_stock_data(symbol)
    df = calculate_price_change_percentage(df)

    if df.empty or 'Price Change %' not in df.columns:
        return {
            'data': [],
            'layout': go.Layout(title="No data found for the given stock symbol.")
        }
    
    trace_price_change = go.Scatter(
        x=df.index,
        y=df['Price Change %'],
        mode='lines',
        name='Price Change %'
    )

    return {
        'data': [trace_price_change],
        'layout': go.Layout(title=f"{symbol} Price Change Percentage", xaxis={'title': 'Date'}, yaxis={'title': 'Price Change %'})
    }


# RSI Callback


# RSI Callback
@app.callback(
    Output('rsi-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_rsi_graph(symbol):
    df = get_stock_data(symbol)
    df = calculate_rsi(df)
    trace = go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI')
    return {'data': [trace], 'layout': go.Layout(title='RSI')}

# MACD Callback
@app.callback(
    Output('macd-graph', 'figure'),
    [Input('stock-input', 'value')]
)
def update_macd_graph(symbol):
    df = get_stock_data(symbol)
    df = calculate_macd(df)
    trace_macd = go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD')
    trace_signal = go.Scatter(x=df.index, y=df['Signal Line'], mode='lines', name='Signal Line')
    return {'data': [trace_macd, trace_signal], 'layout': go.Layout(title='MACD')}



@server.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run_server(debug=True)


