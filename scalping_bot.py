from dotenv import load_dotenv
import os
import MetaTrader5 as mt5
import telebot
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import joblib
import time
import datetime

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EXNESS_ACCOUNT = os.getenv("EXNESS_ACCOUNT")
EXNESS_PASSWORD = os.getenv("EXNESS_PASSWORD")
EXNESS_SERVER = os.getenv("EXNESS_SERVER")

# Check if all required environment variables are set
if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, EXNESS_ACCOUNT, EXNESS_PASSWORD, EXNESS_SERVER]):
    print("❌ Missing environment variables. Please check your .env file.")
    exit()

class ScalpingBot:
    def __init__(self, mode='backtest', account_type='cent'):
        """Initialize the ScalpingBot with trading parameters."""
        self.mt5 = mt5
        self.telegram_bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
        self.mode = mode
        self.account_type = account_type
        
        # Use 'm' suffix symbols for both account types since we're on Exness Trial
        self.symbols = ["EURUSDm", "GBPUSDm", "BTCUSDm", "ETHUSDm", "XAUUSDm"]
        
        # Account settings based on account type
        if account_type == 'cent':
            self.initial_balance = 150  # 150 cents (1.5 USD)
            self.min_lot = 0.01
            self.max_lot = 10.0
        else:  # standard account
            self.initial_balance = 100  # 100 USD
            self.min_lot = 0.01
            self.max_lot = 5.0
        
        # Timeframes
        self.timeframe_trade = mt5.TIMEFRAME_M5  # 5-minute
        self.timeframe_confirm1 = mt5.TIMEFRAME_M15  # 15-minute
        self.timeframe_confirm2 = mt5.TIMEFRAME_H1  # 1-hour
        
        # Risk management
        self.risk_per_trade_initial = 0.01  # 1% risk initially
        self.risk_per_trade_aggressive = 0.05  # 5% risk after securing capital
        self.secure_threshold = 2.0  # Double initial balance
        # Machine learning components
        self.model = None
        self.scaler = StandardScaler()
        # Consistent feature names for training and prediction
        self.feature_names = [
            'ema_fast', 'ema_slow', 'rsi', 'atr',
            'ema_fast_m15', 'ema_slow_m15', 'rsi_m15', 'atr_m15',
            'ema_fast_h1', 'ema_slow_h1', 'rsi_h1', 'atr_h1'
        ]
        
        # Add reconnection settings
        self.max_retries = 3
        self.retry_delay = 60  # seconds

    def connect_mt5_with_retry(self):
        """Connect to MT5 with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if self.mt5.initialize():
                    if self.mt5.login(login=int(EXNESS_ACCOUNT), 
                                    password=EXNESS_PASSWORD, 
                                    server=EXNESS_SERVER):
                        print("✅ MT5 connected successfully.")
                        return True
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
            
            if attempt < self.max_retries - 1:
                print(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        return False

    def fetch_data(self, symbol, timeframe, bars=100):
        """Fetch historical data from MT5."""
        print(f"Fetching data for {symbol} on timeframe {timeframe}...")
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            print(f"❌ Failed to fetch data for {symbol} on timeframe {timeframe}")
            self.send_telegram_message(f"Failed to fetch data for {symbol}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(f"Data fetched for {symbol} successfully.")
        return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    def add_features(self, df):
        """Add technical indicators to the DataFrame."""
        df['ema_fast'] = EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_slow'] = EMAIndicator(df['close'], window=26).ema_indicator()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df.dropna(inplace=True)
        return df

    def label_data(self, df):
        """Label data based on future price movements."""
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['label'] = np.where(df['future_return'] > 0.001, 'buy',
                               np.where(df['future_return'] < -0.001, 'sell', 'hold'))
        return df

    def train_model(self):
        """Train the Random Forest model."""
        print("Starting model training...")
        all_X = []
        all_y = []
        for symbol in self.symbols:
            df_m5 = self.fetch_data(symbol, self.timeframe_trade, bars=1000)
            df_m15 = self.fetch_data(symbol, self.timeframe_confirm1, bars=100)
            df_h1 = self.fetch_data(symbol, self.timeframe_confirm2, bars=100)
            if df_m5 is None or df_m15 is None or df_h1 is None:
                continue
            df_m5 = self.add_features(df_m5)
            df_m15 = self.add_features(df_m15)
            df_h1 = self.add_features(df_h1)
            df_m5 = self.label_data(df_m5)
            # Align M15 and H1 features with M5 data
            df_combined = pd.merge_asof(df_m5, df_m15[['time', 'ema_fast', 'ema_slow', 'rsi', 'atr']],
                                        on='time', direction='backward', suffixes=('', '_m15'))
            df_combined = pd.merge_asof(df_combined, df_h1[['time', 'ema_fast', 'ema_slow', 'rsi', 'atr']],
                                        on='time', direction='backward', suffixes=('', '_h1'))
            X = df_combined[self.feature_names]
            y = df_combined['label']
            all_X.append(X)
            all_y.append(y)
        if not all_X:
            print("❌ No data available for training.")
            self.send_telegram_message("No data available for training.")
            return
        X = pd.concat(all_X)
        y = pd.concat(all_y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")
        print("Model training completed.")
        self.send_telegram_message("Model trained and saved successfully.")

    def create_feature_df(self, latest_m5, latest_m15, latest_h1):
        """Create a DataFrame for features with consistent column names."""
        features_dict = {
            'ema_fast': latest_m5['ema_fast'],
            'ema_slow': latest_m5['ema_slow'],
            'rsi': latest_m5['rsi'],
            'atr': latest_m5['atr'],
            'ema_fast_m15': latest_m15['ema_fast'],
            'ema_slow_m15': latest_m15['ema_slow'],
            'rsi_m15': latest_m15['rsi'],
            'atr_m15': latest_m15['atr'],
            'ema_fast_h1': latest_h1['ema_fast'],
            'ema_slow_h1': latest_h1['ema_slow'],
            'rsi_h1': latest_h1['rsi'],
            'atr_h1': latest_h1['atr']
        }
        return pd.DataFrame([features_dict], columns=self.feature_names)

    def backtest(self):
        """Backtest the strategy using historical data."""
        print("\nStarting backtest...")
        print(f"Initial balance: {'$' if self.account_type == 'standard' else '¢'}{self.initial_balance}")
        
        total_profit = 0
        trades = []
        balance = self.initial_balance
        winning_trades = 0
        losing_trades = 0
        
        for symbol in self.symbols:
            print(f"\nTesting {symbol}...")
            df_m15 = self.fetch_data(symbol, self.timeframe_confirm1, bars=1000)
            df_h1 = self.fetch_data(symbol, self.timeframe_confirm2, bars=1000)
            df_m5 = self.fetch_data(symbol, self.timeframe_trade, bars=1000)
            
            if df_m5 is None or df_m15 is None or df_h1 is None:
                continue
                
            df_m5 = self.add_features(df_m5)
            df_m15 = self.add_features(df_m15)
            df_h1 = self.add_features(df_h1)
            
            for i in range(1, len(df_m5) - 1):
                latest_m5 = df_m5.iloc[i]
                latest_m15 = df_m15[df_m15['time'] <= latest_m5['time']].iloc[-1]
                latest_h1 = df_h1[df_h1['time'] <= latest_m5['time']].iloc[-1]
                
                X_df = self.create_feature_df(latest_m5, latest_m15, latest_h1)
                X_scaled = self.scaler.transform(X_df)
                prediction = self.model.predict(X_scaled)[0]
                
                atr = latest_m5['atr']
                price = latest_m5['close']
                risk_per_trade = self.risk_per_trade_initial if balance < self.initial_balance * self.secure_threshold else self.risk_per_trade_aggressive
                
                tick_value = self.mt5.symbol_info(symbol).trade_tick_value
                pip_value = tick_value * (1 if self.account_type == 'standard' else 10)
                point = self.mt5.symbol_info(symbol).point
                
                if prediction == 'buy' and latest_m15['ema_fast'] > latest_m15['ema_slow'] and latest_h1['ema_fast'] > latest_h1['ema_slow']:
                    sl = price - atr * 1.5
                    tp = price + atr * 3.0
                    sl_pips = (price - sl) / point
                    lot_size = (balance * risk_per_trade) / (sl_pips * pip_value)
                    lot_size = max(self.min_lot, min(lot_size, self.max_lot))
                    future_price = df_m5['close'].iloc[i + 1]
                    if future_price >= tp:
                        profit = (tp - price) / point * pip_value * lot_size
                    elif future_price <= sl:
                        profit = (sl - price) / point * pip_value * lot_size
                    else:
                        profit = (future_price - price) / point * pip_value * lot_size
                    balance += profit
                    total_profit += profit
                    if profit > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    trades.append((symbol, 'buy', profit, balance))
                elif prediction == 'sell' and latest_m15['ema_fast'] < latest_m15['ema_slow'] and latest_h1['ema_fast'] < latest_h1['ema_slow']:
                    sl = price + atr * 1.5
                    tp = price - atr * 3.0
                    sl_pips = (sl - price) / point
                    lot_size = (balance * risk_per_trade) / (sl_pips * pip_value)
                    lot_size = max(self.min_lot, min(lot_size, self.max_lot))
                    future_price = df_m5['close'].iloc[i + 1]
                    if future_price <= tp:
                        profit = (price - tp) / point * pip_value * lot_size
                    elif future_price >= sl:
                        profit = (price - sl) / point * pip_value * lot_size
                    else:
                        profit = (price - future_price) / point * pip_value * lot_size
                    balance += profit
                    total_profit += profit
                    if profit > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    trades.append((symbol, 'sell', profit, balance))

        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        print("\n=== Backtest Results ===")
        print(f"Initial Balance: {'$' if self.account_type == 'standard' else '¢'}{self.initial_balance:.2f}")
        print(f"Final Balance: {'$' if self.account_type == 'standard' else '¢'}{balance:.2f}")
        print(f"Total Profit/Loss: {'$' if self.account_type == 'standard' else '¢'}{total_profit:.2f}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Return on Investment: {((balance - self.initial_balance) / self.initial_balance * 100):.2f}%")
        
        # Print last 5 trades
        print("\nLast 5 trades:")
        for trade in trades[-5:]:
            symbol, action, profit, trade_balance = trade
            print(f"{symbol} {action}: {'$' if self.account_type == 'standard' else '¢'}{profit:.2f} (Balance: {'$' if self.account_type == 'standard' else '¢'}{trade_balance:.2f})")
        
        print("\nBacktest completed.")
        
        # Send summary to Telegram
        summary = (
            f"Backtest Results:\n"
            f"Initial: {'$' if self.account_type == 'standard' else '¢'}{self.initial_balance:.2f}\n"
            f"Final: {'$' if self.account_type == 'standard' else '¢'}{balance:.2f}\n"
            f"Profit/Loss: {'$' if self.account_type == 'standard' else '¢'}{total_profit:.2f}\n"
            f"Trades: {total_trades}\n"
            f"Win Rate: {win_rate:.2f}%"
        )
        self.send_telegram_message(summary)

    def execute_trade(self, symbol, action):
        """Execute a live trade with risk management."""
        account_balance = self.mt5.account_info().balance  # Balance in cents
        risk_per_trade = self.risk_per_trade_initial if account_balance < self.initial_balance * self.secure_threshold else self.risk_per_trade_aggressive
        df_m5 = self.fetch_data(symbol, self.timeframe_trade, bars=100)
        df_m5 = self.add_features(df_m5)
        atr = df_m5['atr'].iloc[-1]
        price = df_m5['close'].iloc[-1]
        point = self.mt5.symbol_info(symbol).point
        tick_value = self.mt5.symbol_info(symbol).trade_tick_value  # In cents
        pip_value = tick_value * 10  # For 5-digit pricing
        if action == 'buy':
            sl = price - atr * 1.5
            tp = price + atr * 3.0
            sl_pips = (price - sl) / point
        else:
            sl = price + atr * 1.5
            tp = price - atr * 3.0
            sl_pips = (sl - price) / point
        lot_size = (account_balance * risk_per_trade) / (sl_pips * pip_value)
        lot_size = max(self.min_lot, min(lot_size, self.max_lot))  # Adjust based on broker limits
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 123456,
            "comment": "Scalping Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.send_telegram_message(f"Trade failed for {symbol}: {result.comment}")
        else:
            self.send_telegram_message(f"Trade executed: {action} {symbol}, Lot: {lot_size:.2f}")

    def run(self):
        """Main trading loop with cloud-friendly modifications"""
        print("\n=== Bot Configuration ===")
        print(f"Mode: {self.mode}")
        print(f"Account Type: {self.account_type}")
        print(f"Trading Symbols: {', '.join(self.symbols)}")
        print(f"Initial Balance: {self.initial_balance}")
        print("=====================\n")
        
        if not self.connect_mt5_with_retry():
            print("Failed to connect to MT5 after multiple attempts")
            return
            
        try:
            if not self.load_model():
                self.train_model()
            
            if self.mode == 'backtest':
                self.backtest()
                return
                
            # Single iteration for cloud service scheduled runs
            self.check_and_trade()
            
        except Exception as e:
            print(f"Error: {e}")
            self.send_telegram_message(f"Bot error: {e}")
        finally:
            if self.mt5.initialize():
                self.mt5.shutdown()

    def check_and_trade(self):
        """Single iteration of trading checks"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}] Checking for trading opportunities...")
        
        for symbol in self.symbols:
            try:
                df_m5 = self.fetch_data(symbol, self.timeframe_trade, bars=100)
                df_m15 = self.fetch_data(symbol, self.timeframe_confirm1, bars=100)
                df_h1 = self.fetch_data(symbol, self.timeframe_confirm2, bars=100)
                
                if df_m5 is None or df_m15 is None or df_h1 is None:
                    print(f"Skipping {symbol} due to missing data")
                    continue
                
                df_m5 = self.add_features(df_m5)
                df_m15 = self.add_features(df_m15)
                df_h1 = self.add_features(df_h1)
                
                latest_m5 = df_m5.iloc[-1]
                latest_m15 = df_m15.iloc[-1]
                latest_h1 = df_h1.iloc[-1]
                
                X_df = self.create_feature_df(latest_m5, latest_m15, latest_h1)
                X_scaled = self.scaler.transform(X_df)
                prediction = self.model.predict(X_scaled)[0]
                
                print(f"\nAnalyzing {symbol}:")
                print(f"Prediction: {prediction}")
                print(f"M15 EMA Status: {'Bullish' if latest_m15['ema_fast'] > latest_m15['ema_slow'] else 'Bearish'}")
                print(f"H1 EMA Status: {'Bullish' if latest_h1['ema_fast'] > latest_h1['ema_slow'] else 'Bearish'}")
                
                if prediction == 'buy' and latest_m15['ema_fast'] > latest_m15['ema_slow'] and latest_h1['ema_fast'] > latest_h1['ema_slow']:
                    print(f"✅ BUY Signal detected for {symbol}")
                    self.execute_trade(symbol, 'buy')
                elif prediction == 'sell' and latest_m15['ema_fast'] < latest_m15['ema_slow'] and latest_h1['ema_fast'] < latest_h1['ema_slow']:
                    print(f"✅ SELL Signal detected for {symbol}")
                    self.execute_trade(symbol, 'sell')
                else:
                    print(f"No trading opportunity for {symbol}")
                    
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                self.send_telegram_message(f"Error processing {symbol}: {str(e)}")
                
        print(f"\nWaiting for 5 minutes before next check...")
        time.sleep(300)  # Check every 5 minutes

    def load_model(self):
        """Load the trained model and scaler if available."""
        if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
            self.model = joblib.load("model.pkl")
            self.scaler = joblib.load("scaler.pkl")
            print("Model loaded successfully.")
            return True
        print("No saved model found.")
        return False

    def send_telegram_message(self, message):
        """Send a message via Telegram."""
        try:
            self.telegram_bot.send_message(self.chat_id, message)
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scalping Bot')
    parser.add_argument('--mode', choices=['live', 'backtest'], default='backtest',
                      help='Trading mode: live or backtest')
    parser.add_argument('--account', choices=['cent', 'standard'], default='cent',
                      help='Account type: cent or standard')
    
    args = parser.parse_args()
    
    try:
        print(f"Starting Scalping Bot in {args.mode} mode with {args.account} account...")
        bot = ScalpingBot(mode=args.mode, account_type=args.account)
        bot.run()  # This line was missing
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if mt5.initialize():
            mt5.shutdown()
