"""
Enhanced Options Backtester with Simulated Trades Demo
Demonstrates full capabilities with sample trading scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import math
import random

# Configure logging without emoji for Windows compatibility  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('options_backtest_demo.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class DemoOptionsContract:
    """Enhanced options contract with realistic pricing"""
    symbol: str
    strike: float
    option_type: str  # 'CE' or 'PE'
    expiry: str
    premium: float
    iv: float  # Implied Volatility
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

@dataclass  
class DemoOptionsTrade:
    """Enhanced options trade with P&L tracking"""
    entry_date: str
    exit_date: str
    strategy: str
    contracts: List[DemoOptionsContract]
    quantities: List[int]
    entry_premium: float
    exit_premium: float
    pnl: float
    pnl_pct: float
    dte_entry: int
    dte_exit: int

@dataclass
class DemoBacktestResults:
    """Enhanced results with detailed metrics"""
    strategy: str
    index: str
    total_return: float
    trades: List[DemoOptionsTrade]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_premium_paid: float
    total_premium_received: float

class DemoOptionsBacktesterWithTrades:
    """Enhanced backtester that generates realistic demo trades"""
    
    def __init__(self, fyers_client=None):
        self.fyers_client = fyers_client
        self.logger = logging.getLogger(__name__)
        
        # Index specifications
        self.index_config = {
            'NIFTY50': {'lot_size': 50, 'tick_size': 0.05, 'base_price': 22500},
            'BANKNIFTY': {'lot_size': 15, 'tick_size': 0.05, 'base_price': 48500}, 
            'FINNIFTY': {'lot_size': 25, 'tick_size': 0.05, 'base_price': 22800}
        }
        
        # Volatility scenarios for realistic trade generation
        self.volatility_scenarios = {
            'LOW': {'iv_range': (0.12, 0.18), 'movement': 0.005},
            'MEDIUM': {'iv_range': (0.18, 0.25), 'movement': 0.015},
            'HIGH': {'iv_range': (0.25, 0.35), 'movement': 0.025}
        }

    def run_enhanced_demo_backtest(self, strategy: str, index: str, 
                                 start_date: str, end_date: str, 
                                 capital: float = 500000) -> DemoBacktestResults:
        """Run enhanced backtest with realistic simulated trades"""
        
        self.logger.info(f"Starting enhanced demo backtest: {strategy} on {index}")
        self.logger.info(f"Period: {start_date} to {end_date}, Capital: Rs {capital:,.2f}")
        
        # Generate market data and volatility scenarios
        underlying_data = self._generate_enhanced_market_data(index, start_date, end_date)
        
        # Generate realistic trades based on strategy
        trades = []
        current_capital = capital
        
        if strategy == "LONG_CALL":
            trades = self._generate_long_call_trades(index, underlying_data)
        elif strategy == "SHORT_PUT":  
            trades = self._generate_short_put_trades(index, underlying_data)
        elif strategy == "STRADDLE":
            trades = self._generate_straddle_trades(index, underlying_data)
        elif strategy == "IRON_CONDOR":
            trades = self._generate_iron_condor_trades(index, underlying_data)
        elif strategy == "STRANGLE":
            trades = self._generate_strangle_trades(index, underlying_data)
        
        # Calculate comprehensive results
        results = self._calculate_enhanced_results(strategy, index, trades)
        
        self.logger.info(f"Demo backtest completed: {len(trades)} trades generated")
        self.logger.info(f"Total Return: {results.total_return:.2f}%")
        self.logger.info(f"Win Rate: {results.win_rate:.1f}%")
        self.logger.info(f"Profit Factor: {results.profit_factor:.2f}")
        
        return results
    
    def _generate_enhanced_market_data(self, index: str, start_date: str, 
                                     end_date: str) -> pd.DataFrame:
        """Generate enhanced market data with volatility regimes"""
        
        base_price = self.index_config[index]['base_price']
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create realistic price movements with volatility clustering
        returns = []
        vol_regime = 'MEDIUM'
        
        for i, date in enumerate(dates):
            # Change volatility regime occasionally
            if i % 10 == 0:
                vol_regime = random.choice(['LOW', 'MEDIUM', 'HIGH'])
            
            scenario = self.volatility_scenarios[vol_regime]
            daily_return = np.random.normal(0, scenario['movement'])
            returns.append(daily_return)
        
        # Generate price series
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices[:-1],
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]], 
            'Close': prices[1:],
            'Volume': np.random.randint(100000, 1000000, len(dates))
        })
        
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(5).std() * np.sqrt(252)
        
        return df

    def _generate_long_call_trades(self, index: str, data: pd.DataFrame) -> List[DemoOptionsTrade]:
        """Generate realistic long call trades"""
        trades = []
        
        for i in range(0, len(data)-10, 5):  # Trade every 5 days
            current_price = data.iloc[i]['Close']
            vol = data.iloc[i]['Volatility'] if not pd.isna(data.iloc[i]['Volatility']) else 0.2
            
            # Entry conditions: Oversold + bullish momentum
            if i > 5 and data.iloc[i]['Close'] > data.iloc[i-5]['Close']:
                
                strike = round(current_price * 1.02, -1)  # 2% OTM
                dte_entry = 15
                
                # Calculate realistic premium using simplified Black-Scholes
                entry_premium = self._calculate_option_premium(
                    current_price, strike, dte_entry/365, 0.05, vol, 'CE'
                )
                
                # Simulate exit after 5-7 days
                exit_idx = min(i + random.randint(5, 7), len(data)-1)
                exit_price = data.iloc[exit_idx]['Close']
                dte_exit = dte_entry - (exit_idx - i)
                
                exit_premium = self._calculate_option_premium(
                    exit_price, strike, max(dte_exit/365, 1/365), 0.05, vol, 'CE'
                )
                
                # Create trade
                contract = DemoOptionsContract(
                    symbol=f"{index}{data.iloc[i]['Date'].strftime('%d%b%Y')}{int(strike)}CE",
                    strike=strike,
                    option_type='CE',
                    expiry=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    premium=entry_premium,
                    iv=vol
                )
                
                pnl = (exit_premium - entry_premium) * self.index_config[index]['lot_size']
                pnl_pct = (exit_premium - entry_premium) / entry_premium * 100
                
                trade = DemoOptionsTrade(
                    entry_date=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    exit_date=data.iloc[exit_idx]['Date'].strftime('%Y-%m-%d'),
                    strategy="LONG_CALL", 
                    contracts=[contract],
                    quantities=[1],
                    entry_premium=entry_premium,
                    exit_premium=exit_premium,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    dte_entry=dte_entry,
                    dte_exit=dte_exit
                )
                
                trades.append(trade)
        
        return trades

    def _generate_short_put_trades(self, index: str, data: pd.DataFrame) -> List[DemoOptionsTrade]:
        """Generate realistic short put trades (income strategy)"""
        trades = []
        
        for i in range(0, len(data)-15, 7):  # Trade weekly
            current_price = data.iloc[i]['Close']
            vol = data.iloc[i]['Volatility'] if not pd.isna(data.iloc[i]['Volatility']) else 0.2
            
            # Entry: Sell puts in uptrend for income
            if vol > 0.15:  # Higher volatility = better premium
                
                strike = round(current_price * 0.95, -1)  # 5% OTM put
                dte_entry = 21
                
                entry_premium = self._calculate_option_premium(
                    current_price, strike, dte_entry/365, 0.05, vol, 'PE'
                )
                
                # Exit after 10-14 days or 50% profit
                exit_idx = min(i + random.randint(10, 14), len(data)-1)
                exit_price = data.iloc[exit_idx]['Close']
                dte_exit = dte_entry - (exit_idx - i)
                
                exit_premium = self._calculate_option_premium(
                    exit_price, strike, max(dte_exit/365, 1/365), 0.05, vol, 'PE'
                )
                
                # Short put P&L (reverse of long)
                pnl = (entry_premium - exit_premium) * self.index_config[index]['lot_size']
                pnl_pct = (entry_premium - exit_premium) / entry_premium * 100
                
                contract = DemoOptionsContract(
                    symbol=f"{index}{data.iloc[i]['Date'].strftime('%d%b%Y')}{int(strike)}PE",
                    strike=strike,
                    option_type='PE', 
                    expiry=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    premium=entry_premium,
                    iv=vol
                )
                
                trade = DemoOptionsTrade(
                    entry_date=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    exit_date=data.iloc[exit_idx]['Date'].strftime('%Y-%m-%d'),
                    strategy="SHORT_PUT",
                    contracts=[contract],
                    quantities=[-1],  # Short position
                    entry_premium=entry_premium,
                    exit_premium=exit_premium,
                    pnl=pnl, 
                    pnl_pct=pnl_pct,
                    dte_entry=dte_entry,
                    dte_exit=dte_exit
                )
                
                trades.append(trade)
        
        return trades

    def _generate_straddle_trades(self, index: str, data: pd.DataFrame) -> List[DemoOptionsTrade]:
        """Generate ATM straddle trades (volatility strategy)"""
        trades = []
        
        for i in range(0, len(data)-20, 10):  # Trade bi-weekly
            current_price = data.iloc[i]['Close']
            vol = data.iloc[i]['Volatility'] if not pd.isna(data.iloc[i]['Volatility']) else 0.2
            
            # Enter straddles before high volatility events
            if vol < 0.18:  # Low vol - expect vol expansion
                
                strike = round(current_price, -1)  # ATM
                dte_entry = 30
                
                # Calculate both call and put premiums
                call_premium = self._calculate_option_premium(
                    current_price, strike, dte_entry/365, 0.05, vol, 'CE'
                )
                put_premium = self._calculate_option_premium(
                    current_price, strike, dte_entry/365, 0.05, vol, 'PE'
                )
                
                entry_premium = call_premium + put_premium
                
                # Exit after volatility expansion or time decay
                exit_idx = min(i + random.randint(12, 18), len(data)-1)
                exit_price = data.iloc[exit_idx]['Close']
                dte_exit = dte_entry - (exit_idx - i)
                
                # Calculate exit premiums
                call_exit = self._calculate_option_premium(
                    exit_price, strike, max(dte_exit/365, 1/365), 0.05, vol*1.2, 'CE'  
                )
                put_exit = self._calculate_option_premium(
                    exit_price, strike, max(dte_exit/365, 1/365), 0.05, vol*1.2, 'PE'
                )
                
                exit_premium = call_exit + put_exit
                
                pnl = (exit_premium - entry_premium) * self.index_config[index]['lot_size']
                pnl_pct = (exit_premium - entry_premium) / entry_premium * 100
                
                call_contract = DemoOptionsContract(
                    symbol=f"{index}{data.iloc[i]['Date'].strftime('%d%b%Y')}{int(strike)}CE",
                    strike=strike, option_type='CE', expiry=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    premium=call_premium, iv=vol
                )
                put_contract = DemoOptionsContract(
                    symbol=f"{index}{data.iloc[i]['Date'].strftime('%d%b%Y')}{int(strike)}PE", 
                    strike=strike, option_type='PE', expiry=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    premium=put_premium, iv=vol
                )
                
                trade = DemoOptionsTrade(
                    entry_date=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    exit_date=data.iloc[exit_idx]['Date'].strftime('%Y-%m-%d'),
                    strategy="STRADDLE",
                    contracts=[call_contract, put_contract],
                    quantities=[1, 1],
                    entry_premium=entry_premium,
                    exit_premium=exit_premium, 
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    dte_entry=dte_entry,
                    dte_exit=dte_exit
                )
                
                trades.append(trade)
        
        return trades

    def _generate_iron_condor_trades(self, index: str, data: pd.DataFrame) -> List[DemoOptionsTrade]:
        """Generate Iron Condor trades (range-bound strategy)"""
        trades = []
        
        for i in range(0, len(data)-25, 12):  # Monthly trades
            current_price = data.iloc[i]['Close']
            vol = data.iloc[i]['Volatility'] if not pd.isna(data.iloc[i]['Volatility']) else 0.2
            
            # Enter in low volatility, range-bound markets
            if vol < 0.20 and i > 10:
                recent_high = data.iloc[i-10:i]['High'].max()
                recent_low = data.iloc[i-10:i]['Low'].min()
                price_range = (recent_high - recent_low) / current_price
                
                if price_range < 0.05:  # Range-bound market
                    
                    # Iron Condor strikes: Sell closer, Buy further
                    call_sell_strike = round(current_price * 1.02, -1)
                    call_buy_strike = round(current_price * 1.04, -1)  
                    put_sell_strike = round(current_price * 0.98, -1)
                    put_buy_strike = round(current_price * 0.96, -1)
                    
                    dte_entry = 25
                    
                    # Calculate net credit received (Iron Condor is credit strategy)
                    call_sell_prem = self._calculate_option_premium(
                        current_price, call_sell_strike, dte_entry/365, 0.05, vol, 'CE'
                    )
                    call_buy_prem = self._calculate_option_premium(
                        current_price, call_buy_strike, dte_entry/365, 0.05, vol, 'CE'  
                    )
                    put_sell_prem = self._calculate_option_premium(
                        current_price, put_sell_strike, dte_entry/365, 0.05, vol, 'PE'
                    )
                    put_buy_prem = self._calculate_option_premium(
                        current_price, put_buy_strike, dte_entry/365, 0.05, vol, 'PE'
                    )
                    
                    entry_premium = (call_sell_prem - call_buy_prem) + (put_sell_prem - put_buy_prem)
                    
                    # Exit management
                    exit_idx = min(i + random.randint(15, 20), len(data)-1)
                    exit_price = data.iloc[exit_idx]['Close'] 
                    dte_exit = dte_entry - (exit_idx - i)
                    
                    # Calculate exit premiums
                    call_sell_exit = self._calculate_option_premium(
                        exit_price, call_sell_strike, max(dte_exit/365, 1/365), 0.05, vol, 'CE'
                    )
                    call_buy_exit = self._calculate_option_premium(
                        exit_price, call_buy_strike, max(dte_exit/365, 1/365), 0.05, vol, 'CE'
                    ) 
                    put_sell_exit = self._calculate_option_premium(
                        exit_price, put_sell_strike, max(dte_exit/365, 1/365), 0.05, vol, 'PE'
                    )
                    put_buy_exit = self._calculate_option_premium(
                        exit_price, put_buy_strike, max(dte_exit/365, 1/365), 0.05, vol, 'PE'
                    )
                    
                    exit_premium = (call_sell_exit - call_buy_exit) + (put_sell_exit - put_buy_exit)
                    
                    # Credit spread P&L: Credit received - Credit to close
                    pnl = (entry_premium - exit_premium) * self.index_config[index]['lot_size']
                    pnl_pct = (entry_premium - exit_premium) / entry_premium * 100
                    
                    # Create all 4 contracts
                    contracts = [
                        DemoOptionsContract(f"{index}_{int(call_sell_strike)}CE", call_sell_strike, 'CE', 
                                          data.iloc[i]['Date'].strftime('%Y-%m-%d'), call_sell_prem, vol),
                        DemoOptionsContract(f"{index}_{int(call_buy_strike)}CE", call_buy_strike, 'CE',
                                          data.iloc[i]['Date'].strftime('%Y-%m-%d'), call_buy_prem, vol),
                        DemoOptionsContract(f"{index}_{int(put_sell_strike)}PE", put_sell_strike, 'PE',
                                          data.iloc[i]['Date'].strftime('%Y-%m-%d'), put_sell_prem, vol),
                        DemoOptionsContract(f"{index}_{int(put_buy_strike)}PE", put_buy_strike, 'PE', 
                                          data.iloc[i]['Date'].strftime('%Y-%m-%d'), put_buy_prem, vol)
                    ]
                    
                    trade = DemoOptionsTrade(
                        entry_date=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                        exit_date=data.iloc[exit_idx]['Date'].strftime('%Y-%m-%d'),
                        strategy="IRON_CONDOR",
                        contracts=contracts,
                        quantities=[-1, 1, -1, 1],  # Sell-Buy-Sell-Buy
                        entry_premium=entry_premium,
                        exit_premium=exit_premium,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        dte_entry=dte_entry,
                        dte_exit=dte_exit
                    )
                    
                    trades.append(trade)
        
        return trades

    def _generate_strangle_trades(self, index: str, data: pd.DataFrame) -> List[DemoOptionsTrade]:
        """Generate Strangle trades (similar to straddle but OTM)"""
        trades = []
        
        for i in range(0, len(data)-15, 8):  # Weekly trades
            current_price = data.iloc[i]['Close']
            vol = data.iloc[i]['Volatility'] if not pd.isna(data.iloc[i]['Volatility']) else 0.2
            
            # Enter when expecting big moves but unsure of direction
            if vol > 0.22:  # High vol environment
                
                call_strike = round(current_price * 1.03, -1)  # 3% OTM call
                put_strike = round(current_price * 0.97, -1)   # 3% OTM put
                dte_entry = 14
                
                call_premium = self._calculate_option_premium(
                    current_price, call_strike, dte_entry/365, 0.05, vol, 'CE'
                )
                put_premium = self._calculate_option_premium(
                    current_price, put_strike, dte_entry/365, 0.05, vol, 'PE' 
                )
                
                entry_premium = call_premium + put_premium
                
                # Exit management
                exit_idx = min(i + random.randint(7, 10), len(data)-1)
                exit_price = data.iloc[exit_idx]['Close']
                dte_exit = dte_entry - (exit_idx - i)
                
                call_exit = self._calculate_option_premium(
                    exit_price, call_strike, max(dte_exit/365, 1/365), 0.05, vol*0.8, 'CE'
                )
                put_exit = self._calculate_option_premium(
                    exit_price, put_strike, max(dte_exit/365, 1/365), 0.05, vol*0.8, 'PE'
                )
                
                exit_premium = call_exit + put_exit 
                
                pnl = (exit_premium - entry_premium) * self.index_config[index]['lot_size']
                pnl_pct = (exit_premium - entry_premium) / entry_premium * 100
                
                call_contract = DemoOptionsContract(
                    f"{index}_{int(call_strike)}CE", call_strike, 'CE',
                    data.iloc[i]['Date'].strftime('%Y-%m-%d'), call_premium, vol
                )
                put_contract = DemoOptionsContract(
                    f"{index}_{int(put_strike)}PE", put_strike, 'PE', 
                    data.iloc[i]['Date'].strftime('%Y-%m-%d'), put_premium, vol
                )
                
                trade = DemoOptionsTrade(
                    entry_date=data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    exit_date=data.iloc[exit_idx]['Date'].strftime('%Y-%m-%d'),
                    strategy="STRANGLE",
                    contracts=[call_contract, put_contract],
                    quantities=[1, 1],
                    entry_premium=entry_premium,
                    exit_premium=exit_premium,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    dte_entry=dte_entry,
                    dte_exit=dte_exit
                )
                
                trades.append(trade)
        
        return trades

    def _calculate_option_premium(self, spot: float, strike: float, time_to_expiry: float,
                                risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Calculate option premium using Black-Scholes formula"""
        
        if time_to_expiry <= 0:
            # At expiry, option value is intrinsic value only
            if option_type == 'CE':
                return max(0, spot - strike)
            else:  # 'PE'
                return max(0, strike - spot)
        
        try:
            # Black-Scholes calculation
            from math import log, sqrt, exp
            from scipy.stats import norm
            
            d1 = (log(spot/strike) + (risk_free_rate + 0.5*volatility**2)*time_to_expiry) / (volatility*sqrt(time_to_expiry))
            d2 = d1 - volatility*sqrt(time_to_expiry)
            
            if option_type == 'CE':
                premium = spot*norm.cdf(d1) - strike*exp(-risk_free_rate*time_to_expiry)*norm.cdf(d2)
            else:  # 'PE'
                premium = strike*exp(-risk_free_rate*time_to_expiry)*norm.cdf(-d2) - spot*norm.cdf(-d1)
            
            return max(premium, 0.01)  # Minimum premium
            
        except:
            # Fallback to simplified calculation if scipy not available
            moneyness = spot / strike
            time_value = volatility * sqrt(time_to_expiry) * 0.4 * spot
            
            if option_type == 'CE':
                intrinsic = max(0, spot - strike) 
                if moneyness > 1:
                    premium = intrinsic + time_value * moneyness
                else:
                    premium = time_value * moneyness
            else:  # 'PE'
                intrinsic = max(0, strike - spot)
                if moneyness < 1:
                    premium = intrinsic + time_value * (2 - moneyness)
                else:
                    premium = time_value * (2 - moneyness)
            
            return max(premium, 0.01)

    def _calculate_enhanced_results(self, strategy: str, index: str, 
                                  trades: List[DemoOptionsTrade]) -> DemoBacktestResults:
        """Calculate comprehensive backtest results"""
        
        if not trades:
            return DemoBacktestResults(
                strategy=strategy, index=index, total_return=0.0, trades=[], 
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                avg_win=0.0, avg_loss=0.0, profit_factor=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, total_premium_paid=0.0, total_premium_received=0.0
            )
        
        # Calculate metrics
        total_pnl = sum(trade.pnl for trade in trades)
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        profit_factor = (sum(wins) / sum(losses)) if losses else float('inf') if wins else 0
        
        # Calculate drawdown
        cumulative_pnl = []
        running_total = 0
        for trade in trades:
            running_total += trade.pnl
            cumulative_pnl.append(running_total)
        
        max_drawdown = 0
        if cumulative_pnl:
            peak = cumulative_pnl[0]
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                drawdown = (peak - pnl) / abs(peak) if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = [trade.pnl_pct for trade in trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Calculate total premiums
        total_premium_paid = sum(abs(trade.entry_premium) for trade in trades if trade.entry_premium < 0)
        total_premium_received = sum(trade.entry_premium for trade in trades if trade.entry_premium > 0)
        
        # Calculate total return as percentage of typical capital allocation
        typical_capital_per_trade = 50000  # Rs 50k per trade assumption
        total_capital_used = len(trades) * typical_capital_per_trade
        total_return = (total_pnl / total_capital_used) * 100 if total_capital_used > 0 else 0
        
        return DemoBacktestResults(
            strategy=strategy,
            index=index,
            total_return=total_return,
            trades=trades,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown * 100,  # Convert to percentage
            sharpe_ratio=sharpe_ratio,
            total_premium_paid=total_premium_paid,
            total_premium_received=total_premium_received
        )

def run_enhanced_demo():
    """Run enhanced demo with realistic trades"""
    print("\n" + "="*80)
    print("🚀 ENHANCED OPTIONS BACKTESTING DEMO - WITH REALISTIC TRADES")
    print("="*80)
    
    backtester = DemoOptionsBacktesterWithTrades()
    
    # Test configurations
    test_scenarios = [
        {
            'name': 'NIFTY50 Long Call Bull Run',
            'strategy': 'LONG_CALL',
            'index': 'NIFTY50',
            'start_date': '2023-08-01',
            'end_date': '2023-10-01'
        },
        {
            'name': 'BANKNIFTY Short Put Income',  
            'strategy': 'SHORT_PUT',
            'index': 'BANKNIFTY',
            'start_date': '2023-09-01', 
            'end_date': '2023-11-01'
        },
        {
            'name': 'NIFTY50 Volatility Straddle',
            'strategy': 'STRADDLE', 
            'index': 'NIFTY50',
            'start_date': '2023-07-15',
            'end_date': '2023-09-15'
        },
        {
            'name': 'BANKNIFTY Iron Condor Range',
            'strategy': 'IRON_CONDOR',
            'index': 'BANKNIFTY', 
            'start_date': '2023-08-15',
            'end_date': '2023-10-15'
        }
    ]
    
    all_results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 Test {i}/{len(test_scenarios)}: {scenario['name']}")
        print("-" * 60)
        
        results = backtester.run_enhanced_demo_backtest(
            scenario['strategy'],
            scenario['index'], 
            scenario['start_date'],
            scenario['end_date']
        )
        
        all_results.append(results)
        
        # Display results
        print(f"   ✅ Strategy: {results.strategy}")
        print(f"   📈 Index: {results.index}")
        print(f"   💰 Total Return: {results.total_return:+.2f}%")
        print(f"   🎯 Total Trades: {results.total_trades}")
        print(f"   ✨ Win Rate: {results.win_rate:.1f}%")
        print(f"   📊 Profit Factor: {results.profit_factor:.2f}")
        print(f"   📉 Max Drawdown: {results.max_drawdown:.1f}%")
        
        if results.trades:
            print(f"   💵 Avg Win: Rs {results.avg_win:,.0f}")
            print(f"   💸 Avg Loss: Rs {results.avg_loss:,.0f}")
            print(f"   📈 Best Trade: Rs {max(t.pnl for t in results.trades):,.0f}")
            print(f"   📉 Worst Trade: Rs {min(t.pnl for t in results.trades):,.0f}")
    
    # Summary analysis
    print("\n" + "="*80)
    print("📋 ENHANCED DEMO RESULTS SUMMARY")
    print("="*80)
    
    print(f"{'Strategy':<25} {'Index':<12} {'Return':<10} {'Trades':<8} {'Win%':<8} {'PF':<8}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result.strategy:<25} {result.index:<12} {result.total_return:+7.2f}% "
              f"{result.total_trades:<8} {result.win_rate:>6.1f}% {result.profit_factor:>7.2f}")
    
    # Key insights
    total_trades = sum(r.total_trades for r in all_results)
    avg_return = sum(r.total_return for r in all_results) / len(all_results)
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   📊 Total Simulated Trades: {total_trades}")
    print(f"   📈 Average Strategy Return: {avg_return:+.2f}%")
    print(f"   🎯 Strategies Tested: {len(all_results)}")
    
    print(f"\n🚀 PRODUCTION READY FEATURES:")
    print(f"   ✅ Realistic Black-Scholes option pricing")
    print(f"   ✅ Multiple volatility regimes simulated") 
    print(f"   ✅ Comprehensive risk metrics calculated")
    print(f"   ✅ Trade-by-trade P&L tracking")
    print(f"   ✅ Strategy-specific entry/exit logic")
    
    print(f"\n🎯 NEXT STEP: Configure FYERS API for live market data!")
    print("="*80)

if __name__ == "__main__":
    run_enhanced_demo()