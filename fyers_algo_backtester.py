"""
FYERS ALGORITHMIC TRADING BACKTESTER
Comprehensive backtesting system for the FYERS integrated trading platform
Test your strategies before risking real money!
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import the trading system components
from jeafx_advanced_system import AdvancedJeafxSystem, AdvancedSignal
from jeafx_risk_manager import JeafxRiskManager


@dataclass
class BacktestTrade:
    """Individual trade record"""
    entry_date: datetime
    exit_date: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    signal_confidence: float
    exit_reason: str  # 'STOP_LOSS', 'TARGET', 'TIME_EXIT'
    trade_duration_days: int


@dataclass
class BacktestResults:
    """Complete backtest results"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[BacktestTrade]


class FyersAlgoBacktester:
    """
    Complete backtesting system for FYERS algorithmic trading
    Test your strategies on historical data before live trading
    """
    
    def __init__(self):
        self.jeafx_system = AdvancedJeafxSystem()
        self.risk_manager = JeafxRiskManager()
        
        # Default configuration
        self.config = {
            'initial_capital': 100000,  # ₹1 lakh
            'commission_per_trade': 20,  # ₹20 per trade
            'max_positions': 5,
            'risk_per_trade': 0.02,  # 2% per trade
            'stop_loss_percent': 2.0,
            'take_profit_percent': 4.0,
            'min_confidence': 75,
            'max_hold_days': 30
        }
        
        self._setup_logging()
        self.logger.info("🧪 FYERS Algo Backtester Initialized")
        
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fyers_backtest.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FYERS_BACKTEST')
        
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download historical data"""
        
        try:
            # Convert to Yahoo Finance format
            yf_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
            self.logger.info(f"📈 Downloading {yf_symbol} from {start_date} to {end_date}")
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"⚠️ No data for {yf_symbol}")
                return pd.DataFrame()
                
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            data = data.dropna()
            
            self.logger.info(f"✅ Downloaded {len(data)} bars for {yf_symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading {symbol}: {e}")
            return pd.DataFrame()
            
    def run_backtest(self, symbol: str, start_date: str, end_date: str) -> BacktestResults:
        """Run comprehensive backtest"""
        
        self.logger.info(f"🧪 Starting backtest: {symbol} ({start_date} to {end_date})")
        
        # Get data
        data = self.get_data(symbol, start_date, end_date)
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
            
        # Initialize backtest
        capital = self.config['initial_capital']
        position = None
        trades = []
        equity_curve = []
        
        self.logger.info(f"💰 Initial capital: ₹{capital:,.2f}")
        
        # Process each day (start after 100 bars for indicators)
        for i in range(100, len(data)):
            current_date = data.index[i]
            current_price = data.iloc[i]['close']
            current_data = data.iloc[:i+1]
            
            # Track portfolio value
            portfolio_value = capital
            if position:
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                if position['side'] == 'SELL':
                    unrealized_pnl = -unrealized_pnl
                portfolio_value += unrealized_pnl
                
            equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value
            })
            
            # Check exit conditions first
            if position:
                should_exit, exit_reason = self._check_exit_conditions(
                    position, current_price, current_date
                )
                
                if should_exit:
                    trade = self._close_position(position, current_price, current_date, exit_reason)
                    trades.append(trade)
                    capital += trade.pnl
                    position = None
                    
                    self.logger.debug(f"🔚 Closed: {trade.symbol} P&L: ₹{trade.pnl:+.2f}")
                    
            # Look for new signals (only if no position)
            if not position:
                signals = self._get_signals(symbol, current_data)
                
                for signal in signals:
                    if self._should_trade(signal):
                        position = self._enter_position(signal, current_price, current_date, capital)
                        if position:
                            self.logger.debug(f"▶️ Entered: {symbol} @ ₹{current_price}")
                        break
                        
        # Close final position if exists
        if position:
            final_price = data.iloc[-1]['close']
            trade = self._close_position(position, final_price, data.index[-1], "TIME_EXIT")
            trades.append(trade)
            capital += trade.pnl
            
        # Calculate results
        results = self._calculate_results(symbol, self.config['initial_capital'], capital, trades, equity_curve)
        
        self.logger.info(f"✅ Backtest completed!")
        self.logger.info(f"   📊 Total Return: {results.total_return:.2f}%")
        self.logger.info(f"   🎯 Win Rate: {results.win_rate:.1f}%")
        self.logger.info(f"   📈 Total Trades: {results.total_trades}")
        
        return results
        
    def _get_signals(self, symbol: str, data: pd.DataFrame) -> List[AdvancedSignal]:
        """Generate trading signals"""
        try:
            return self.jeafx_system.generate_trading_signals_from_data(symbol, data)
        except Exception as e:
            self.logger.debug(f"Signal generation error: {e}")
            return []
            
    def _should_trade(self, signal: AdvancedSignal) -> bool:
        """Check if signal meets trading criteria"""
        
        # Confidence threshold
        if signal.confidence_score < self.config['min_confidence']:
            return False
            
        # Risk-reward ratio
        if hasattr(signal, 'risk_reward_ratio') and signal.risk_reward_ratio < 1.5:
            return False
            
        return True
        
    def _enter_position(self, signal: AdvancedSignal, price: float, date: datetime, capital: float) -> Optional[Dict]:
        """Enter new position"""
        
        try:
            # Calculate position size using risk management
            risk_amount = capital * self.config['risk_per_trade']
            
            # Estimate stop loss distance (2% by default)
            stop_loss_distance = price * (self.config['stop_loss_percent'] / 100)
            quantity = int(risk_amount / stop_loss_distance)
            
            # Position size limits
            min_quantity = 1
            max_value = capital * 0.2  # Max 20% per position
            max_quantity = int(max_value / price)
            
            quantity = max(min_quantity, min(quantity, max_quantity))
            
            if quantity <= 0:
                return None
                
            return {
                'symbol': signal.symbol,
                'side': signal.signal_type,
                'entry_date': date,
                'entry_price': price,
                'quantity': quantity,
                'confidence': signal.confidence_score
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error entering position: {e}")
            return None
            
    def _check_exit_conditions(self, position: Dict, current_price: float, current_date: datetime) -> Tuple[bool, str]:
        """Check if position should be closed"""
        
        # Calculate P&L percentage
        if position['side'] == 'BUY':
            pnl_percent = (current_price - position['entry_price']) / position['entry_price'] * 100
        else:
            pnl_percent = (position['entry_price'] - current_price) / position['entry_price'] * 100
            
        # Stop loss
        if pnl_percent <= -self.config['stop_loss_percent']:
            return True, "STOP_LOSS"
            
        # Take profit
        if pnl_percent >= self.config['take_profit_percent']:
            return True, "TARGET"
            
        # Time exit
        hold_days = (current_date - position['entry_date']).days
        if hold_days >= self.config['max_hold_days']:
            return True, "TIME_EXIT"
            
        return False, ""
        
    def _close_position(self, position: Dict, exit_price: float, exit_date: datetime, exit_reason: str) -> BacktestTrade:
        """Close position and create trade record"""
        
        # Calculate P&L
        if position['side'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            
        # Subtract costs
        pnl -= self.config['commission_per_trade'] * 2  # Entry + Exit
        
        # Calculate percentage
        position_value = position['quantity'] * position['entry_price']
        pnl_percent = (pnl / position_value) * 100 if position_value > 0 else 0
        
        return BacktestTrade(
            entry_date=position['entry_date'],
            exit_date=exit_date,
            symbol=position['symbol'],
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_confidence=position['confidence'],
            exit_reason=exit_reason,
            trade_duration_days=(exit_date - position['entry_date']).days
        )
        
    def _calculate_results(self, symbol: str, initial: float, final: float, 
                         trades: List[BacktestTrade], equity_curve: List[Dict]) -> BacktestResults:
        """Calculate comprehensive results"""
        
        if not trades:
            return BacktestResults(
                start_date=datetime.now(), end_date=datetime.now(),
                initial_capital=initial, final_capital=initial,
                total_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
                max_drawdown=0.0, sharpe_ratio=0.0, trades=[]
            )
            
        # Basic metrics
        total_return = (final - initial) / initial * 100
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Max drawdown
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve).set_index('date')
            equity_df['peak'] = equity_df['portfolio_value'].cummax()
            equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak'] * 100
            max_drawdown = abs(equity_df['drawdown'].min())
            
            # Sharpe ratio (simplified)
            returns = equity_df['portfolio_value'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
            
        return BacktestResults(
            start_date=trades[0].entry_date,
            end_date=trades[-1].exit_date,
            initial_capital=initial,
            final_capital=final,
            total_return=total_return,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=trades
        )
        
    def run_portfolio_backtest(self, symbols: List[str], start_date: str, end_date: str) -> BacktestResults:
        """Run backtest on multiple symbols"""
        
        self.logger.info(f"🧪 Portfolio backtest: {len(symbols)} symbols")
        
        all_trades = []
        initial_capital = self.config['initial_capital']
        current_capital = initial_capital
        positions = {}  # symbol -> position
        portfolio_equity = []
        
        # Get all data
        all_data = {}
        for symbol in symbols:
            data = self.get_data(symbol, start_date, end_date)
            if not data.empty:
                all_data[symbol] = data
                
        if not all_data:
            raise ValueError("No data available for any symbols")
            
        # Find common trading days
        common_dates = None
        for data in all_data.values():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
                
        self.logger.info(f"📅 Trading period: {len(common_dates)} days")
        
        # Process each day
        for date in common_dates[100:]:  # Start after 100 bars
            daily_value = current_capital
            
            # Process each symbol
            for symbol in symbols:
                if symbol not in all_data:
                    continue
                    
                data = all_data[symbol]
                if date not in data.index:
                    continue
                    
                current_price = data.loc[date, 'close']
                historical_data = data.loc[:date]
                
                # Check exits for existing positions
                if symbol in positions:
                    should_exit, exit_reason = self._check_exit_conditions(
                        positions[symbol], current_price, date
                    )
                    
                    if should_exit:
                        trade = self._close_position(positions[symbol], current_price, date, exit_reason)
                        all_trades.append(trade)
                        current_capital += trade.pnl
                        del positions[symbol]
                        
                # Check for new entries
                if symbol not in positions and len(positions) < self.config['max_positions']:
                    signals = self._get_signals(symbol, historical_data)
                    
                    for signal in signals:
                        if self._should_trade(signal):
                            position = self._enter_position(signal, current_price, date, current_capital)
                            if position:
                                positions[symbol] = position
                            break
                            
                # Add position value
                if symbol in positions:
                    pos = positions[symbol]
                    if pos['side'] == 'BUY':
                        pos_pnl = (current_price - pos['entry_price']) * pos['quantity']
                    else:
                        pos_pnl = (pos['entry_price'] - current_price) * pos['quantity']
                    daily_value += pos_pnl
                    
            portfolio_equity.append({'date': date, 'portfolio_value': daily_value})
            
        # Close remaining positions
        for symbol, position in positions.items():
            if symbol in all_data:
                final_price = all_data[symbol].iloc[-1]['close']
                trade = self._close_position(position, final_price, common_dates[-1], "TIME_EXIT")
                all_trades.append(trade)
                current_capital += trade.pnl
                
        # Calculate results
        results = self._calculate_results(
            "PORTFOLIO", initial_capital, current_capital, all_trades, portfolio_equity
        )
        
        self.logger.info(f"✅ Portfolio backtest completed!")
        self.logger.info(f"   📊 Total Return: {results.total_return:.2f}%")
        self.logger.info(f"   🎯 Win Rate: {results.win_rate:.1f}%")
        self.logger.info(f"   📈 Trades: {results.total_trades}")
        
        return results
        
    def create_report(self, results: BacktestResults) -> str:
        """Generate comprehensive backtest report"""
        
        report = f"""
🧪 FYERS ALGORITHMIC TRADING BACKTEST REPORT
{'='*55}

📊 PERFORMANCE SUMMARY:
   Initial Capital: ₹{results.initial_capital:,.2f}
   Final Capital: ₹{results.final_capital:,.2f}
   Total Return: {results.total_return:+.2f}%
   
🎯 TRADE STATISTICS:
   Total Trades: {results.total_trades}
   Winning Trades: {results.winning_trades}
   Losing Trades: {results.losing_trades}
   Win Rate: {results.win_rate:.1f}%
   
💰 PROFIT & LOSS:
   Average Win: ₹{results.avg_win:+,.2f}
   Average Loss: ₹{results.avg_loss:+,.2f}
   Profit Factor: {results.profit_factor:.2f}
   
📉 RISK METRICS:
   Maximum Drawdown: {results.max_drawdown:.2f}%
   Sharpe Ratio: {results.sharpe_ratio:.2f}
   
📅 PERIOD:
   Start: {results.start_date.strftime('%Y-%m-%d')}
   End: {results.end_date.strftime('%Y-%m-%d')}
   Duration: {(results.end_date - results.start_date).days} days

{'='*55}

✅ READY FOR LIVE TRADING WITH FYERS API
💡 Configure FYERS credentials to start automated trading
🚀 Use fyers_live_portfolio.py for real money trading

"""
        return report
        
    def plot_results(self, results: BacktestResults, save_plot: bool = True):
        """Create backtest visualization"""
        
        if not results.trades:
            print("⚠️ No trades to plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade P&L
        trade_pnls = [t.pnl for t in results.trades]
        colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
        
        ax1.bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_title('Trade P&L', fontweight='bold')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('P&L (₹)')
        ax1.grid(True, alpha=0.3)
        
        # P&L Distribution
        ax2.hist(trade_pnls, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('P&L Distribution', fontweight='bold')
        ax2.set_xlabel('P&L (₹)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Win/Loss by Exit Reason
        exit_reasons = [t.exit_reason for t in results.trades]
        exit_pnl = {}
        
        for trade in results.trades:
            if trade.exit_reason not in exit_pnl:
                exit_pnl[trade.exit_reason] = []
            exit_pnl[trade.exit_reason].append(trade.pnl)
            
        reasons = list(exit_pnl.keys())
        avg_pnls = [np.mean(exit_pnl[reason]) for reason in reasons]
        colors = ['green' if pnl > 0 else 'red' for pnl in avg_pnls]
        
        ax3.bar(reasons, avg_pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_title('Average P&L by Exit Reason', fontweight='bold')
        ax3.set_ylabel('Average P&L (₹)')
        ax3.grid(True, alpha=0.3)
        
        # Trade Duration
        durations = [t.trade_duration_days for t in results.trades]
        ax4.hist(durations, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title('Trade Duration Distribution', fontweight='bold')
        ax4.set_xlabel('Duration (Days)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f'fyers_backtest_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: fyers_backtest_{timestamp}.png")
            
        plt.show()


def main():
    """Demo the backtesting system"""
    
    print("🧪 FYERS ALGORITHMIC TRADING BACKTESTER")
    print("="*48)
    
    try:
        # Initialize backtester
        backtester = FyersAlgoBacktester()
        
        # Configure test parameters
        symbol = "RELIANCE"
        start_date = "2023-01-01"
        end_date = "2024-12-31"
        
        print(f"\n📊 Testing strategy on {symbol}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Initial Capital: ₹{backtester.config['initial_capital']:,.2f}")
        
        # Run single symbol backtest
        results = backtester.run_backtest(symbol, start_date, end_date)
        
        # Generate report
        report = backtester.create_report(results)
        print(report)
        
        # Create plots
        backtester.plot_results(results)
        
        print("💡 TIP: Run portfolio backtest with multiple symbols:")
        print("   symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']")
        print("   results = backtester.run_portfolio_backtest(symbols, start_date, end_date)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have internet connection for data download")


if __name__ == "__main__":
    main()