#!/usr/bin/env python3
"""
📊 DETAILED BACKTESTING ANALYSIS REPORT 📊
================================================================================
Complete Trade-by-Trade Analysis of Real Market Data Backtesting
Every detail, every trade, every metric for full transparency
================================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from fyers_client import FyersClient

@dataclass
class DetailedTradeAnalysis:
    """Complete trade analysis with all details"""
    trade_id: int
    entry_datetime: datetime
    exit_datetime: datetime
    strategy: str
    side: str
    entry_price: float
    exit_price: float
    stop_loss: float
    target_price: float
    entry_rsi: float
    entry_volume_ratio: float
    entry_ema_9: float
    entry_ema_21: float
    entry_sma_50: float
    trade_duration_minutes: int
    points_captured: float
    quantity: int
    gross_pnl: float
    commission: float
    net_pnl: float
    result: str
    risk_reward_ratio: float
    capital_before: float
    capital_after: float
    running_drawdown: float
    mae: float  # Maximum Adverse Excursion
    mfe: float  # Maximum Favorable Excursion

class DetailedBacktestAnalyzer:
    """Detailed analysis of backtesting results"""
    
    def __init__(self):
        print("📊 DETAILED BACKTESTING ANALYSIS REPORT 📊")
        print("=" * 80)
        print("Complete Trade-by-Trade Analysis of Real Market Data Backtesting")
        print("Every detail, every trade, every metric for full transparency")
        print("=" * 80)
        
        # Initialize Fyers client
        try:
            self.fyers_client = FyersClient('fyers_config.json')
            print("✅ Connected to real Fyers account for data verification")
        except Exception as e:
            print(f"❌ Fyers connection error: {e}")
            return
            
        # Analysis parameters
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.risk_per_trade = 0.01
        self.commission = 20
        
        # Detailed tracking
        self.detailed_trades = []
        self.daily_pnl = {}
        self.monthly_pnl = {}
        self.equity_curve = []
        self.drawdown_curve = []
        
    def fetch_and_analyze_real_data(self, symbol: str = "NSE:NIFTY50-INDEX", days: int = 60):
        """Fetch real data and perform complete analysis"""
        
        print(f"\n📊 FETCHING REAL DATA FOR DETAILED ANALYSIS")
        print("=" * 60)
        print(f"🎯 Symbol: {symbol}")
        print(f"📅 Analysis Period: {days} days")
        print(f"🔌 Data Source: Real Fyers API")
        
        # Get real market data
        df = self.get_real_market_data(symbol, days)
        if df is None:
            return None
            
        # Add comprehensive technical indicators
        df = self.add_comprehensive_indicators(df)
        
        # Run detailed strategy analysis
        trades = self.detailed_momentum_strategy_analysis(df)
        
        # Generate comprehensive analysis
        analysis = self.generate_comprehensive_analysis(df, trades)
        
        return analysis
    
    def get_real_market_data(self, symbol: str, days: int):
        """Get real market data with verification"""
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data_request = {
                "symbol": symbol,
                "resolution": "5",
                "date_format": "1", 
                "range_from": start_date.strftime('%Y-%m-%d'),
                "range_to": end_date.strftime('%Y-%m-%d'),
                "cont_flag": "1"
            }
            
            response = self.fyers_client.fyers.history(data_request)
            
            if response and response.get('s') == 'ok' and 'candles' in response:
                candles = response['candles']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                print(f"✅ Real data fetched: {len(df):,} candles")
                print(f"📅 Period: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
                print(f"📈 Price range: Rs.{df['low'].min():.2f} - Rs.{df['high'].max():.2f}")
                
                return df
            else:
                print(f"❌ Data fetch failed: {response}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def add_comprehensive_indicators(self, df):
        """Add all technical indicators used in analysis"""
        
        print(f"🔧 Calculating comprehensive technical indicators...")
        
        # Moving averages
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Momentum indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price levels
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        # Volatility
        df['atr'] = self.calculate_atr(df)
        
        # Market structure
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        
        print(f"✅ Technical indicators calculated")
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def detailed_momentum_strategy_analysis(self, df):
        """Detailed momentum strategy with complete trade analysis"""
        
        print(f"\n🎯 DETAILED MOMENTUM STRATEGY ANALYSIS")
        print("=" * 50)
        
        trades = []
        trade_id = 1
        
        for i in range(60, len(df) - 20):
            current = df.iloc[i]
            
            # Only trade during market hours
            hour = current['datetime'].time().hour
            if not (9 <= hour <= 14):
                continue
                
            # Check for valid indicators
            if (pd.notna(current['ema_9']) and pd.notna(current['ema_21']) and 
                pd.notna(current['sma_50']) and pd.notna(current['rsi'])):
                
                # LONG momentum signal
                if (current['close'] > current['ema_9'] > current['ema_21'] and
                    current['close'] > current['sma_50'] * 1.001 and
                    current['close'] > current['high_20'] * 0.998 and
                    50 < current['rsi'] < 75 and
                    current['volume_ratio'] > 1.5 and
                    current['close'] > current['open']):
                    
                    trade = self.execute_detailed_trade(
                        df, i, trade_id, 'BUY', 'momentum_long',
                        current, 1.2, 3.0  # stop_pct, target_pct
                    )
                    if trade:
                        trades.append(trade)
                        trade_id += 1
                
                # SHORT momentum signal  
                elif (current['close'] < current['ema_9'] < current['ema_21'] and
                      current['close'] < current['sma_50'] * 0.999 and
                      current['close'] < current['low_20'] * 1.002 and
                      25 < current['rsi'] < 50 and
                      current['volume_ratio'] > 1.5 and
                      current['close'] < current['open']):
                    
                    trade = self.execute_detailed_trade(
                        df, i, trade_id, 'SELL', 'momentum_short',
                        current, 1.2, 3.0  # stop_pct, target_pct
                    )
                    if trade:
                        trades.append(trade)
                        trade_id += 1
        
        print(f"✅ Detailed analysis completed: {len(trades)} trades")
        return trades
    
    def execute_detailed_trade(self, df, entry_idx, trade_id, side, strategy, 
                              entry_candle, stop_pct, target_pct):
        """Execute trade with complete detailed tracking"""
        
        entry_price = entry_candle['close']
        
        # Calculate stop loss and target
        if side == 'BUY':
            stop_loss = entry_price * (1 - stop_pct / 100)
            target_price = entry_price * (1 + target_pct / 100)
        else:
            stop_loss = entry_price * (1 + stop_pct / 100)
            target_price = entry_price * (1 - target_pct / 100)
        
        # Track MAE and MFE during trade
        mae = 0  # Maximum Adverse Excursion
        mfe = 0  # Maximum Favorable Excursion
        
        # Look for exit
        for i in range(entry_idx + 1, min(entry_idx + 80, len(df))):
            candle = df.iloc[i]
            
            # Update MAE and MFE
            if side == 'BUY':
                adverse = entry_price - candle['low']
                favorable = candle['high'] - entry_price
                mae = max(mae, adverse)
                mfe = max(mfe, favorable)
                
                # Check exit conditions
                if candle['high'] >= target_price:
                    return self.create_detailed_trade_analysis(
                        trade_id, entry_candle, candle, side, strategy,
                        entry_price, target_price, stop_loss, target_price,
                        'WIN', mae, mfe
                    )
                elif candle['low'] <= stop_loss:
                    return self.create_detailed_trade_analysis(
                        trade_id, entry_candle, candle, side, strategy,
                        entry_price, stop_loss, stop_loss, target_price,
                        'LOSS', mae, mfe
                    )
            else:  # SELL
                adverse = candle['high'] - entry_price
                favorable = entry_price - candle['low']
                mae = max(mae, adverse)
                mfe = max(mfe, favorable)
                
                # Check exit conditions
                if candle['low'] <= target_price:
                    return self.create_detailed_trade_analysis(
                        trade_id, entry_candle, candle, side, strategy,
                        entry_price, target_price, stop_loss, target_price,
                        'WIN', mae, mfe
                    )
                elif candle['high'] >= stop_loss:
                    return self.create_detailed_trade_analysis(
                        trade_id, entry_candle, candle, side, strategy,
                        entry_price, stop_loss, stop_loss, target_price,
                        'LOSS', mae, mfe
                    )
        
        return None
    
    def create_detailed_trade_analysis(self, trade_id, entry_candle, exit_candle,
                                     side, strategy, entry_price, exit_price,
                                     stop_loss, target_price, result, mae, mfe):
        """Create comprehensive trade analysis"""
        
        # Calculate trade metrics
        if side == 'BUY':
            points_captured = exit_price - entry_price
        else:
            points_captured = entry_price - exit_price
        
        # Position sizing
        risk_amount = self.current_capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        quantity = max(1, int(risk_amount / stop_distance)) if stop_distance > 0 else 1
        
        # P&L calculation
        gross_pnl = points_captured * quantity
        net_pnl = gross_pnl - self.commission
        
        # Risk-reward ratio
        if side == 'BUY':
            risk = entry_price - stop_loss
            reward = target_price - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - target_price
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Update capital and drawdown
        capital_before = self.current_capital
        self.current_capital += net_pnl
        capital_after = self.current_capital
        
        # Calculate running drawdown
        if not hasattr(self, 'peak_capital'):
            self.peak_capital = self.initial_capital
        
        if capital_after > self.peak_capital:
            self.peak_capital = capital_after
        
        running_drawdown = (self.peak_capital - capital_after) / self.peak_capital * 100
        
        # Trade duration
        duration = (exit_candle['datetime'] - entry_candle['datetime']).total_seconds() / 60
        
        return DetailedTradeAnalysis(
            trade_id=trade_id,
            entry_datetime=entry_candle['datetime'],
            exit_datetime=exit_candle['datetime'],
            strategy=strategy,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            target_price=target_price,
            entry_rsi=entry_candle.get('rsi', 0),
            entry_volume_ratio=entry_candle.get('volume_ratio', 0),
            entry_ema_9=entry_candle.get('ema_9', 0),
            entry_ema_21=entry_candle.get('ema_21', 0),
            entry_sma_50=entry_candle.get('sma_50', 0),
            trade_duration_minutes=int(duration),
            points_captured=points_captured,
            quantity=quantity,
            gross_pnl=gross_pnl,
            commission=self.commission,
            net_pnl=net_pnl,
            result=result,
            risk_reward_ratio=risk_reward_ratio,
            capital_before=capital_before,
            capital_after=capital_after,
            running_drawdown=running_drawdown,
            mae=mae,
            mfe=mfe
        )
    
    def generate_comprehensive_analysis(self, df, trades):
        """Generate complete comprehensive analysis"""
        
        print(f"\n📊 COMPREHENSIVE BACKTESTING ANALYSIS REPORT 📊")
        print("=" * 80)
        
        if not trades:
            print("❌ No trades to analyze")
            return None
        
        # Convert trades to DataFrame for analysis
        trade_data = [asdict(trade) for trade in trades]
        trade_df = pd.DataFrame(trade_data)
        
        # Basic performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.result == 'WIN'])
        losing_trades = len([t for t in trades if t.result == 'LOSS'])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum(t.net_pnl for t in trades)
        roi = (total_pnl / self.initial_capital) * 100
        
        avg_win = np.mean([t.net_pnl for t in trades if t.result == 'WIN']) if winning_trades > 0 else 0
        avg_loss = np.mean([t.net_pnl for t in trades if t.result == 'LOSS']) if losing_trades > 0 else 0
        
        # Advanced metrics
        profit_factor = abs(sum(t.net_pnl for t in trades if t.result == 'WIN') / 
                           sum(t.net_pnl for t in trades if t.result == 'LOSS')) if losing_trades > 0 else float('inf')
        
        max_drawdown = max([t.running_drawdown for t in trades])
        avg_trade_duration = np.mean([t.trade_duration_minutes for t in trades])
        
        # Print detailed analysis
        print(f"\n📈 OVERALL PERFORMANCE SUMMARY:")
        print(f"   💰 Initial Capital:      Rs.{self.initial_capital:10,.0f}")
        print(f"   🎯 Final Capital:        Rs.{self.current_capital:10,.0f}")
        print(f"   🚀 Net Profit:           Rs.{total_pnl:+9,.0f}")
        print(f"   📊 ROI:                  {roi:+8.1f}%")
        print(f"   ⚡ Total Trades:          {total_trades:10d}")
        print(f"   🏆 Win Rate:             {win_rate:9.1f}%")
        print(f"   ✅ Winning Trades:       {winning_trades:10d}")
        print(f"   ❌ Losing Trades:        {losing_trades:10d}")
        print(f"   💚 Average Win:          Rs.{avg_win:+8,.0f}")
        print(f"   💔 Average Loss:         Rs.{avg_loss:+8,.0f}")
        print(f"   📊 Profit Factor:        {profit_factor:8.2f}")
        print(f"   📉 Max Drawdown:         {max_drawdown:8.1f}%")
        print(f"   ⏱️ Avg Trade Duration:   {avg_trade_duration:8.0f} minutes")
        
        # Detailed trade-by-trade analysis
        print(f"\n📋 DETAILED TRADE-BY-TRADE ANALYSIS:")
        print("-" * 120)
        print(f"{'ID':<3} {'Date':<12} {'Time':<5} {'Strategy':<15} {'Side':<4} {'Entry':<7} {'Exit':<7} {'Points':<6} {'Duration':<8} {'P&L':<8} {'Result':<6} {'Capital':<10}")
        print("-" * 120)
        
        for trade in trades:
            print(f"{trade.trade_id:<3d} "
                  f"{trade.entry_datetime.strftime('%m-%d'):<12} "
                  f"{trade.entry_datetime.strftime('%H:%M'):<5} "
                  f"{trade.strategy:<15} "
                  f"{trade.side:<4} "
                  f"{trade.entry_price:<7.0f} "
                  f"{trade.exit_price:<7.0f} "
                  f"{trade.points_captured:+6.0f} "
                  f"{trade.trade_duration_minutes:<8d} "
                  f"{trade.net_pnl:+8,.0f} "
                  f"{trade.result:<6} "
                  f"{trade.capital_after:<10,.0f}")
        
        # Monthly breakdown
        monthly_analysis = self.analyze_monthly_performance(trades)
        
        # Risk analysis
        risk_analysis = self.analyze_risk_metrics(trades)
        
        # Strategy breakdown
        strategy_analysis = self.analyze_strategy_performance(trades)
        
        # Market timing analysis
        timing_analysis = self.analyze_market_timing(trades)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'roi': roi,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'monthly_analysis': monthly_analysis,
            'risk_analysis': risk_analysis,
            'strategy_analysis': strategy_analysis,
            'timing_analysis': timing_analysis
        }
    
    def analyze_monthly_performance(self, trades):
        """Analyze monthly performance breakdown"""
        
        print(f"\n📅 MONTHLY PERFORMANCE BREAKDOWN:")
        print("-" * 50)
        
        monthly_stats = {}
        
        for trade in trades:
            month_key = trade.entry_datetime.strftime('%Y-%m')
            if month_key not in monthly_stats:
                monthly_stats[month_key] = {
                    'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0
                }
            
            monthly_stats[month_key]['trades'] += 1
            monthly_stats[month_key]['pnl'] += trade.net_pnl
            if trade.result == 'WIN':
                monthly_stats[month_key]['wins'] += 1
            else:
                monthly_stats[month_key]['losses'] += 1
        
        for month, stats in monthly_stats.items():
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            print(f"   {month}: {stats['trades']:2d} trades, "
                  f"{win_rate:5.1f}% win rate, Rs.{stats['pnl']:+8,.0f}")
        
        return monthly_stats
    
    def analyze_risk_metrics(self, trades):
        """Detailed risk analysis"""
        
        print(f"\n⚠️ RISK ANALYSIS:")
        print("-" * 30)
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in trades:
            if trade.result == 'WIN':
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Drawdown analysis
        drawdowns = [t.running_drawdown for t in trades]
        max_dd = max(drawdowns) if drawdowns else 0
        avg_dd = np.mean(drawdowns) if drawdowns else 0
        
        # MAE/MFE analysis
        avg_mae = np.mean([t.mae for t in trades])
        avg_mfe = np.mean([t.mfe for t in trades])
        
        print(f"   🎯 Max Consecutive Wins:     {max_consecutive_wins}")
        print(f"   💔 Max Consecutive Losses:   {max_consecutive_losses}")
        print(f"   📉 Maximum Drawdown:         {max_dd:.1f}%")
        print(f"   📊 Average Drawdown:         {avg_dd:.1f}%")
        print(f"   ⬇️ Average MAE:              {avg_mae:.0f} points")
        print(f"   ⬆️ Average MFE:              {avg_mfe:.0f} points")
        
        return {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe
        }
    
    def analyze_strategy_performance(self, trades):
        """Strategy-specific performance analysis"""
        
        print(f"\n🎯 STRATEGY PERFORMANCE ANALYSIS:")
        print("-" * 40)
        
        strategy_stats = {}
        
        for trade in trades:
            if trade.strategy not in strategy_stats:
                strategy_stats[trade.strategy] = {
                    'trades': 0, 'wins': 0, 'pnl': 0, 'durations': []
                }
            
            strategy_stats[trade.strategy]['trades'] += 1
            strategy_stats[trade.strategy]['pnl'] += trade.net_pnl
            strategy_stats[trade.strategy]['durations'].append(trade.trade_duration_minutes)
            
            if trade.result == 'WIN':
                strategy_stats[trade.strategy]['wins'] += 1
        
        for strategy, stats in strategy_stats.items():
            win_rate = (stats['wins'] / stats['trades']) * 100
            avg_duration = np.mean(stats['durations'])
            
            print(f"   {strategy}:")
            print(f"      Trades: {stats['trades']:2d}, Win Rate: {win_rate:5.1f}%")
            print(f"      P&L: Rs.{stats['pnl']:+8,.0f}, Avg Duration: {avg_duration:.0f}min")
        
        return strategy_stats
    
    def analyze_market_timing(self, trades):
        """Market timing analysis"""
        
        print(f"\n⏰ MARKET TIMING ANALYSIS:")
        print("-" * 30)
        
        # Hourly performance
        hourly_stats = {}
        
        for trade in trades:
            hour = trade.entry_datetime.hour
            if hour not in hourly_stats:
                hourly_stats[hour] = {'trades': 0, 'wins': 0, 'pnl': 0}
            
            hourly_stats[hour]['trades'] += 1
            hourly_stats[hour]['pnl'] += trade.net_pnl
            if trade.result == 'WIN':
                hourly_stats[hour]['wins'] += 1
        
        print(f"   📊 Performance by Hour:")
        for hour in sorted(hourly_stats.keys()):
            stats = hourly_stats[hour]
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            print(f"      {hour:02d}:00 - {stats['trades']:2d} trades, "
                  f"{win_rate:5.1f}% win rate, Rs.{stats['pnl']:+6,.0f}")
        
        return hourly_stats

if __name__ == "__main__":
    print("📊 Starting Detailed Backtesting Analysis...")
    
    try:
        analyzer = DetailedBacktestAnalyzer()
        
        # Run comprehensive analysis
        analysis = analyzer.fetch_and_analyze_real_data(
            symbol="NSE:NIFTY50-INDEX",
            days=60
        )
        
        if analysis:
            print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE")
            print(f"📊 {analysis['total_trades']} trades analyzed in detail")
            print(f"💰 {analysis['roi']:+.1f}% ROI with detailed breakdowns")
            print(f"📋 All trade details, risk metrics, and timing analysis provided")
        else:
            print(f"\n❌ Analysis could not be completed")
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()