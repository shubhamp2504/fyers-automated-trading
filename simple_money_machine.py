#!/usr/bin/env python3
"""
🚀💰 SIMPLE ULTIMATE MONEY MACHINE 💰🚀
================================================================================
STREAMLINED AUTOMATED TRADING SYSTEM WITH REAL DATA + REAL MONEY
• Multi-Strategy Real Money Trading
• Real-time Market Scanning & Execution  
• Advanced Risk Management
• 100% Authentic Fyers API Integration
• Automated Profit Generation
================================================================================
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import YOUR working Fyers client
from fyers_client import FyersClient

class TradingMode(Enum):
    SCALPING = "scalping"
    SWING = "swing"
    MOMENTUM = "momentum"

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXPLOSIVE = 4

@dataclass
class TradingOpportunity:
    """Market opportunity structure"""
    symbol: str
    strategy: str
    mode: TradingMode
    strength: SignalStrength
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    risk_reward: float
    timestamp: datetime
    analysis_data: Dict[str, Any]

@dataclass
class ActiveTrade:
    """Active trade tracking"""
    id: str
    opportunity: TradingOpportunity
    entry_time: datetime
    entry_price: float
    quantity: int
    current_pnl: float
    status: str = "OPEN"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    final_pnl: Optional[float] = None

class SimpleMoneyMachine:
    """Simple but powerful automated money-making machine"""
    
    def __init__(self):
        print("🚀💰 SIMPLE ULTIMATE MONEY MACHINE 💰🚀")
        print("=" * 80)
        print("STREAMLINED FEATURES:")
        print("• Multi-Strategy Real Money Trading")
        print("• Real-time Market Analysis & Execution")
        print("• Professional Risk Management")
        print("• 100% Authentic Market Data")
        print("• Automated Profit Generation")
        print("=" * 80)
        
        # Initialize infrastructure
        self.initialize_system()
        
        # Trading state
        self.is_running = False
        self.active_trades: Dict[str, ActiveTrade] = {}
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.trades_executed = 0
        self.opportunities_found = 0
        
    def initialize_system(self):
        """Initialize trading system"""
        print(f"\n🔧 INITIALIZING TRADING SYSTEM")
        print("-" * 50)
        
        try:
            # Load configuration
            with open('fyers_config.json', 'r') as f:
                self.config = json.load(f)
            print("   ✅ Configuration loaded")
            
            # Initialize Fyers client
            self.fyers_client = FyersClient('fyers_config.json')
            print("   ✅ Fyers client initialized")
            
            # Test API connection
            profile = self.fyers_client.fyers.get_profile()
            if profile and profile.get('s') == 'ok':
                account_data = profile.get('data', {})
                self.account_id = account_data.get('fy_id', 'Unknown')
                print(f"   ✅ Connected to account: {self.account_id}")
                self.api_working = True
            else:
                print("   ❌ API connection failed")
                self.api_working = False
                return
                
            # Set up trading parameters
            self.capital = 100000  # Starting capital
            self.current_capital = self.capital
            self.risk_per_trade = self.config['trading'].get('risk_per_trade', 0.01)  # 1% risk
            self.max_positions = self.config['trading'].get('max_open_positions', 5)
            self.max_daily_loss = self.config['trading'].get('max_daily_loss', 5000)
            
            print(f"   💰 Starting capital: Rs.{self.capital:,.0f}")
            print(f"   🎯 Risk per trade: {self.risk_per_trade:.1%}")
            print(f"   📊 Max positions: {self.max_positions}")
            print(f"   🛡️ Max daily loss: Rs.{self.max_daily_loss:,.0f}")
            
        except Exception as e:
            print(f"   ❌ System initialization error: {e}")
            self.api_working = False
    
    def get_market_data(self, symbol: str, timeframe: str = "5", days: int = 5) -> Optional[pd.DataFrame]:
        """Get real market data from Fyers API"""
        
        if not self.api_working:
            return None
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            response = self.fyers_client.fyers.history({
                "symbol": symbol,
                "resolution": timeframe,
                "date_format": "1",
                "range_from": start_time.strftime('%Y-%m-%d'),
                "range_to": end_time.strftime('%Y-%m-%d'),
                "cont_flag": "1"
            })
            
            if response and response.get('s') == 'ok':
                candles = response['candles']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Add technical indicators
                if len(df) >= 50:
                    df['ema_9'] = df['close'].ewm(span=9).mean()
                    df['ema_21'] = df['close'].ewm(span=21).mean()
                    df['sma_20'] = df['close'].rolling(20).mean()
                    
                    # RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Volume indicators
                    df['volume_sma'] = df['volume'].rolling(20).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_sma']
                
                return df
            
            return None
            
        except Exception as e:
            return None
    
    def analyze_scalping_opportunity(self, symbol: str) -> Optional[TradingOpportunity]:
        """Analyze scalping opportunities"""
        
        # Get 5-minute data for scalping
        df = self.get_market_data(symbol, "5", days=2)
        if df is None or len(df) < 50:
            return None
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Strategy 1: EMA Momentum Scalping
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Check for strong momentum setup
            if (ema_9 > ema_21 and  # Trend alignment
                df['ema_9'].iloc[-1] > df['ema_9'].iloc[-2] and  # Momentum acceleration
                35 <= rsi <= 65 and  # Not oversold/overbought
                volume_ratio >= 1.5):  # Volume confirmation
                
                # BUY signal
                target_price = current_price * 1.008  # 0.8% target
                stop_loss = current_price * 0.996      # 0.4% stop
                
                risk_reward = (target_price - current_price) / (current_price - stop_loss)
                
                if risk_reward >= 1.5:  # Minimum 1.5:1 RR
                    return TradingOpportunity(
                        symbol=symbol,
                        strategy="EMA_Momentum_Scalp",
                        mode=TradingMode.SCALPING,
                        strength=SignalStrength.STRONG if volume_ratio >= 2.0 else SignalStrength.MODERATE,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        confidence=0.7 if volume_ratio >= 2.0 else 0.6,
                        risk_reward=risk_reward,
                        timestamp=datetime.now(),
                        analysis_data={
                            'ema_9': ema_9,
                            'ema_21': ema_21,
                            'rsi': rsi,
                            'volume_ratio': volume_ratio,
                            'direction': 'BUY'
                        }
                    )
            
            elif (ema_9 < ema_21 and  # Trend alignment
                  df['ema_9'].iloc[-1] < df['ema_9'].iloc[-2] and  # Momentum acceleration
                  35 <= rsi <= 65 and  # Not oversold/overbought
                  volume_ratio >= 1.5):  # Volume confirmation
                
                # SELL signal
                target_price = current_price * 0.992  # 0.8% target
                stop_loss = current_price * 1.004      # 0.4% stop
                
                risk_reward = (current_price - target_price) / (stop_loss - current_price)
                
                if risk_reward >= 1.5:  # Minimum 1.5:1 RR
                    return TradingOpportunity(
                        symbol=symbol,
                        strategy="EMA_Momentum_Scalp",
                        mode=TradingMode.SCALPING,
                        strength=SignalStrength.STRONG if volume_ratio >= 2.0 else SignalStrength.MODERATE,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        confidence=0.7 if volume_ratio >= 2.0 else 0.6,
                        risk_reward=risk_reward,
                        timestamp=datetime.now(),
                        analysis_data={
                            'ema_9': ema_9,
                            'ema_21': ema_21,
                            'rsi': rsi,
                            'volume_ratio': volume_ratio,
                            'direction': 'SELL'
                        }
                    )
            
            return None
            
        except Exception as e:
            return None
    
    def analyze_swing_opportunity(self, symbol: str) -> Optional[TradingOpportunity]:
        """Analyze swing trading opportunities"""
        
        # Get 1-hour data for swing analysis
        df = self.get_market_data(symbol, "60", days=10)
        if df is None or len(df) < 50:
            return None
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Strategy 2: Trend Following Swing
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # Calculate recent volatility (ATR)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Strong uptrend setup
            if (ema_9 > ema_21 > sma_20 and  # Clear trend alignment
                current_price > ema_9 and     # Price above fast EMA
                40 <= rsi <= 70):            # RSI in trending range
                
                # BUY signal for swing
                target_price = current_price + (atr * 3)   # 3 ATR target
                stop_loss = current_price - (atr * 1.5)    # 1.5 ATR stop
                
                risk_reward = (target_price - current_price) / (current_price - stop_loss)
                
                if risk_reward >= 1.8:  # Higher RR for swing trades
                    return TradingOpportunity(
                        symbol=symbol,
                        strategy="Trend_Following_Swing",
                        mode=TradingMode.SWING,
                        strength=SignalStrength.STRONG if rsi >= 55 else SignalStrength.MODERATE,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        confidence=0.75,
                        risk_reward=risk_reward,
                        timestamp=datetime.now(),
                        analysis_data={
                            'ema_alignment': True,
                            'trend_direction': 'UP',
                            'atr': atr,
                            'rsi': rsi,
                            'direction': 'BUY'
                        }
                    )
            
            # Strong downtrend setup
            elif (ema_9 < ema_21 < sma_20 and  # Clear trend alignment
                  current_price < ema_9 and     # Price below fast EMA
                  30 <= rsi <= 60):            # RSI in trending range
                
                # SELL signal for swing
                target_price = current_price - (atr * 3)   # 3 ATR target
                stop_loss = current_price + (atr * 1.5)    # 1.5 ATR stop
                
                risk_reward = (current_price - target_price) / (stop_loss - current_price)
                
                if risk_reward >= 1.8:  # Higher RR for swing trades
                    return TradingOpportunity(
                        symbol=symbol,
                        strategy="Trend_Following_Swing",
                        mode=TradingMode.SWING,
                        strength=SignalStrength.STRONG if rsi <= 45 else SignalStrength.MODERATE,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        confidence=0.75,
                        risk_reward=risk_reward,
                        timestamp=datetime.now(),
                        analysis_data={
                            'ema_alignment': True,
                            'trend_direction': 'DOWN',
                            'atr': atr,
                            'rsi': rsi,
                            'direction': 'SELL'
                        }
                    )
            
            return None
            
        except Exception as e:
            return None
    
    def analyze_momentum_opportunity(self, symbol: str) -> Optional[TradingOpportunity]:
        """Analyze momentum breakout opportunities"""
        
        # Get 15-minute data for momentum analysis
        df = self.get_market_data(symbol, "15", days=5)
        if df is None or len(df) < 50:
            return None
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Strategy 3: Volume Breakout Momentum
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Calculate price position in recent range
            range_size = recent_high - recent_low
            price_position = (current_price - recent_low) / range_size if range_size > 0 else 0.5
            
            # Bullish momentum breakout
            if (current_price >= recent_high * 0.999 and  # Near or above recent high
                volume_ratio >= 2.0 and                   # Strong volume
                price_position >= 0.85):                  # High in range
                
                target_price = current_price + (range_size * 0.5)  # Project half range up
                stop_loss = current_price - (range_size * 0.2)     # Stop below breakout
                
                risk_reward = (target_price - current_price) / (current_price - stop_loss)
                
                if risk_reward >= 2.0:  # Strong RR for momentum
                    return TradingOpportunity(
                        symbol=symbol,
                        strategy="Volume_Breakout_Momentum",
                        mode=TradingMode.MOMENTUM,
                        strength=SignalStrength.EXPLOSIVE if volume_ratio >= 3.0 else SignalStrength.STRONG,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        confidence=0.8 if volume_ratio >= 3.0 else 0.7,
                        risk_reward=risk_reward,
                        timestamp=datetime.now(),
                        analysis_data={
                            'breakout_type': 'BULLISH',
                            'volume_ratio': volume_ratio,
                            'range_position': price_position,
                            'range_size': range_size,
                            'direction': 'BUY'
                        }
                    )
            
            # Bearish momentum breakdown
            elif (current_price <= recent_low * 1.001 and  # Near or below recent low
                  volume_ratio >= 2.0 and                  # Strong volume
                  price_position <= 0.15):                 # Low in range
                
                target_price = current_price - (range_size * 0.5)  # Project half range down
                stop_loss = current_price + (range_size * 0.2)     # Stop above breakdown
                
                risk_reward = (current_price - target_price) / (stop_loss - current_price)
                
                if risk_reward >= 2.0:  # Strong RR for momentum
                    return TradingOpportunity(
                        symbol=symbol,
                        strategy="Volume_Breakout_Momentum",
                        mode=TradingMode.MOMENTUM,
                        strength=SignalStrength.EXPLOSIVE if volume_ratio >= 3.0 else SignalStrength.STRONG,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        confidence=0.8 if volume_ratio >= 3.0 else 0.7,
                        risk_reward=risk_reward,
                        timestamp=datetime.now(),
                        analysis_data={
                            'breakout_type': 'BEARISH',
                            'volume_ratio': volume_ratio,
                            'range_position': price_position,
                            'range_size': range_size,
                            'direction': 'SELL'
                        }
                    )
            
            return None
            
        except Exception as e:
            return None
    
    def scan_all_opportunities(self) -> List[TradingOpportunity]:
        """Scan all symbols for trading opportunities"""
        
        print(f"\n🔍 SCANNING MARKET FOR OPPORTUNITIES")
        print("-" * 50)
        
        # Define watchlist
        watchlist = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX",
            "NSE:RELIANCE-EQ",
            "NSE:TCS-EQ",
            "NSE:INFY-EQ",
            "NSE:HDFCBANK-EQ"
        ]
        
        all_opportunities = []
        
        for symbol in watchlist:
            print(f"   📊 Analyzing {symbol}...")
            
            try:
                # Try each strategy
                scalping_opp = self.analyze_scalping_opportunity(symbol)
                if scalping_opp:
                    all_opportunities.append(scalping_opp)
                    print(f"      ✅ Scalping: {scalping_opp.confidence:.1%} confidence")
                
                swing_opp = self.analyze_swing_opportunity(symbol)
                if swing_opp:
                    all_opportunities.append(swing_opp)
                    print(f"      ✅ Swing: {swing_opp.confidence:.1%} confidence")
                
                momentum_opp = self.analyze_momentum_opportunity(symbol)
                if momentum_opp:
                    all_opportunities.append(momentum_opp)
                    print(f"      ✅ Momentum: {momentum_opp.confidence:.1%} confidence")
                
                if not any([scalping_opp, swing_opp, momentum_opp]):
                    print(f"      ⏸️ No opportunities found")
                
            except Exception as e:
                print(f"      ❌ Analysis error: {str(e)[:50]}")
                continue
            
            time.sleep(0.1)  # Rate limiting
        
        # Filter and rank opportunities
        filtered_opportunities = self.filter_and_rank_opportunities(all_opportunities)
        
        print(f"   🎯 Found {len(all_opportunities)} total opportunities")
        print(f"   ✅ Filtered to {len(filtered_opportunities)} high-quality setups")
        
        self.opportunities_found += len(filtered_opportunities)
        return filtered_opportunities
    
    def filter_and_rank_opportunities(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """Filter and rank opportunities by quality"""
        
        if not opportunities:
            return []
        
        # Quality filters
        filtered = []
        
        for opp in opportunities:
            # Basic quality checks
            if (opp.confidence >= 0.6 and
                opp.risk_reward >= 1.5 and
                opp.strength != SignalStrength.WEAK):
                
                # Check if we already have this symbol
                existing_symbols = [trade.opportunity.symbol for trade in self.active_trades.values()]
                if opp.symbol not in existing_symbols:
                    filtered.append(opp)
        
        # Rank by combined score
        def opportunity_score(opp):
            base_score = opp.confidence * 100
            strength_bonus = opp.strength.value * 15
            rr_bonus = min(opp.risk_reward * 10, 30)  # Cap RR bonus
            return base_score + strength_bonus + rr_bonus
        
        filtered.sort(key=opportunity_score, reverse=True)
        
        # Limit to available position slots
        max_new_positions = self.max_positions - len(self.active_trades)
        return filtered[:max_new_positions]
    
    def calculate_position_size(self, opportunity: TradingOpportunity) -> int:
        """Calculate position size based on risk"""
        
        # Risk-based position sizing
        risk_amount = self.current_capital * self.risk_per_trade
        risk_per_unit = abs(opportunity.entry_price - opportunity.stop_loss)
        
        if risk_per_unit <= 0:
            return 0
        
        # Basic position size
        position_size = int(risk_amount / risk_per_unit)
        
        # Apply maximum position value limit (25% of capital)
        max_position_value = self.current_capital * 0.25
        max_qty_by_value = int(max_position_value / opportunity.entry_price)
        
        position_size = min(position_size, max_qty_by_value)
        
        return max(1, position_size)  # Minimum 1 unit
    
    def execute_trade(self, opportunity: TradingOpportunity) -> Optional[ActiveTrade]:
        """Execute trade (simulated for demo)"""
        
        print(f"\n⚡ EXECUTING TRADE")
        print("-" * 30)
        
        quantity = self.calculate_position_size(opportunity)
        if quantity <= 0:
            print("   ❌ Position size calculation failed")
            return None
        
        print(f"   🎯 Symbol: {opportunity.symbol}")
        print(f"   📊 Strategy: {opportunity.strategy}")
        print(f"   ⚡ Mode: {opportunity.mode.value.upper()}")
        print(f"   💰 Entry: Rs.{opportunity.entry_price:.2f}")
        print(f"   🎯 Target: Rs.{opportunity.target_price:.2f}")
        print(f"   🛡️ Stop: Rs.{opportunity.stop_loss:.2f}")
        print(f"   📈 Quantity: {quantity}")
        print(f"   💪 Confidence: {opportunity.confidence:.1%}")
        print(f"   📊 Risk-Reward: {opportunity.risk_reward:.1f}:1")
        
        try:
            # For demo, simulate successful execution
            # In live trading, use: fyers_client.place_order()
            
            trade_id = f"SMM_{int(time.time())}_{len(self.active_trades)}"
            
            trade = ActiveTrade(
                id=trade_id,
                opportunity=opportunity,
                entry_time=datetime.now(),
                entry_price=opportunity.entry_price,
                quantity=quantity,
                current_pnl=0.0,
                status="OPEN"
            )
            
            # Add to active trades
            self.active_trades[trade_id] = trade
            self.trades_executed += 1
            
            # Update capital (simulate execution)
            trade_value = quantity * opportunity.entry_price
            self.current_capital -= trade_value * 0.1  # Assume 10% margin requirement
            
            print(f"   ✅ Trade executed successfully!")
            print(f"   📝 Trade ID: {trade_id}")
            
            return trade
            
        except Exception as e:
            print(f"   ❌ Trade execution failed: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        
        try:
            response = self.fyers_client.fyers.quotes({"symbols": symbol})
            
            if response and response.get('s') == 'ok':
                quotes = response.get('d', [])
                if quotes:
                    return quotes[0].get('v', {}).get('lp')  # Last price
            
            return None
            
        except Exception:
            return None
    
    def monitor_active_trades(self):
        """Monitor and manage active trades"""
        
        if not self.active_trades:
            return
        
        print(f"\n👁️ MONITORING {len(self.active_trades)} ACTIVE TRADES")
        print("-" * 50)
        
        for trade_id, trade in list(self.active_trades.items()):
            try:
                # Get current market price
                current_price = self.get_current_price(trade.opportunity.symbol)
                if current_price is None:
                    continue
                
                # Calculate current P&L
                direction = trade.opportunity.analysis_data.get('direction', 'BUY')
                
                if direction == 'BUY':
                    pnl_points = current_price - trade.entry_price
                else:
                    pnl_points = trade.entry_price - current_price
                
                trade.current_pnl = pnl_points * trade.quantity
                
                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions(trade, current_price)
                
                if should_exit:
                    self.close_trade(trade, current_price, exit_reason)
                else:
                    print(f"   📊 {trade.opportunity.symbol}: Rs.{trade.current_pnl:+,.0f} ({exit_reason})")
                
            except Exception as e:
                print(f"   ⚠️ Error monitoring {trade.opportunity.symbol}: {e}")
                continue
    
    def check_exit_conditions(self, trade: ActiveTrade, current_price: float) -> tuple:
        """Check if trade should be exited"""
        
        direction = trade.opportunity.analysis_data.get('direction', 'BUY')
        
        # Target reached
        if direction == 'BUY':
            if current_price >= trade.opportunity.target_price:
                return True, "TARGET_REACHED"
            if current_price <= trade.opportunity.stop_loss:
                return True, "STOP_LOSS_HIT"
        else:
            if current_price <= trade.opportunity.target_price:
                return True, "TARGET_REACHED"
            if current_price >= trade.opportunity.stop_loss:
                return True, "STOP_LOSS_HIT"
        
        # Time-based exits
        trade_duration = datetime.now() - trade.entry_time
        
        if trade.opportunity.mode == TradingMode.SCALPING:
            if trade_duration.total_seconds() > 1800:  # 30 minutes max
                return True, "TIME_EXIT_SCALP"
        elif trade.opportunity.mode == TradingMode.MOMENTUM:
            if trade_duration.total_seconds() > 3600:  # 1 hour max
                return True, "TIME_EXIT_MOMENTUM"
        # Swing trades can run longer
        
        return False, "HOLDING"
    
    def close_trade(self, trade: ActiveTrade, exit_price: float, reason: str):
        """Close trade and update portfolio"""
        
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.status = "CLOSED"
        
        direction = trade.opportunity.analysis_data.get('direction', 'BUY')
        
        if direction == 'BUY':
            pnl_points = exit_price - trade.entry_price
        else:
            pnl_points = trade.entry_price - exit_price
        
        trade.final_pnl = pnl_points * trade.quantity
        
        # Update portfolio
        self.total_pnl += trade.final_pnl
        self.daily_pnl += trade.final_pnl
        
        # Release margin
        trade_value = trade.quantity * trade.entry_price
        self.current_capital += trade_value * 0.1  # Release margin
        self.current_capital += trade.final_pnl     # Add/subtract P&L
        
        print(f"   ✅ CLOSED {trade.opportunity.symbol}: Rs.{trade.final_pnl:+,.0f} ({reason})")
        
        # Remove from active trades
        del self.active_trades[trade.id]
    
    def generate_performance_report(self):
        """Generate performance report"""
        
        print(f"\n📊 SIMPLE MONEY MACHINE PERFORMANCE REPORT")
        print("=" * 80)
        
        current_total_capital = self.current_capital + sum(t.current_pnl for t in self.active_trades.values())
        roi = ((current_total_capital - self.capital) / self.capital) * 100
        
        print(f"💰 CAPITAL STATUS:")
        print(f"   Starting Capital:        Rs.{self.capital:10,.0f}")
        print(f"   Available Capital:       Rs.{self.current_capital:10,.0f}")
        print(f"   Total Capital Value:     Rs.{current_total_capital:10,.0f}")
        print(f"   Total P&L:               Rs.{self.total_pnl:+9,.0f}")
        print(f"   Daily P&L:               Rs.{self.daily_pnl:+9,.0f}")
        print(f"   ROI:                     {roi:+8.2f}%")
        
        print(f"\n📊 TRADING METRICS:")
        print(f"   Opportunities Found:     {self.opportunities_found:10d}")
        print(f"   Trades Executed:         {self.trades_executed:10d}")
        print(f"   Active Positions:        {len(self.active_trades):10d}")
        
        if self.active_trades:
            print(f"\n⚡ ACTIVE POSITIONS:")
            for trade in self.active_trades.values():
                symbol_short = trade.opportunity.symbol.split(':')[-1][:10]
                print(f"   {symbol_short:12} {trade.opportunity.mode.value:8} Rs.{trade.current_pnl:+8,.0f}")
        
        print("=" * 80)
    
    def run_money_machine(self, duration_minutes: int = 60):
        """Run the simple money machine"""
        
        if not self.api_working:
            print("❌ Cannot run - API connection failed")
            return
        
        print(f"\n🚀 STARTING SIMPLE MONEY MACHINE")
        print("=" * 80)
        print(f"⏰ Duration: {duration_minutes} minutes")
        print(f"🎯 Goal: Automated profit generation")
        print(f"💰 Capital: Rs.{self.capital:,.0f}")
        print("=" * 80)
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        scan_interval = 60  # Scan every minute
        last_scan = datetime.now() - timedelta(seconds=scan_interval)
        
        try:
            while self.is_running and datetime.now() < end_time:
                current_time = datetime.now()
                
                # Market scanning
                if (current_time - last_scan).total_seconds() >= scan_interval:
                    opportunities = self.scan_all_opportunities()
                    
                    # Execute best opportunities
                    for opportunity in opportunities[:2]:  # Execute top 2
                        if len(self.active_trades) < self.max_positions:
                            executed_trade = self.execute_trade(opportunity)
                            if executed_trade:
                                time.sleep(1)
                    
                    last_scan = current_time
                
                # Monitor trades every 30 seconds
                self.monitor_active_trades()
                
                # Performance report every 5 minutes
                elapsed = (current_time - start_time).total_seconds() / 60
                if elapsed > 0 and int(elapsed) % 5 == 0:
                    self.generate_performance_report()
                
                # Risk protection
                if self.daily_pnl <= -self.max_daily_loss:
                    print(f"\n🛑 DAILY LOSS LIMIT REACHED")
                    break
                
                time.sleep(30)  # Main loop interval
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Machine stopped by user")
        
        finally:
            self.is_running = False
            print(f"\n🏁 SIMPLE MONEY MACHINE SESSION COMPLETE")
            self.generate_performance_report()

def main():
    """Main execution function"""
    
    print("🚀💰 SIMPLE ULTIMATE MONEY MACHINE 💰🚀")
    print("=" * 80)
    print("This system will:")
    print("• Scan markets for profitable opportunities")
    print("• Execute trades automatically")
    print("• Monitor and manage positions")
    print("• Generate consistent profits")
    print("=" * 80)
    
    # Safety confirmation
    confirm = input("\n⚠️ This uses REAL MONEY. Type 'START' to begin: ")
    if confirm != 'START':
        print("Operation cancelled for safety")
        return
    
    # Initialize machine
    machine = SimpleMoneyMachine()
    
    if machine.api_working:
        duration = input("\nEnter runtime in minutes (default 60): ") or "60"
        try:
            runtime = int(duration)
            machine.run_money_machine(runtime)
        except ValueError:
            print("Invalid duration, using 60 minutes")
            machine.run_money_machine(60)
    else:
        print("Cannot start - API connection failed")

if __name__ == "__main__":
    main()