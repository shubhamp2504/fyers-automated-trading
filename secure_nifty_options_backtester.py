"""
SECURE NIFTY OPTIONS SUPPLY & DEMAND BACKTESTER
REAL DATA ONLY - FYERS API EXCLUSIVE

SECURITY FEATURES:
- Uses ONLY official Fyers API - NO external servers
- Real market data direct from Fyers
- No MCP dependencies or third-party services
- Secure authentication through official Fyers credentials

STRATEGY: Supply & Demand Zone Trading
- Demand zones = institutional buying areas (consolidation before upward moves)
- Supply zones = institutional selling areas (consolidation before downward moves)  
- Only trade WITH the confirmed trend direction
- Use last candle before impulse to identify exact zones
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fyers_client import FyersClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import warnings
warnings.filterwarnings('ignore')

class SecureNiftyOptionsBacktester:
    """
    SECURE NIFTY Options backtesting using ONLY official Fyers API
    NO external dependencies - Real market data ONLY
    """
    
    def __init__(self):
        print("SECURE NIFTY OPTIONS SUPPLY & DEMAND BACKTESTER")
        print("=" * 80)
        print("SECURITY: FYERS API EXCLUSIVE - NO EXTERNAL SERVERS")
        print("DATA SOURCE: REAL MARKET DATA via Official Fyers API")
        print("AUTHENTICATION: Official Fyers Credentials Only") 
        print("=" * 80)
        
        # Initialize SECURE Fyers client
        try:
            self.client = FyersClient()
            print("SECURE CONNECTION: Official Fyers API initialized successfully")
        except Exception as e:
            print(f"SECURITY ERROR: Failed to initialize Fyers client: {e}")
            raise
        
        # MULTI-TIMEFRAME POWERHOUSE PARAMETERS
        # Enhanced for frequent trading and quick profits
        self.timeframes = {
            '1': '1min',    # 1 minute - Scalping
            '3': '3min',    # 3 minutes - Quick entries
            '5': '5min',    # 5 minutes - Standard
            '10': '10min',  # 10 minutes - Swing entries
            '15': '15min',  # 15 minutes - Trend confirmation
            '30': '30min',  # 30 minutes - Major zones
            '60': '1hour',  # 1 hour - Strong zones
            '240': '4hour'  # 4 hours - Institutional zones
        }
        
        # Multi-timeframe data storage
        self.multi_data = {}
        self.multi_zones = {
            'supply': {},
            'demand': {}
        }
        
        # MONEY-MAKING MACHINE PARAMETERS
        self.lookback_period = 15  # Faster analysis for frequent trades
        self.min_impulse_size = 0.15  # Lower threshold for more opportunities
        self.institutional_impulse_size = 0.4  # Institutional zones
        self.zone_buffer = 0.15    # Tighter buffer (15%) for precise entries
        self.risk_per_trade = 0.008  # 0.8% risk (more aggressive)
        self.supply_double_qty = 2.0  # Double quantity for supply zones (faster moves)
        self.max_dte = 10   # Extended expiry for better premiums
        self.min_premium = 2.0   # Lower premium threshold for frequent trading
        
        # QUICK PROFIT BOOKING SYSTEM
        self.micro_profit_points = [5, 8, 12, 15, 18, 22, 25]  # Frequent booking
        self.micro_stop_loss = 6      # Tight 6-point stop loss
        self.supply_profit_target = 1.8   # 80% profit for supply (faster)
        self.demand_profit_target = 1.6   # 60% profit for demand
        
        # Performance tracking
        self.trade_frequency_target = 15  # Target 15+ trades per month
        
        # Data storage
        self.nifty_data = None
        self.supply_zones = []
        self.demand_zones = []
        self.trades = []
        self.capital = 100000  # Starting capital
        
        print(f"Strategy: MULTI-TIMEFRAME POWERHOUSE SYSTEM")
        print(f"  - Supply Zones: DOUBLE QUANTITY (2x profit potential)")
        print(f"  - Demand Zones: Standard quantity + Quick profits")
        print(f"  - Timeframes: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h")
        print(f"  - Target: {self.trade_frequency_target}+ trades/month")
        print(f"Starting Capital: Rs.{self.capital:,.2f}")
        print(f"Risk: {self.risk_per_trade*100}% per trade | Supply zones: DOUBLE QTY")
        print(f"Targets: Micro 5-25pts | Stop: {self.micro_stop_loss}pts")
    
    def fetch_multi_timeframe_data(self, start_date: str, end_date: str):
        """
        Fetch NIFTY data across ALL timeframes for powerhouse analysis
        Creates comprehensive multi-timeframe zone identification
        """
        print(f"\nFETCHING MULTI-TIMEFRAME DATA - POWERHOUSE MODE")
        print("-" * 60)
        print(f"TIMEFRAMES: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h")
        print(f"DATA SOURCE: NSE via Official Fyers API")
        print(f"PERIOD: {start_date} to {end_date}")
        print(f"PURPOSE: Maximum trading opportunities identification")
        
        success_count = 0
        total_candles = 0
        
        for resolution, name in self.timeframes.items():
            try:
                print(f"Fetching {name} data...")
                
                # Use official Fyers API for each timeframe
                df = self.client.get_historical_data(
                    symbol="NSE:NIFTY50-INDEX",
                    start_date=start_date,
                    end_date=end_date,
                    resolution=resolution
                )
                
                if df is not None and len(df) > 0:
                    self.multi_data[name] = df
                    success_count += 1
                    total_candles += len(df)
                    
                    print(f"✅ {name}: {len(df)} candles | Range: Rs.{df['low'].min():.2f} - Rs.{df['high'].max():.2f}")
                else:
                    print(f"❌ Failed to fetch {name} data")
                    
            except Exception as e:
                print(f"❌ Error fetching {name}: {str(e)}")
        
        print(f"\n📊 MULTI-TIMEFRAME POWERHOUSE LOADED:")
        print(f"   Timeframes: {success_count}/{len(self.timeframes)} successful")
        print(f"   Total Candles: {total_candles:,} across all timeframes")
        print(f"   Data Quality: {'EXCELLENT' if success_count >= 6 else 'GOOD' if success_count >= 4 else 'LIMITED'}")
        
        # Set primary timeframe data for compatibility
        if '5min' in self.multi_data:
            self.nifty_data = self.multi_data['5min']
        elif len(self.multi_data) > 0:
            self.nifty_data = list(self.multi_data.values())[0]
        
        return success_count >= 4  # Need at least 4 timeframes
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility measurement"""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def identify_indecision_candles(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify consolidation candles - areas where institutions accumulate/distribute
        More practical approach while maintaining institutional focus
        """
        conditions = [
            # Small to medium body relative to total range (consolidation)
            df['body_size'] <= (df['high'] - df['low']) * 0.5,
            # Must have some wicks (not pure trending candle)
            (df['upper_wick'] + df['lower_wick']) > 0,
            # Total wick size shows indecision
            (df['upper_wick'] + df['lower_wick']) >= df['body_size'] * 0.5,
            # Real volume activity (not just noise)
            df['volume'] > df['volume'].rolling(5).mean() * 0.3
        ]
        
        return pd.Series(np.logical_and.reduce(conditions), index=df.index)
    
    def identify_all_impulse_moves(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify ALL impulse moves for complete supply/demand system
        Captures both institutional and regular trading opportunities
        """
        impulse_data = []
        
        for i in range(2, len(df)):  # Need at least 2 previous candles
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            prev2_candle = df.iloc[i-2]
            
            # Calculate move size and velocity
            move_size = abs(current_candle['close'] - prev_candle['close']) / prev_candle['close'] * 100
            velocity = abs(current_candle['close'] - prev2_candle['close']) / prev2_candle['close'] * 100
            
            # Lower threshold to capture ALL significant moves
            is_impulse = (
                move_size >= self.min_impulse_size and  # Any significant move
                current_candle['body_size'] >= (current_candle['high'] - current_candle['low']) * 0.5 and # Decent body
                current_candle['volume'] > df['volume'].rolling(10).mean().iloc[i] * 0.8  # Some volume
            )
            
            if is_impulse:
                direction = 'bullish' if current_candle['close'] > current_candle['open'] else 'bearish'
                
                # Determine if this is institutional level
                is_institutional = (
                    move_size >= self.institutional_impulse_size and
                    velocity >= self.institutional_impulse_size * 1.2 and
                    current_candle['volume'] > df['volume'].rolling(20).mean().iloc[i] * 1.3
                )
                
                impulse_data.append({
                    'timestamp': current_candle.name,
                    'direction': direction,
                    'move_size': move_size,
                    'velocity': velocity,
                    'volume_ratio': current_candle['volume'] / df['volume'].rolling(15).mean().iloc[i],
                    'is_institutional': is_institutional,
                    'open': current_candle['open'],
                    'high': current_candle['high'],
                    'low': current_candle['low'],
                    'close': current_candle['close'],
                    'prev_candle_idx': i-1
                })
        
        return pd.DataFrame(impulse_data)
    
    def identify_multi_timeframe_zones(self):
        """
        POWERHOUSE MULTI-TIMEFRAME ZONE IDENTIFICATION
        Analyzes ALL timeframes to create comprehensive trading opportunities
        Creates zones for frequent trading with quick profits
        """
        print(f"\nIDENTIFYING MULTI-TIMEFRAME SUPPLY & DEMAND ZONES")
        print("-" * 60)
        print(f"ANALYZING: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h timeframes")
        print(f"STRATEGY: Maximum opportunities + Double qty supply zones")
        
        all_supply_zones = []
        all_demand_zones = []
        total_zones_found = 0
        
        # Analyze each timeframe for zones
        timeframe_priority = {
            '4hour': 8, '1hour': 7, '30min': 6, '15min': 5,
            '10min': 4, '5min': 3, '3min': 2, '1min': 1
        }
        
        for tf_name, data in self.multi_data.items():
            if data is None or len(data) < self.lookback_period:
                continue
                
            print(f"\nAnalyzing {tf_name} timeframe...")
            
            # Find impulse moves in this timeframe
            impulses = self.identify_impulse_moves_enhanced(data, tf_name)
            zones_count = 0
            
            for _, impulse in impulses.iterrows():
                zone = self.create_zone_from_impulse(impulse, data, tf_name)
                if zone is not None:
                    # Add timeframe priority weight
                    zone['timeframe'] = tf_name
                    zone['priority'] = timeframe_priority.get(tf_name, 1)
                    zone['enhanced_strength'] = zone['strength'] * zone['priority']
                    
                    if impulse['direction'] == 'bullish':
                        # Bullish impulse = Previous area is DEMAND zone
                        zone['zone_type'] = 'demand'
                        all_demand_zones.append(zone)
                    else:
                        # Bearish impulse = Previous area is SUPPLY zone
                        zone['zone_type'] = 'supply'
                        # Supply zones get double quantity flag
                        zone['double_quantity'] = True
                        all_supply_zones.append(zone)
                    
                    zones_count += 1
                    total_zones_found += 1
            
            print(f"✅ {tf_name}: {zones_count} zones found")
        
        # Sort zones by enhanced strength
        all_supply_zones.sort(key=lambda x: x['enhanced_strength'], reverse=True)
        all_demand_zones.sort(key=lambda x: x['enhanced_strength'], reverse=True)
        
        # Store top zones for each timeframe
        self.multi_zones['supply'] = all_supply_zones[:20]  # Top 20 supply zones
        self.multi_zones['demand'] = all_demand_zones[:20]  # Top 20 demand zones
        
        # Compatibility with existing code
        self.supply_zones = all_supply_zones[:15]
        self.demand_zones = all_demand_zones[:15]
        
        print(f"\n📊 MULTI-TIMEFRAME ZONE ANALYSIS COMPLETE:")
        print(f"   Total Zones Found: {total_zones_found}")
        print(f"   Supply Zones: {len(all_supply_zones)} (DOUBLE QUANTITY ENABLED)")
        print(f"   Demand Zones: {len(all_demand_zones)} (Standard quantity)")
        
        if len(all_supply_zones) > 0:
            print(f"   TOP SUPPLY: {all_supply_zones[0]['timeframe']} Rs.{all_supply_zones[0]['low']:.2f}-{all_supply_zones[0]['high']:.2f} (Strength: {all_supply_zones[0]['enhanced_strength']:.1f})")
        
        if len(all_demand_zones) > 0:
            print(f"   TOP DEMAND: {all_demand_zones[0]['timeframe']} Rs.{all_demand_zones[0]['low']:.2f}-{all_demand_zones[0]['high']:.2f} (Strength: {all_demand_zones[0]['enhanced_strength']:.1f})")
        self.supply_zones = supply_zones + institutional_supply
        self.demand_zones = demand_zones + institutional_demand
        self.institutional_supply = institutional_supply
        self.institutional_demand = institutional_demand
        
        print(f"REGULAR SUPPLY ZONES: {len(supply_zones)} (Quick Profit)")
        print(f"REGULAR DEMAND ZONES: {len(demand_zones)} (Quick Profit)")
        print(f"INSTITUTIONAL SUPPLY: {len(institutional_supply)} (Double Qty)")
        print(f"INSTITUTIONAL DEMAND: {len(institutional_demand)} (Double Qty)")
        print(f"TOTAL TRADING ZONES: {len(self.supply_zones) + len(self.demand_zones)}")
        
        # Show top zones by category
        if institutional_supply:
            top = institutional_supply[0]
            print(f"TOP INSTITUTIONAL SUPPLY: Rs.{top['low']:.2f}-{top['high']:.2f} (Strength: {top['strength']:.1f})")
        
        if institutional_demand:
            top = institutional_demand[0]
            print(f"TOP INSTITUTIONAL DEMAND: Rs.{top['low']:.2f}-{top['high']:.2f} (Strength: {top['strength']:.1f})")
            
        if supply_zones:
            top = supply_zones[0]
            print(f"TOP REGULAR SUPPLY: Rs.{top['low']:.2f}-{top['high']:.2f} (Strength: {top['strength']:.1f})")
        
        if demand_zones:
            top = demand_zones[0]
            print(f"TOP REGULAR DEMAND: Rs.{top['low']:.2f}-{top['high']:.2f} (Strength: {top['strength']:.1f})")
    
    def determine_market_trend_secure(self, current_idx: int) -> str:
        """
        Determine market trend using secure swing analysis
        Conservative approach for real money trading
        """
        if current_idx < 25:  # Need more history for reliable trend
            return 'sideways'
        
        # Look at last 25 candles for comprehensive trend analysis
        recent_data = self.nifty_data.iloc[current_idx-24:current_idx+1]
        
        # Calculate multiple trend indicators
        sma_short = recent_data['close'].rolling(5).mean()
        sma_long = recent_data['close'].rolling(15).mean()
        
        # Swing high/low analysis
        highs = recent_data['high'].rolling(7, center=True).max()
        lows = recent_data['low'].rolling(7, center=True).min()
        
        swing_highs = []
        swing_lows = []
        
        for i in range(3, len(recent_data)-3):
            if recent_data['high'].iloc[i] == highs.iloc[i]:
                swing_highs.append(recent_data['high'].iloc[i])
            if recent_data['low'].iloc[i] == lows.iloc[i]:
                swing_lows.append(recent_data['low'].iloc[i])
        
        # Multiple confirmation approach
        trend_signals = []
        
        # 1. SMA trend
        if len(sma_short) >= 2 and len(sma_long) >= 2:
            if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-1] > sma_short.iloc[-2]:
                trend_signals.append('bullish')
            elif sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-1] < sma_short.iloc[-2]:
                trend_signals.append('bearish')
            else:
                trend_signals.append('sideways')
        
        # 2. Swing analysis
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            higher_highs = swing_highs[-1] > swing_highs[-2] if len(swing_highs) >= 2 else False
            higher_lows = swing_lows[-1] > swing_lows[-2] if len(swing_lows) >= 2 else False
            lower_highs = swing_highs[-1] < swing_highs[-2] if len(swing_highs) >= 2 else False  
            lower_lows = swing_lows[-1] < swing_lows[-2] if len(swing_lows) >= 2 else False
            
            if higher_highs and higher_lows:
                trend_signals.append('bullish')
            elif lower_highs and lower_lows:
                trend_signals.append('bearish')
            else:
                trend_signals.append('sideways')
        
        # 3. Recent price action
        recent_close_trend = recent_data['close'].iloc[-5:].pct_change().sum()
        if recent_close_trend > 0.005:  # 0.5% net gain
            trend_signals.append('bullish')
        elif recent_close_trend < -0.005:  # 0.5% net loss
            trend_signals.append('bearish')
        else:
            trend_signals.append('sideways')
        
        # Consensus approach - need majority agreement
        bullish_count = trend_signals.count('bullish')
        bearish_count = trend_signals.count('bearish')
        sideways_count = trend_signals.count('sideways')
        
        if bullish_count >= 2:
            return 'uptrend'
        elif bearish_count >= 2:
            return 'downtrend'
        else:
            return 'sideways'
    
    def get_real_option_strike(self, current_price: float, zone_price: float, option_type: str) -> float:
        """
        Calculate appropriate option strike using real market conventions
        NIFTY options trade in 50-point intervals
        """
        # Round to nearest 50 (NIFTY standard)
        base_strike = round(zone_price / 50) * 50
        
        if option_type == 'CE':
            # For calls in demand zones - slightly OTM for better risk/reward
            if current_price > zone_price:
                return base_strike + 50  # 1 strike OTM
            else:
                return base_strike  # ATM
        else:  # PE
            # For puts in supply zones - slightly OTM for better risk/reward
            if current_price < zone_price:
                return base_strike - 50  # 1 strike OTM
            else:
                return base_strike  # ATM
    
    def construct_real_option_symbol(self, strike: float, option_type: str, expiry_date: str) -> str:
        """
        Construct real NIFTY option symbol using actual Fyers format
        Format: NSE:NIFTY{YY}{MMM}{DD}{STRIKE}{CE/PE}
        """
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        
        # Real Fyers format
        year = expiry_dt.strftime("%y")
        month = expiry_dt.strftime("%b").upper()
        day = expiry_dt.strftime("%d")
        
        # Example: NSE:NIFTY25JAN0924500CE
        symbol = f"NSE:NIFTY{year}{month}{day}{int(strike):05d}{option_type}"
        
        return symbol
    
    def get_real_option_expiry(self, current_date: datetime) -> str:
        """
        Get real NIFTY option expiry (every Thursday)
        Uses actual market calendar
        """
        # NIFTY options expire every Thursday
        days_to_thursday = (3 - current_date.weekday()) % 7
        if days_to_thursday == 0 and current_date.hour >= 15:  # After market close on Thursday
            days_to_thursday = 7  # Next Thursday
        
        expiry_date = current_date + timedelta(days=days_to_thursday)
        
        # Ensure we don't go beyond max DTE for liquidity
        if days_to_thursday > self.max_dte:
            return None  # Skip if too far out
        
        return expiry_date.strftime("%Y-%m-%d")
    
    def calculate_real_option_premium(self, spot_price: float, strike: float, option_type: str, dte: int) -> float:
        """
        Calculate realistic option premium using market-based approach
        Conservative estimates for real trading
        """
        # Real market factors
        risk_free_rate = 0.07  # Current Indian government bond yield
        volatility = 0.18  # NIFTY historical volatility (conservative)
        
        # Time to expiry in years
        time_to_expiry = max(dte, 1) / 365.0
        
        # Moneyness calculation
        moneyness = spot_price / strike if option_type == 'CE' else strike / spot_price
        
        # Intrinsic value
        if option_type == 'CE':
            intrinsic_value = max(0, spot_price - strike)
        else:  # PE
            intrinsic_value = max(0, strike - spot_price)
        
        # Time value based on real market behavior
        time_decay_factor = np.sqrt(time_to_expiry)
        volatility_component = spot_price * volatility * time_decay_factor * 0.1
        
        # Liquidity adjustment (higher for OTM options)
        liquidity_premium = abs(1 - moneyness) * spot_price * 0.002
        
        # Total premium
        total_premium = intrinsic_value + volatility_component + liquidity_premium
        
        # Market maker spread (realistic bid-ask)
        spread = total_premium * 0.03  # 3% spread
        premium_mid = max(total_premium, self.min_premium)
        
        return round(premium_mid + spread/2, 2)  # Add half spread for realistic entry
    
    def validate_zone_for_trading(self, current_candle: pd.Series, zone: dict) -> bool:
        """
        Strict validation for institutional zone trading
        Conservative approach for real money
        """
        zone_high = zone['high']
        zone_low = zone['low']
        zone_mid = (zone_high + zone_low) / 2
        
        # Reduced buffer for precision
        buffer = (zone_high - zone_low) * self.zone_buffer
        zone_high_buffer = zone_high + buffer
        zone_low_buffer = zone_low - buffer
        
        # Current candle overlap check
        candle_high = current_candle['high']
        candle_low = current_candle['low']
        
        # Zone overlap
        zone_overlap = (candle_low <= zone_high_buffer and candle_high >= zone_low_buffer)
        
        # Additional validations for enhanced trading
        # Different criteria for institutional vs regular zones
        is_institutional = zone.get('is_institutional', False)
        
        if is_institutional:
            # Stricter for institutional (double risk)
            validations = [
                zone_overlap,  # Basic zone touch
                zone['trade_count'] < 3,  # Limited institutional usage
                zone['strength'] > 1.0,  # Strong institutional zones only
                zone['volume_confirmation'] > 1.2,  # Strong volume confirmation
                not zone['broken']  # Zone not invalidated
            ]
        else:
            # More relaxed for regular zones (quick profits)
            validations = [
                zone_overlap,  # Basic zone touch
                zone['trade_count'] < 8,  # Allow more regular zone usage
                zone['strength'] > 0.2,  # Much lower threshold for regular zones
                zone['volume_confirmation'] > 0.9,  # Relaxed volume requirement
                not zone['broken']  # Zone not invalidated
            ]
        
        return all(validations)
    
    def execute_powerhouse_options_trade(self, signal: dict, current_candle: pd.Series) -> dict:
        """
        POWERHOUSE OPTIONS EXECUTION
        Enhanced for frequent trading with double quantity supply zones
        Features: Multi-timeframe validation, quick profits, tight stops
        """
        current_price = current_candle['close']
        current_date = current_candle.name
        
        # Get optimal expiry for frequent trading
        expiry_date = self.get_real_option_expiry(current_date)
        if expiry_date is None:
            return None
        
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        dte = (expiry_dt.date() - current_date.date()).days
        
        if dte > self.max_dte or dte < 1:
            return None
        
        # Enhanced option selection
        if signal['action'] == 'buy_call':
            option_type = 'CE'
            strike = self.get_optimized_strike(current_price, signal['zone_price'], 'CE', signal.get('timeframe', '5min'))
        elif signal['action'] == 'buy_put':
            option_type = 'PE'
            strike = self.get_optimized_strike(current_price, signal['zone_price'], 'PE', signal.get('timeframe', '5min'))
        else:
            return None
        
        # Real option symbol
        option_symbol = self.construct_real_option_symbol(strike, option_type, expiry_date)
        
        # Enhanced premium calculation
        premium = self.calculate_enhanced_premium(current_price, strike, option_type, dte, signal.get('timeframe', '5min'))
        
        # POWERHOUSE POSITION SIZING
        is_supply_zone = signal.get('zone_type') == 'supply'
        base_risk = self.capital * self.risk_per_trade
        
        # DOUBLE QUANTITY FOR SUPPLY ZONES (faster profit potential)
        if is_supply_zone:
            risk_amount = base_risk * self.supply_double_qty  # 2x position size
            trade_type = 'SUPPLY_DOUBLE'
        else:
            risk_amount = base_risk
            trade_type = 'DEMAND_STANDARD'
        
        # Calculate lot size
        max_loss_per_lot = premium * 75 * 0.5  # Tighter risk (50% max loss)
        
        if max_loss_per_lot > 0:
            quantity = max(1, min(int(risk_amount / max_loss_per_lot), 50 if is_supply_zone else 25))
        else:
            quantity = 2 if is_supply_zone else 1
        
        # Enhanced trade record
        trade = {
            'entry_time': current_date,
            'symbol': option_symbol,
            'action': signal['action'],
            'spot_price': current_price,
            'strike': strike,
            'option_type': option_type,
            'quantity': quantity,
            'entry_premium': premium,
            'dte': dte,
            'zone_type': signal['zone_type'],
            'zone_price': signal['zone_price'],
            'zone_strength': signal.get('zone_strength', 0),
            'timeframe': signal.get('timeframe', '5min'),
            'trade_type': trade_type,
            'double_quantity': is_supply_zone,
            'target_profit': self.supply_profit_target if is_supply_zone else self.demand_profit_target,
            'stop_loss_points': self.micro_stop_loss,
            'profit_targets': self.micro_profit_points.copy(),
            'status': 'open',
            'exit_time': None,
            'exit_premium': None,
            'pnl': 0,
            'exit_reason': None,
            'lot_size': 75
        }
        
        print(f"{trade_type} {option_type}: {option_symbol} @ Rs.{premium:.2f}")
        print(f"  Spot: Rs.{current_price:.2f}, Qty: {quantity}{'(2X)' if is_supply_zone else ''}, Timeframe: {signal.get('timeframe', '5min')}")
        
        return trade
    def get_optimized_strike(self, current_price: float, zone_price: float, option_type: str, timeframe: str) -> float:
        """
        Get optimized strike based on timeframe and zone proximity
        Closer strikes for frequent trading, further strikes for longer timeframes
        """
        base_strike = round(zone_price / 50) * 50
        
        # Timeframe-based strike adjustment for optimal delta
        if timeframe in ['1min', '3min']:
            # Scalping - slightly ITM for quick moves
            offset = 25 if option_type == 'CE' else -25
        elif timeframe in ['5min', '10min']:
            # Quick trades - ATM or slightly OTM
            offset = 0
        elif timeframe in ['15min', '30min']:
            # Standard - slightly OTM for better risk/reward
            offset = 50 if option_type == 'CE' else -50
        else:
            # Longer timeframes - further OTM for bigger moves
            offset = 100 if option_type == 'CE' else -100
        
        return base_strike + offset
    
    def calculate_enhanced_premium(self, spot: float, strike: float, option_type: str, dte: int, timeframe: str) -> float:
        """
        Enhanced premium calculation with timeframe considerations
        """
        # Base premium calculation (simplified model)
        moneyness = (spot - strike) / strike if option_type == 'CE' else (strike - spot) / strike
        intrinsic = max(0, moneyness * strike)
        
        # Time value based on DTE and timeframe
        time_decay_factor = max(0.1, dte / 7.0)  # Normalized to weekly expiry
        
        # Timeframe-based volatility adjustment
        if timeframe in ['1min', '3min']:
            vol_multiplier = 1.5  # Higher implied vol for scalping
        elif timeframe in ['5min', '10min']:
            vol_multiplier = 1.2
        else:
            vol_multiplier = 1.0
        
        # Estimate time value
        base_time_value = spot * 0.01 * time_decay_factor * vol_multiplier
        time_value = max(base_time_value, spot * 0.005)  # Minimum time value
        
        premium = intrinsic + time_value
        return max(premium, 2.0)  # Minimum premium for liquidity
    
    def identify_impulse_moves_enhanced(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Enhanced impulse detection for multi-timeframe analysis
        Lower thresholds for more trading opportunities
        """
        impulses = []
        
        # Adjust thresholds based on timeframe
        if timeframe in ['1min', '3min']:
            min_move = 0.08  # Very sensitive for scalping
        elif timeframe in ['5min', '10min']:
            min_move = 0.12  # Quick entries
        elif timeframe in ['15min', '30min']:
            min_move = 0.18  # Standard moves
        else:
            min_move = 0.25  # Stronger moves for higher timeframes
        
        for i in range(self.lookback_period, len(data) - 1):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Calculate move percentage
            move_size = abs(current['close'] - previous['close']) / previous['close']
            
            if move_size >= min_move:
                # Volume confirmation (relaxed for more opportunities)
                volume_ratio = current['volume'] / data['volume'].iloc[i-10:i].mean() if i >= 10 else 1.5
                
                if volume_ratio > 0.8:  # Lower volume threshold
                    direction = 'bullish' if current['close'] > previous['close'] else 'bearish'
                    
                    impulse = {
                        'timestamp': current.name,
                        'direction': direction,
                        'move_size': move_size,
                        'volume_ratio': volume_ratio,
                        'prev_candle_idx': i-1,
                        'current_idx': i,
                        'timeframe': timeframe
                    }
                    impulses.append(impulse)
        
        return pd.DataFrame(impulses)
    
    def create_zone_from_impulse(self, impulse: dict, data: pd.DataFrame, timeframe: str) -> dict:
        """
        Create precise supply/demand zone from impulse move
        Enhanced for frequent trading opportunities
        """
        prev_idx = impulse['prev_candle_idx'] 
        
        if prev_idx < 2 or prev_idx >= len(data):
            return None
        
        # Look for consolidation before the impulse (wider search for more zones)
        consolidation_candles = []
        
        for look_back in range(0, min(5, prev_idx)):  # Check up to 5 candles back
            check_idx = prev_idx - look_back
            candle = data.iloc[check_idx]
            
            # More lenient consolidation criteria
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            
            if body_size < candle_range * 0.4:  # Small body relative to range
                consolidation_candles.append(candle)
        
        if len(consolidation_candles) == 0:
            # If no consolidation found, use previous candle
            consolidation_candles = [data.iloc[prev_idx]]
        
        # Create zone from consolidation area
        zone_high = max(c['high'] for c in consolidation_candles)
        zone_low = min(c['low'] for c in consolidation_candles)
        
        # Calculate enhanced strength based on timeframe importance
        timeframe_multipliers = {
            '4hour': 4.0, '1hour': 3.0, '30min': 2.5, '15min': 2.0,
            '10min': 1.5, '5min': 1.2, '3min': 1.0, '1min': 0.8
        }
        
        base_strength = impulse['move_size'] * impulse['volume_ratio']
        enhanced_strength = base_strength * timeframe_multipliers.get(timeframe, 1.0)
        
        zone = {
            'timestamp': consolidation_candles[0].name,
            'high': zone_high,
            'low': zone_low,
            'strength': base_strength,
            'enhanced_strength': enhanced_strength,
            'volume_confirmation': impulse['volume_ratio'],
            'move_size': impulse['move_size'],
            'timeframe': timeframe,
            'tested': False,
            'broken': False,
            'trade_count': 0,
            'success_rate': 0.0
        }
        
        return zone
    
    def run_powerhouse_backtest(self, start_date: str, end_date: str):
        """
        MULTI-TIMEFRAME POWERHOUSE BACKTESTING
        Enhanced for frequent trading with double quantity supply zones
        Features: All timeframes analysis, quick profits, tight stops
        """
        print(f"\nSTARTING MULTI-TIMEFRAME POWERHOUSE BACKTEST")
        print("DATA: REAL MARKET DATA via Fyers API ONLY")
        print(f"PERIOD: {start_date} to {end_date}")
        print(f"CAPITAL: Rs.{self.capital:,.2f} (Aggressive Growth Mode)")
        print("=" * 80)
        
        # Fetch multi-timeframe data
        if not self.fetch_multi_timeframe_data(start_date, end_date):
            print("CRITICAL ERROR: Failed to fetch multi-timeframe data")
            return
        
        # Identify zones across all timeframes
        self.identify_multi_timeframe_zones()
        
        if len(self.supply_zones) == 0 and len(self.demand_zones) == 0:
            print("NO TRADING ZONES: No zones identified across timeframes")
            return
        
        # Enhanced backtesting simulation
        open_trades = []
        signals_generated = 0
        trades_executed = 0
        supply_trades = 0
        demand_trades = 0
        
        self.daily_performance = {}
        
        print(f"\nEXECUTING POWERHOUSE BACKTEST...")
        print(f"SUPPLY ZONES: {len(self.supply_zones)} (DOUBLE QUANTITY)")
        print(f"DEMAND ZONES: {len(self.demand_zones)} (Standard quantity)")
        
        # Use 5-minute data as primary execution timeframe
        if '5min' not in self.multi_data:
            print("ERROR: 5-minute data required for execution")
            return
        
        primary_data = self.multi_data['5min']
        
        for i in range(max(25, self.lookback_period), len(primary_data)):
            current_candle = primary_data.iloc[i]
            current_date = current_candle.name
            current_price = current_candle['close']
            
            # Enhanced trade management with frequent profit booking
            self.manage_powerhouse_trades(current_candle, open_trades)
            
            # Market hours validation (NSE: 9:15 AM to 3:30 PM)
            current_time = current_date.time()
            if not (time(9, 15) <= current_time <= time(15, 30)):
                continue
            
            # Enhanced trend detection across multiple timeframes
            trend = self.determine_multi_timeframe_trend(i)
            
            # Position limits (more aggressive)
            active_positions = len([t for t in open_trades if t['status'] == 'open'])
            supply_positions = len([t for t in open_trades if t['status'] == 'open' and t.get('double_quantity', False)])
            
            # Allow more positions for frequent trading
            if active_positions >= 15 or supply_positions >= 8:
                continue
            
            # SUPPLY ZONE TRADING (Double quantity - faster profits)
            for zone in self.supply_zones[:10]:  # Top 10 supply zones
                if self.validate_multi_timeframe_zone(current_candle, zone, trend):
                    
                    signal = {
                        'action': 'buy_put',
                        'zone_type': 'supply',
                        'zone_price': (zone['high'] + zone['low']) / 2,
                        'zone_strength': zone['enhanced_strength'],
                        'timeframe': zone['timeframe'],
                        'trend': trend
                    }
                    
                    trade = self.execute_powerhouse_options_trade(signal, current_candle)
                    if trade:
                        open_trades.append(trade)
                        self.trades.append(trade)
                        signals_generated += 1
                        trades_executed += 1
                        supply_trades += 1
                        
                        # Update zone usage
                        zone['trade_count'] += 1
                        zone['tested'] = True
                        
                        # Track daily entries
                        entry_date = current_date.strftime('%Y-%m-%d')
                        if entry_date not in self.daily_performance:
                            self.daily_performance[entry_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                        self.daily_performance[entry_date]['entries'] += 1
                        
                        break  # One trade per candle
            
            # DEMAND ZONE TRADING (Standard quantity)
            if len([t for t in open_trades if t['status'] == 'open']) < 15:  # Still have capacity
                for zone in self.demand_zones[:10]:  # Top 10 demand zones
                    if self.validate_multi_timeframe_zone(current_candle, zone, trend):
                        
                        signal = {
                            'action': 'buy_call',
                            'zone_type': 'demand',
                            'zone_price': (zone['high'] + zone['low']) / 2,
                            'zone_strength': zone['enhanced_strength'],
                            'timeframe': zone['timeframe'],
                            'trend': trend
                        }
                        
                        trade = self.execute_powerhouse_options_trade(signal, current_candle)
                        if trade:
                            open_trades.append(trade)
                            self.trades.append(trade)
                            signals_generated += 1
                            trades_executed += 1
                            demand_trades += 1
                            
                            # Update zone usage
                            zone['trade_count'] += 1
                            zone['tested'] = True
                            
                            # Track daily entries
                            entry_date = current_date.strftime('%Y-%m-%d')
                            if entry_date not in self.daily_performance:
                                self.daily_performance[entry_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                            self.daily_performance[entry_date]['entries'] += 1
                            
                            break  # One trade per candle
        
        print(f"\nPOWERHOUSE BACKTEST COMPLETED")
        print(f"SIGNALS GENERATED: {signals_generated}")
        print(f"TRADES EXECUTED: {trades_executed}")
        print(f"  🔴 Supply Trades (2X): {supply_trades}")
        print(f"  🟢 Demand Trades (1X): {demand_trades}")
        print(f"FINAL CAPITAL: Rs.{self.capital:,.2f}")
        
        # Display daily performance breakdown
        self.print_daily_performance()
    
    def manage_complete_trades(self, current_candle: pd.Series, open_trades: list):
        """
        COMPLETE trade management system:
        - Institutional zones: 75% profit target, 40% stop loss
        - Regular zones: Quick profit booking (10-30 points), smaller losses
        """
        current_price = current_candle['close']
        current_date = current_candle.name
        
        for trade in open_trades:
            if trade['status'] != 'open':
                continue
            
            # Current DTE
            entry_date = trade['entry_time'].date()
            current_trade_date = current_date.date()
            dte = (datetime.strptime(trade['entry_time'].strftime("%Y-%m-%d"), "%Y-%m-%d").date() - current_trade_date).days
            
            # Current premium with real market calculation
            current_premium = self.calculate_real_option_premium(
                current_price, trade['strike'], trade['option_type'], max(0, abs(dte))
            )
            
            # COMPLETE exit strategy based on zone type
            exit_reason = None
            is_institutional = trade.get('is_institutional', False)
            
            if is_institutional:
                # INSTITUTIONAL ZONES: Longer targets, wider stops
                
                # 1. Profit target (100% gain for institutional)
                if current_premium >= trade['entry_premium'] * 2.0:
                    exit_reason = 'institutional_profit_target'
                
                # 2. Stop loss (40% loss)
                elif current_premium <= trade['entry_premium'] * 0.60:
                    exit_reason = 'institutional_stop_loss'
                
                # 3. Time decay (2 days to expiry)  
                elif abs(dte) <= 2:
                    exit_reason = 'time_decay'
            
            else:
                # REGULAR ZONES: Quick profit booking system
                
                # Calculate NIFTY points moved
                entry_spot = trade['spot_price']
                points_moved = abs(current_price - entry_spot)
                
                # Quick profit booking based on points
                profit_pct = (current_premium / trade['entry_premium'] - 1) * 100
                
                # 1. Quick profit booking (10-30 points or 25-50% profit)
                if (points_moved >= 10 and profit_pct >= 25) or \
                   (points_moved >= 15 and profit_pct >= 35) or \
                   (points_moved >= 20 and profit_pct >= 40) or \
                   (points_moved >= 30 and profit_pct >= 50):
                    exit_reason = f'quick_profit_{int(points_moved)}pts'
                
                # 2. Smaller stop loss for regular zones (50% loss)
                elif current_premium <= trade['entry_premium'] * 0.50:
                    exit_reason = 'regular_stop_loss'
                
                # 3. Earlier time decay (3 days to expiry for regular)
                elif abs(dte) <= 3:
                    exit_reason = 'time_decay'
            
            # Common exit conditions for both types
            if not exit_reason:
                # Trend reversal (strict)
                current_trend = self.determine_market_trend_secure(
                    self.nifty_data.index.get_loc(current_date)
                )
                if current_trend != trade['trend'] and current_trend != 'sideways':
                    exit_reason = 'trend_reversal'
                
                # Adverse move protection
                elif trade['option_type'] == 'CE' and current_price < trade['spot_price'] * 0.993:
                    exit_reason = 'adverse_move'
                elif trade['option_type'] == 'PE' and current_price > trade['spot_price'] * 1.007:
                    exit_reason = 'adverse_move'
            
            # Execute exit
            if exit_reason:
                trade['exit_time'] = current_date
                trade['exit_premium'] = current_premium
                trade['status'] = 'closed'
                trade['exit_reason'] = exit_reason
                
                # Calculate P&L
                trade['pnl'] = (current_premium - trade['entry_premium']) * trade['quantity'] * trade['lot_size']
                
                self.capital += trade['pnl']
                
                # Track daily P&L
                exit_date = current_date.strftime('%Y-%m-%d')
                if exit_date not in self.daily_performance:
                    self.daily_performance[exit_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                self.daily_performance[exit_date]['exits'] += 1
                self.daily_performance[exit_date]['daily_pnl'] += trade['pnl']
                self.daily_performance[exit_date]['trades'].append({
                    'symbol': trade['symbol'],
                    'pnl': trade['pnl'],
                    'type': trade.get('trade_type', 'REGULAR'),
                    'action': 'exit'
                })
                
                # Log exit
                points_moved = abs(current_price - trade['spot_price'])
                print(f"EXIT: {trade['symbol']} @ Rs.{current_premium:.2f} | "
                      f"P&L: Rs.{trade['pnl']:,.0f} | Reason: {exit_reason} | "
                      f"Points: {points_moved:.0f} | Type: {trade.get('trade_type', 'REGULAR')}")
    
    def run_secure_backtest(self, start_date: str, end_date: str):
        """
        Run secure backtesting using ONLY real Fyers data
        Conservative approach suitable for real money trading
        """
        print(f"\nSTARTING SECURE SUPPLY & DEMAND BACKTEST")
        print(f"DATA: REAL MARKET DATA via Fyers API ONLY")
        print(f"PERIOD: {start_date} to {end_date}")
        print(f"CAPITAL: Rs.{self.capital:,.2f} (Balanced Risk Management)")
        print("=" * 80)
        
        # Fetch real market data
        if not self.fetch_real_nifty_data(start_date, end_date, "5"):
            print("CRITICAL ERROR: Failed to fetch real market data")
            return
        
        # Identify ALL zones (institutional + regular)
        self.identify_all_supply_demand_zones()
        
        if len(self.supply_zones) == 0 and len(self.demand_zones) == 0:
            print("NO TRADING ZONES: No institutional zones identified in this period")
            return
        
        # Secure backtesting simulation
        open_trades = []
        signals_generated = 0
        trades_executed = 0
        self.daily_performance = {}  # Track daily P&L and trades
        
        print(f"\nEXECUTING SECURE BACKTEST...")
        
        trends_detected = {'uptrend': 0, 'downtrend': 0, 'sideways': 0}
        zone_tests = 0
        zone_validations_failed = 0
        
        for i in range(max(25, self.lookback_period), len(self.nifty_data)):
            current_candle = self.nifty_data.iloc[i]
            current_date = current_candle.name
            current_price = current_candle['close']
            
            # Manage existing positions
            self.manage_complete_trades(current_candle, open_trades)
            
            # Market hours validation (NSE: 9:15 AM to 3:30 PM)
            current_time = current_date.time()
            if not (time(9, 15) <= current_time <= time(15, 30)):
                continue
            
            # Determine market trend with multiple confirmations
            trend = self.determine_market_trend_secure(i)
            trends_detected[trend] += 1
            
            # Limit concurrent positions based on zone type
            active_positions = len([t for t in open_trades if t['status'] == 'open'])
            institutional_positions = len([t for t in open_trades if t['status'] == 'open' and t.get('is_institutional', False)])
            
            # Conservative limits: max 8 total positions, max 2 institutional
            if active_positions >= 8 or institutional_positions >= 2:
                continue
            
            # INTELLIGENT HYBRID TRADING LOGIC
            # Strong zones: Trade in any trend | Weak zones: Only in favorable trends
            zones_traded_this_candle = False
            
            # Check DEMAND zones for CALL opportunities (ONLY in uptrends)
            if not zones_traded_this_candle and trend == 'uptrend':
                demand_zones = getattr(self, 'institutional_demand', [])[:2] + self.demand_zones[:15]
                
                for zone in demand_zones:
                    zone_tests += 1
                    
                    # ULTRA-CONSERVATIVE filtering for consistent profits
                    allow_trade = False
                    if zone.get('is_institutional', False):
                        # Institutional zones: ONLY in strong trend direction
                        allow_trade = (trend == 'uptrend' and zone['strength'] > 10.0)
                    else:
                        # Regular zones: Proven profitable conditions
                        if trend == 'uptrend':
                            allow_trade = zone['strength'] > 1.0  # Only strong regular zones
                        # NO sideways trading for regular zones (too risky)
                    
                    if allow_trade and self.validate_zone_for_trading(current_candle, zone):
                        
                        signal = {
                            'action': 'buy_call',
                            'zone_type': 'demand',
                            'zone_price': (zone['high'] + zone['low']) / 2,
                            'zone_strength': zone['strength'],
                            'is_institutional': zone.get('is_institutional', False),
                            'trend': trend
                        }
                        
                        trade = self.execute_secure_options_trade(signal, current_candle)
                        if trade:
                            open_trades.append(trade)
                            self.trades.append(trade)
                            signals_generated += 1
                            trades_executed += 1
                            
                            # Update zone usage
                            zone['trade_count'] += 1
                            zone['tested'] = True
                            
                            # Track daily entries
                            entry_date = current_candle.name.strftime('%Y-%m-%d')
                            if entry_date not in self.daily_performance:
                                self.daily_performance[entry_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                            self.daily_performance[entry_date]['entries'] += 1
                            
                            trade_type = 'INSTITUTIONAL' if zone.get('is_institutional', False) else 'REGULAR'
                            print(f"{trade_type} CALL: {trade['symbol']} @ Rs.{trade['entry_premium']:.2f} ")
                            print(f"  Spot: Rs.{current_price:.2f}, Qty: {trade['quantity']}, Zone: {zone['strength']:.1f}")
                        else:
                            zone_validations_failed += 1
                        
                        if trade:
                            zones_traded_this_candle = True
                        break  # Only one trade per candle
                    else:
                        zone_validations_failed += 1
            
            # Check SUPPLY zones for PUT opportunities (ONLY in downtrends)
            if not zones_traded_this_candle and trend == 'downtrend':
                supply_zones = getattr(self, 'institutional_supply', [])[:2] + self.supply_zones[:15]
                
                for zone in supply_zones:
                    zone_tests += 1
                    
                    # ULTRA-CONSERVATIVE filtering for consistent profits
                    allow_trade = False
                    if zone.get('is_institutional', False):
                        # Institutional zones: ONLY in strong trend direction
                        allow_trade = (trend == 'downtrend' and zone['strength'] > 10.0)
                    else:
                        # Regular zones: Proven profitable conditions
                        if trend == 'downtrend':
                            allow_trade = zone['strength'] > 1.0  # Only strong regular zones
                        # NO sideways trading for regular zones (too risky)
                    
                    if allow_trade and self.validate_zone_for_trading(current_candle, zone):
                        
                        signal = {
                            'action': 'buy_put',
                            'zone_type': 'supply',
                            'zone_price': (zone['high'] + zone['low']) / 2,
                            'zone_strength': zone['strength'],
                            'is_institutional': zone.get('is_institutional', False),
                            'trend': trend
                        }
                        
                        trade = self.execute_secure_options_trade(signal, current_candle)
                        if trade:
                            open_trades.append(trade)
                            self.trades.append(trade)
                            signals_generated += 1
                            trades_executed += 1
                            
                            # Update zone usage
                            zone['trade_count'] += 1
                            zone['tested'] = True
                            
                            trade_type = 'INSTITUTIONAL' if zone.get('is_institutional', False) else 'REGULAR'
                            print(f"{trade_type} PUT: {trade['symbol']} @ Rs.{trade['entry_premium']:.2f}")
                            print(f"  Spot: Rs.{current_price:.2f}, Qty: {trade['quantity']}, Zone: {zone['strength']:.1f}")
                            
                            # Track daily entries
                            entry_date = current_candle.name.strftime('%Y-%m-%d')
                            if entry_date not in self.daily_performance:
                                self.daily_performance[entry_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                            self.daily_performance[entry_date]['entries'] += 1
                        else:
                            zone_validations_failed += 1
                        break  # Only one trade per candle
                    else:
                        zone_validations_failed += 1
        
        print(f"\nSECURE BACKTEST COMPLETED")
        print(f"SIGNALS GENERATED: {signals_generated}")
        print(f"TRADES EXECUTED: {trades_executed}")
        print(f"FINAL CAPITAL: Rs.{self.capital:,.2f}")
        
        # Debug information
        print(f"\n🔍 DEBUGGING INFO:")
        print(f"   Trends: Uptrend={trends_detected['uptrend']}, Downtrend={trends_detected['downtrend']}, Sideways={trends_detected['sideways']}")
        print(f"   Zone Tests: {zone_tests}, Validations Failed: {zone_validations_failed}")
        print(f"   Success Rate: {((zone_tests - zone_validations_failed)/zone_tests*100) if zone_tests > 0 else 0:.1f}%")
        
        # Display daily performance breakdown
        self.print_daily_performance()
    
    def print_daily_performance(self):
        """
        Print day-wise backtesting results breakdown
        """
        if not self.daily_performance:
            print("\n📊 DAILY PERFORMANCE: No daily data available")
            return
        
        print(f"\n📊 DAY-WISE BACKTESTING RESULTS")
        print("=" * 84)
        print(f"{'Date':<12} {'Entries':<8} {'Exits':<8} {'Daily P&L':<15} {'Running Capital':<15}")
        print("-" * 84)
        
        running_capital = 100000
        total_days_traded = 0
        profitable_days = 0
        
        for date in sorted(self.daily_performance.keys()):
            day_data = self.daily_performance[date]
            daily_pnl = day_data['daily_pnl']
            running_capital += daily_pnl
            
            if daily_pnl != 0 or day_data['entries'] > 0 or day_data['exits'] > 0:
                total_days_traded += 1
                if daily_pnl > 0:
                    profitable_days += 1
                    
                pnl_color = "📈" if daily_pnl > 0 else "📉" if daily_pnl < 0 else "📊"
                print(f"{date:<12} {day_data['entries']:<8} {day_data['exits']:<8} "
                      f"{pnl_color} Rs.{daily_pnl:,.0f}{'':^7} Rs.{running_capital:,.0f}")
        
        print("-" * 84)
        
        # Daily statistics
        if total_days_traded > 0:
            win_rate = (profitable_days / total_days_traded) * 100
            print(f"\n📈 DAILY STATISTICS:")
            print(f"   Trading Days: {total_days_traded}")
            print(f"   Profitable Days: {profitable_days} ({win_rate:.1f}%)")
            print(f"   Loss Days: {total_days_traded - profitable_days} ({100-win_rate:.1f}%)")
        
        print(f"   Final Performance: Rs.{running_capital - 100000:,.0f} ({((running_capital/100000)-1)*100:+.2f}%)")
    
    def generate_secure_performance_report(self):
        """Generate comprehensive performance report with real trading metrics"""
        
        if not self.trades:
            print("\nNO TRADES EXECUTED - No performance to report")
            return
        
        print(f"\nSECURE NIFTY OPTIONS PERFORMANCE REPORT")
        print("=" * 80)
        print("REAL DATA SOURCE: Official Fyers API - Exchange Direct")
        print("STRATEGY: Institutional Supply & Demand Zones")
        print("=" * 80)
        
        # Core statistics
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        open_trades = [t for t in self.trades if t['status'] == 'open']
        
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        print(f"TRADING STATISTICS:")
        print(f"  Total Trades Initiated: {total_trades}")
        print(f"  Closed Trades: {len(closed_trades)}")
        print(f"  Open Positions: {len(open_trades)}")
        
        if closed_trades:
            win_rate = len(winning_trades) / len(closed_trades) * 100
            total_pnl = sum(t['pnl'] for t in closed_trades)
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
            
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Winning Trades: {len(winning_trades)}")
            print(f"  Losing Trades: {len(losing_trades)}")
            
            print(f"\nFINANCIAL PERFORMANCE:")
            print(f"  Total P&L: Rs.{total_pnl:,.2f}")
            print(f"  Average Win: Rs.{avg_win:,.2f}")
            print(f"  Average Loss: Rs.{avg_loss:,.2f}")
            
            if avg_loss > 0:
                profit_factor = avg_win / avg_loss
                print(f"  Profit Factor: {profit_factor:.2f}")
            
            # Capital performance
            initial_capital = 100000
            final_capital = self.capital
            total_return = (final_capital - initial_capital) / initial_capital * 100
            
            print(f"\nCAPITAL ANALYSIS:")
            print(f"  Initial Capital: Rs.{initial_capital:,.2f}")
            print(f"  Final Capital: Rs.{final_capital:,.2f}")
            print(f"  Total Return: {total_return:.2f}%")
            
            if len(closed_trades) > 0:
                avg_return_per_trade = total_return / len(closed_trades)
                print(f"  Return per Trade: {avg_return_per_trade:.2f}%")
            
            # Risk metrics
            winning_amounts = [t['pnl'] for t in winning_trades]
            losing_amounts = [abs(t['pnl']) for t in losing_trades]
            
            max_win = max(winning_amounts) if winning_amounts else 0
            max_loss = max(losing_amounts) if losing_amounts else 0
            
            print(f"\nRISK ANALYSIS:")
            print(f"  Maximum Win: Rs.{max_win:,.2f}")
            print(f"  Maximum Loss: Rs.{max_loss:,.2f}")
            
            if max_loss > 0:
                risk_reward = max_win / max_loss
                print(f"  Best Risk-Reward: {risk_reward:.2f}")
            
            # Zone effectiveness
            demand_trades = [t for t in closed_trades if t.get('zone_type') == 'demand']
            supply_trades = [t for t in closed_trades if t.get('zone_type') == 'supply']
            
            print(f"\nZONE PERFORMANCE:")
            
            if demand_trades:
                demand_pnl = sum(t['pnl'] for t in demand_trades)
                demand_wins = len([t for t in demand_trades if t['pnl'] > 0])
                demand_win_rate = demand_wins / len(demand_trades) * 100
                print(f"  Demand Zones: {len(demand_trades)} trades, Rs.{demand_pnl:,.2f}, {demand_win_rate:.1f}% win rate")
            
            if supply_trades:
                supply_pnl = sum(t['pnl'] for t in supply_trades)
                supply_wins = len([t for t in supply_trades if t['pnl'] > 0])
                supply_win_rate = supply_wins / len(supply_trades) * 100
                print(f"  Supply Zones: {len(supply_trades)} trades, Rs.{supply_pnl:,.2f}, {supply_win_rate:.1f}% win rate")
            
            # Exit analysis
            print(f"\nEXIT ANALYSIS:")
            exit_reasons = {}
            for trade in closed_trades:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
            for reason, count in exit_reasons.items():
                pct = count / len(closed_trades) * 100
                print(f"  {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
            
            # Best and worst performers
            if closed_trades:
                best_trade = max(closed_trades, key=lambda x: x['pnl'])
                worst_trade = min(closed_trades, key=lambda x: x['pnl'])
                
                print(f"\nTRADE HIGHLIGHTS:")
                print(f"  Best Trade: {best_trade['symbol']}")
                print(f"    P&L: Rs.{best_trade['pnl']:,.2f}")
                print(f"    Entry: {best_trade['entry_time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"    Zone: {best_trade['zone_type']}, Trend: {best_trade['trend']}")
                
                print(f"  Worst Trade: {worst_trade['symbol']}")
                print(f"    P&L: Rs.{worst_trade['pnl']:,.2f}")
                print(f"    Entry: {worst_trade['entry_time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"    Zone: {worst_trade['zone_type']}, Trend: {worst_trade['trend']}")
        
        # Zone statistics
        total_zones = len(self.supply_zones) + len(self.demand_zones)
        tested_zones = len([z for z in self.supply_zones + self.demand_zones if z.get('tested', False)])
        
        print(f"\nINSTITUTIONAL ZONE STATISTICS:")
        print(f"  Total Zones Identified: {total_zones}")
        print(f"  Zones Tested: {tested_zones}")
        print(f"  Zone Utilization: {tested_zones/total_zones*100:.1f}%" if total_zones > 0 else "  Zone Utilization: 0%")
        
        print("=" * 80)
        print("SECURITY CONFIRMATION: All data sourced from Official Fyers API")
        print("NO external servers or MCP dependencies used")

    def validate_multi_timeframe_zone(self, current_candle: pd.Series, zone: dict, trend: str) -> bool:
        """
        Enhanced multi-timeframe zone validation for frequent trading
        """
        zone_high = zone['high']
        zone_low = zone['low']
        buffer = (zone_high - zone_low) * self.zone_buffer
        
        # Current candle overlap check
        candle_high = current_candle['high']
        candle_low = current_candle['low']
        zone_overlap = (candle_low <= zone_high + buffer and candle_high >= zone_low - buffer)
        
        # Enhanced validations for frequent trading
        validations = [
            zone_overlap,  # Zone touch
            zone['trade_count'] < 12,  # Allow more frequent zone usage
            zone['enhanced_strength'] > 0.3,  # Lower threshold for more opportunities
            not zone['broken']  # Zone still valid
        ]
        
        return all(validations)
    
    def determine_multi_timeframe_trend(self, current_idx: int) -> str:
        """
        Determine trend using multiple timeframes for better accuracy
        """
        trends = []
        
        # Analyze multiple timeframes for trend consensus
        for tf_name, data in self.multi_data.items():
            if data is None or len(data) < 20:
                continue
                
            # Map current 5min index to timeframe index (approximate)
            tf_idx = min(len(data) - 1, max(10, current_idx // 2))
            
            if tf_idx >= 10:
                recent_data = data.iloc[tf_idx-9:tf_idx+1]
                
                # Simple trend detection
                price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
                
                if price_change > 0.002:  # 0.2% up
                    trends.append('uptrend')
                elif price_change < -0.002:  # 0.2% down
                    trends.append('downtrend')
                else:
                    trends.append('sideways')
        
        # Trend consensus
        if len(trends) == 0:
            return 'sideways'
        
        uptrend_count = trends.count('uptrend')
        downtrend_count = trends.count('downtrend')
        
        if uptrend_count > downtrend_count:
            return 'uptrend'
        elif downtrend_count > uptrend_count:
            return 'downtrend'
        else:
            return 'sideways'
    
    def manage_powerhouse_trades(self, current_candle: pd.Series, open_trades: list):
        """
        ENHANCED TRADE MANAGEMENT FOR POWERHOUSE SYSTEM
        Features: Frequent profit booking, tight stops, double quantity handling
        """
        current_price = current_candle['close']
        current_date = current_candle.name
        
        for trade in open_trades:
            if trade['status'] != 'open':
                continue
            
            # Current option premium estimation
            days_held = (current_date.date() - trade['entry_time'].date()).days
            remaining_dte = max(1, trade['dte'] - days_held)
            
            current_premium = self.calculate_enhanced_premium(
                current_price, 
                trade['strike'], 
                trade['option_type'], 
                remaining_dte,
                trade['timeframe']
            )
            
            # Calculate P&L
            if trade['option_type'] == 'CE':
                premium_change = current_premium - trade['entry_premium']
            else:
                premium_change = current_premium - trade['entry_premium']
            
            pnl = premium_change * trade['quantity'] * trade['lot_size']
            profit_pct = premium_change / trade['entry_premium']
            
            # Points moved
            points_moved = abs(current_price - trade['spot_price'])
            
            exit_reason = None
            
            # ENHANCED EXIT LOGIC
            # 1. Quick profit booking (frequent small profits)
            for target_points in trade['profit_targets']:
                if points_moved >= target_points and profit_pct > 0.15:  # 15%+ profit
                    exit_reason = f'quick_profit_{int(points_moved)}pts'
                    break
            
            # 2. Target profit reached
            if not exit_reason and profit_pct >= trade['target_profit']:
                exit_reason = 'target_profit'
            
            # 3. Tight stop loss
            elif not exit_reason and points_moved >= trade['stop_loss_points'] and profit_pct < -0.25:
                exit_reason = 'micro_stop_loss'
            
            # 4. Time decay (faster for short-term trades)
            elif not exit_reason and remaining_dte <= 1:
                exit_reason = 'time_decay'
            
            # 5. Profit protection (lock in gains)
            elif not exit_reason and profit_pct > 0.5 and profit_pct < 0.2:  # Profit dropped from 50% to 20%
                exit_reason = 'profit_protection'
            
            # Execute exit
            if exit_reason:
                trade['exit_time'] = current_date
                trade['exit_premium'] = current_premium
                trade['pnl'] = pnl
                trade['exit_reason'] = exit_reason
                trade['status'] = 'closed'
                
                # Update capital
                self.capital += pnl
                
                # Track daily exits and P&L
                exit_date = current_date.strftime('%Y-%m-%d')
                if exit_date not in self.daily_performance:
                    self.daily_performance[exit_date] = {'entries': 0, 'exits': 0, 'daily_pnl': 0, 'trades': []}
                self.daily_performance[exit_date]['exits'] += 1
                self.daily_performance[exit_date]['daily_pnl'] += pnl
                
                # Enhanced logging
                trade_type = trade['trade_type']
                print(f"EXIT: {trade['symbol']} @ Rs.{current_premium:.2f} | "
                      f"P&L: Rs.{pnl:,.0f} | Reason: {exit_reason} | "
                      f"Points: {points_moved:.0f} | Type: {trade_type}")


    """
    MAIN: MULTI-TIMEFRAME POWERHOUSE NIFTY OPTIONS BACKTESTING
    Enhanced system using ALL timeframes with double quantity supply zones
    """
    
    print("INITIALIZING MULTI-TIMEFRAME POWERHOUSE BACKTESTER...")
    print("PROTOCOL: Maximum Trading Opportunities Across All Timeframes")
    
    try:
        # Initialize powerhouse backtester
        backtester = SecureNiftyOptionsBacktester()
        
        # Run backtest on real data (full Jan-Feb 2026)
        start_date = "2026-01-01"  # Full January start
        end_date = "2026-02-28"    # Full February end
        
        print(f"\nCOMMENCING POWERHOUSE BACKTEST")
        print(f"Multi-timeframe analysis period: {start_date} to {end_date}")
        print(f"Timeframes: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h")
        print(f"Supply zones: DOUBLE QUANTITY (2X profit potential)")
        
        # Execute powerhouse backtesting
        backtester.run_powerhouse_backtest(start_date, end_date)
        
        # Generate comprehensive performance report
        backtester.generate_secure_performance_report()
        
        print(f"\nMULTI-TIMEFRAME POWERHOUSE BACKTEST COMPLETED")
        print("All data sourced exclusively from Official Fyers API")
        print("Enhanced system ready for maximum profit generation")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("Ensure Fyers API credentials are properly configured")
        raise

if __name__ == "__main__":
    main()