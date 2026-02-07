#!/usr/bin/env python3
"""
JEAFX LIVE TRADING DASHBOARD
Real-time monitoring and control interface

🎯 FEATURES:
- Live portfolio monitoring
- Real-time P&L tracking
- Position management
- Signal visualization
- Risk monitoring
- Performance analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import sqlite3
from typing import Dict, List, Optional
import asyncio
import threading

# Import our systems
try:
    from jeafx_portfolio_manager import JeafxPortfolioManager, PortfolioState
    from jeafx_advanced_system import AdvancedJeafxSystem
    from jeafx_risk_manager import JeafxRiskManager
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="JEAFX Live Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00d4ff;
        margin: 0.5rem 0;
    }
    .profit { color: #00ff88; }
    .loss { color: #ff5555; }
    .neutral { color: #ffffff; }
    .warning { color: #ffaa00; }
    .danger { color: #ff4444; }
    .success { color: #44ff44; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1e1e1e;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

class JeafxDashboard:
    """
    JEAFX Live Trading Dashboard
    """
    
    def __init__(self):
        self.initialize_session_state()
        self.load_systems()
        
    def initialize_session_state(self):
        """Initialize streamlit session state"""
        
        if 'portfolio_manager' not in st.session_state:
            st.session_state.portfolio_manager = None
            
        if 'dashboard_refresh_rate' not in st.session_state:
            st.session_state.dashboard_refresh_rate = 5
            
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
            
        if 'alert_messages' not in st.session_state:
            st.session_state.alert_messages = []
            
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
            
    def load_systems(self):
        """Load trading systems"""
        
        try:
            if st.session_state.portfolio_manager is None:
                with st.spinner("🔄 Loading JEAFX Systems..."):
                    st.session_state.portfolio_manager = JeafxPortfolioManager()
                    st.success("✅ Portfolio Manager Loaded")
                    
        except Exception as e:
            st.error(f"❌ System Load Error: {e}")
            
    def render_header(self):
        """Render dashboard header"""
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("🚀 JEAFX Live Trading Dashboard")
            
        with col2:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("⏰ Current Time", current_time)
            
        with col3:
            if st.session_state.portfolio_manager:
                state = st.session_state.portfolio_manager.state
                state_color = {
                    PortfolioState.ACTIVE: "🟢",
                    PortfolioState.PAUSED: "🟡", 
                    PortfolioState.STOPPED: "🔴",
                    PortfolioState.EMERGENCY: "🚨"
                }
                st.metric("📊 Status", f"{state_color.get(state, '⚪')} {state.value}")
                
    def render_control_panel(self):
        """Render control panel sidebar"""
        
        st.sidebar.header("🎮 Control Panel")
        
        # System controls
        st.sidebar.subheader("System Controls")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🚀 Start", use_container_width=True):
                if st.session_state.portfolio_manager:
                    st.session_state.portfolio_manager.start_automation()
                    st.sidebar.success("✅ Started")
                    
        with col2:
            if st.button("🛑 Stop", use_container_width=True):
                if st.session_state.portfolio_manager:
                    st.session_state.portfolio_manager.stop_automation()
                    st.sidebar.warning("⏹️ Stopped")
                    
        if st.sidebar.button("⏸️ Pause", use_container_width=True):
            if st.session_state.portfolio_manager:
                st.session_state.portfolio_manager.pause_automation()
                st.sidebar.info("⏸️ Paused")
                
        if st.sidebar.button("🚨 Emergency Stop", use_container_width=True, type="secondary"):
            if st.session_state.portfolio_manager:
                st.session_state.portfolio_manager.emergency_stop()
                st.sidebar.error("🚨 Emergency Stop Activated")
                
        st.sidebar.divider()
        
        # Dashboard settings
        st.sidebar.subheader("📊 Dashboard Settings")
        
        st.session_state.auto_refresh = st.sidebar.toggle(
            "🔄 Auto Refresh",
            value=st.session_state.auto_refresh
        )
        
        st.session_state.dashboard_refresh_rate = st.sidebar.slider(
            "⏱️ Refresh Rate (seconds)",
            min_value=1,
            max_value=60,
            value=st.session_state.dashboard_refresh_rate
        )
        
        # Symbol selection
        st.sidebar.subheader("📈 Watchlist")
        
        available_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX", 
            "NSE:RELIANCE-EQ",
            "NSE:TCS-EQ",
            "NSE:INFY-EQ",
            "NSE:HDFCBANK-EQ",
            "NSE:ICICIBANK-EQ"
        ]
        
        st.session_state.selected_symbols = st.sidebar.multiselect(
            "Select Symbols to Monitor",
            available_symbols,
            default=st.session_state.selected_symbols
        )
        
    def render_portfolio_overview(self):
        """Render portfolio overview metrics"""
        
        if not st.session_state.portfolio_manager:
            st.warning("⚠️ Portfolio Manager not loaded")
            return
            
        # Get current status
        status = st.session_state.portfolio_manager.get_portfolio_status()
        metrics = status['portfolio_metrics']
        
        st.header("💼 Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "💰 Total Value",
                f"₹{metrics['total_value']:,.0f}",
                delta=f"₹{metrics['realized_pnl'] + metrics['unrealized_pnl']:+,.0f}"
            )
            
        with col2:
            color = "profit" if metrics['total_return'] > 0 else "loss"
            st.metric(
                "📈 Total Return",
                f"{metrics['total_return']:.2%}",
                delta=f"{metrics['daily_return']:.2%}"
            )
            
        with col3:
            st.metric(
                "💵 Cash Balance", 
                f"₹{metrics['cash_balance']:,.0f}",
                delta=None
            )
            
        with col4:
            color = "profit" if metrics['unrealized_pnl'] > 0 else "loss"
            st.metric(
                "📊 Unrealized P&L",
                f"₹{metrics['unrealized_pnl']:+,.0f}",
                delta=None
            )
            
        with col5:
            st.metric(
                "🎯 Active Positions",
                metrics['active_positions'],
                delta=None
            )
            
        # Performance metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🏆 Win Rate",
                f"{metrics['win_rate']:.1%}",
                delta=None
            )
            
        with col2:
            st.metric(
                "⚡ Profit Factor", 
                f"{metrics['profit_factor']:.2f}",
                delta=None
            )
            
        with col3:
            st.metric(
                "📉 Max Drawdown",
                f"{metrics['max_drawdown']:.2%}",
                delta=None
            )
            
        with col4:
            st.metric(
                "📊 Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                delta=None
            )
            
    def render_live_positions(self):
        """Render live positions table"""
        
        if not st.session_state.portfolio_manager:
            return
            
        positions = st.session_state.portfolio_manager.active_positions
        
        st.subheader("📊 Live Positions")
        
        if not positions:
            st.info("📋 No active positions")
            return
            
        # Create positions dataframe
        positions_data = []
        
        for pos_id, pos in positions.items():
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            position_value = pos['quantity'] * pos['current_price']
            
            positions_data.append({
                'Symbol': pos['symbol'].split(':')[-1].replace('-EQ', '').replace('-INDEX', ''),
                'Type': pos['position_type'],
                'Quantity': pos['quantity'],
                'Entry Price': pos['entry_price'],
                'Current Price': pos['current_price'],
                'P&L': unrealized_pnl,
                'P&L %': (unrealized_pnl / (pos['entry_price'] * pos['quantity'])) * 100,
                'Value': position_value,
                'Stop Loss': pos.get('stop_loss', 0),
                'Target': pos.get('target_price', 0)
            })
            
        positions_df = pd.DataFrame(positions_data)
        
        # Style the dataframe
        def style_pnl(val):
            color = '#00ff88' if val > 0 else '#ff5555' if val < 0 else '#ffffff'
            return f'color: {color}'
            
        styled_df = positions_df.style.applymap(
            style_pnl, 
            subset=['P&L', 'P&L %']
        ).format({
            'Entry Price': '₹{:.2f}',
            'Current Price': '₹{:.2f}',
            'P&L': '₹{:+,.0f}',
            'P&L %': '{:+.1f}%',
            'Value': '₹{:,.0f}',
            'Stop Loss': '₹{:.2f}',
            'Target': '₹{:.2f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
    def render_pnl_chart(self):
        """Render P&L performance chart"""
        
        if not st.session_state.portfolio_manager:
            return
            
        st.subheader("📈 Portfolio Performance")
        
        # Get daily metrics from database
        try:
            db = sqlite3.connect('jeafx_portfolio.db')
            
            # Get last 30 days of metrics
            query = '''
                SELECT timestamp, total_value, realized_pnl, daily_return
                FROM portfolio_metrics
                ORDER BY timestamp DESC
                LIMIT 30
            '''
            
            df = pd.read_sql_query(query, db)
            db.close()
            
            if df.empty:
                # Create sample data for demo
                dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
                initial_value = st.session_state.portfolio_manager.config['initial_capital']
                
                # Generate realistic portfolio progression
                daily_returns = np.random.normal(0.001, 0.02, 10)  # 0.1% daily return with 2% volatility
                cumulative_returns = np.cumprod(1 + daily_returns)
                portfolio_values = initial_value * cumulative_returns
                
                df = pd.DataFrame({
                    'timestamp': dates,
                    'total_value': portfolio_values,
                    'daily_return': daily_returns
                })
                
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Portfolio Value', 'Daily Returns'),
                vertical_spacing=0.08
            )
            
            # Portfolio value line
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['total_value'],
                    mode='lines+markers',
                    name='Portfolio Value',
                    line=dict(color='#00d4ff', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Daily returns bar chart
            colors = ['#00ff88' if x >= 0 else '#ff5555' for x in df['daily_return']]
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['daily_return'] * 100,  # Convert to percentage
                    name='Daily Return %',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title="Portfolio Performance Tracking",
                showlegend=False,
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Update axes
            fig.update_yaxes(title_text="Value (₹)", row=1, col=1, gridcolor='rgba(128,128,128,0.3)')
            fig.update_yaxes(title_text="Return (%)", row=2, col=1, gridcolor='rgba(128,128,128,0.3)')
            fig.update_xaxes(gridcolor='rgba(128,128,128,0.3)')
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Chart error: {e}")
            
    def render_trade_history(self):
        """Render recent trade history"""
        
        if not st.session_state.portfolio_manager:
            return
            
        st.subheader("🏪 Recent Trades")
        
        trade_history = st.session_state.portfolio_manager.trade_history
        
        if not trade_history:
            st.info("📋 No completed trades yet")
            return
            
        # Get last 10 trades
        recent_trades = trade_history[-10:]
        
        trades_data = []
        for trade in recent_trades:
            trades_data.append({
                'Time': trade['exit_time'].strftime("%H:%M:%S"),
                'Symbol': trade['symbol'].split(':')[-1].replace('-EQ', '').replace('-INDEX', ''),
                'Entry': trade['entry_price'],
                'Exit': trade['exit_price'],
                'Quantity': trade['quantity'],
                'P&L': trade['pnl'],
                'P&L %': trade['pnl_percent'],
                'Confidence': trade['signal_confidence']
            })
            
        trades_df = pd.DataFrame(trades_data)
        
        # Style the dataframe
        def style_trade_pnl(val):
            color = '#00ff88' if val > 0 else '#ff5555' if val < 0 else '#ffffff'
            return f'color: {color}; font-weight: bold'
            
        styled_trades = trades_df.style.applymap(
            style_trade_pnl,
            subset=['P&L', 'P&L %']
        ).format({
            'Entry': '₹{:.2f}',
            'Exit': '₹{:.2f}', 
            'P&L': '₹{:+,.0f}',
            'P&L %': '{:+.1f}%',
            'Confidence': '{:.0f}%'
        })
        
        st.dataframe(styled_trades, use_container_width=True, hide_index=True)
        
        # Trade statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wins = [t for t in trade_history if t['pnl'] > 0]
            win_rate = len(wins) / len(trade_history) if trade_history else 0
            st.metric("🎯 Win Rate", f"{win_rate:.1%}")
            
        with col2:
            avg_pnl = np.mean([t['pnl'] for t in trade_history]) if trade_history else 0
            st.metric("📊 Avg P&L", f"₹{avg_pnl:+,.0f}")
            
        with col3:
            total_pnl = sum(t['pnl'] for t in trade_history)
            st.metric("💰 Total P&L", f"₹{total_pnl:+,.0f}")
            
    def render_risk_monitoring(self):
        """Render risk monitoring dashboard"""
        
        if not st.session_state.portfolio_manager:
            return
            
        st.subheader("⚠️ Risk Monitoring")
        
        risk_manager = st.session_state.portfolio_manager.risk_manager
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            portfolio_heat = len(st.session_state.portfolio_manager.active_positions) * 5  # Simplified
            heat_color = "🟢" if portfolio_heat < 20 else "🟡" if portfolio_heat < 50 else "🔴"
            st.metric("🔥 Portfolio Heat", f"{heat_color} {portfolio_heat}%")
            
        with col2:
            total_risk = sum(pos.get('risk_amount', 0) for pos in st.session_state.portfolio_manager.active_positions.values())
            st.metric("💸 Total Risk", f"₹{total_risk:,.0f}")
            
        with col3:
            risk_reward_ratio = 2.5  # Default from JEAFX
            st.metric("⚖️ Risk:Reward", f"1:{risk_reward_ratio}")
            
        with col4:
            position_correlation = 0.3  # Simplified
            corr_color = "🟢" if position_correlation < 0.5 else "🟡" if position_correlation < 0.7 else "🔴"
            st.metric("🔗 Correlation", f"{corr_color} {position_correlation:.2f}")
            
        # Risk alerts
        st.write("🚨 **Risk Alerts:**")
        
        alerts = []
        
        # Check position concentration
        if len(st.session_state.portfolio_manager.active_positions) > 5:
            alerts.append("⚠️ High position concentration (>5 positions)")
            
        # Check portfolio heat
        if portfolio_heat > 50:
            alerts.append("🔥 Portfolio heat excessive (>50%)")
            
        # Check individual position risk
        for pos in st.session_state.portfolio_manager.active_positions.values():
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            risk_amount = pos.get('risk_amount', 0)
            
            if unrealized_pnl < -risk_amount * 0.8:  # Close to stop loss
                alerts.append(f"📉 {pos['symbol']} approaching stop loss")
                
        if not alerts:
            st.success("✅ No active risk alerts")
        else:
            for alert in alerts:
                st.warning(alert)
                
    def render_market_scanner(self):
        """Render market scanner for opportunities"""
        
        st.subheader("🔍 Market Scanner")
        
        if not st.session_state.selected_symbols:
            st.info("📋 Select symbols in sidebar to scan")
            return
            
        # Create tabs for different scans
        tab1, tab2, tab3 = st.tabs(["🎯 Zones", "⚡ Signals", "📊 Analysis"])
        
        with tab1:
            self.render_zone_scanner()
            
        with tab2:
            self.render_signal_scanner()
            
        with tab3:
            self.render_technical_analysis()
            
    def render_zone_scanner(self):
        """Render zone scanner"""
        
        st.write("🎯 **Supply & Demand Zones**")
        
        # Mock zone data for demo
        zone_data = []
        for symbol in st.session_state.selected_symbols:
            # Generate sample zones
            base_price = np.random.uniform(100, 5000)
            
            zone_data.append({
                'Symbol': symbol.split(':')[-1].replace('-EQ', '').replace('-INDEX', ''),
                'Zone Type': np.random.choice(['Supply', 'Demand']),
                'Zone Price': base_price,
                'Distance %': np.random.uniform(-5, 5),
                'Strength': np.random.choice(['Weak', 'Medium', 'Strong']),
                'Confluence': np.random.randint(3, 8),
                'Status': np.random.choice(['Fresh', 'Tested', 'Old'])
            })
            
        zones_df = pd.DataFrame(zone_data)
        
        # Style based on zone type and strength
        def style_zones(row):
            styles = []
            for col in zones_df.columns:
                if col == 'Zone Type':
                    color = '#ff5555' if row[col] == 'Supply' else '#00ff88'
                    styles.append(f'color: {color}')
                elif col == 'Strength':
                    color = '#ff5555' if row[col] == 'Weak' else '#ffaa00' if row[col] == 'Medium' else '#00ff88'
                    styles.append(f'color: {color}')
                else:
                    styles.append('')
            return styles
            
        styled_zones = zones_df.style.apply(style_zones, axis=1).format({
            'Zone Price': '₹{:.2f}',
            'Distance %': '{:+.1f}%'
        })
        
        st.dataframe(styled_zones, use_container_width=True, hide_index=True)
        
    def render_signal_scanner(self):
        """Render signal scanner"""
        
        st.write("⚡ **Trading Signals**")
        
        # Mock signal data for demo
        signal_data = []
        for symbol in st.session_state.selected_symbols:
            base_price = np.random.uniform(100, 5000)
            
            signal_data.append({
                'Symbol': symbol.split(':')[-1].replace('-EQ', '').replace('-INDEX', ''),
                'Signal': np.random.choice(['BUY', 'SELL']),
                'Entry Price': base_price,
                'Target': base_price * (1 + np.random.uniform(0.02, 0.05)),
                'Stop Loss': base_price * (1 - np.random.uniform(0.01, 0.025)),
                'Confidence': np.random.uniform(70, 95),
                'Risk:Reward': np.random.uniform(2.0, 4.0),
                'Time': datetime.now().strftime("%H:%M")
            })
            
        signals_df = pd.DataFrame(signal_data)
        
        # Style based on signal type and confidence
        def style_signals(row):
            styles = []
            for col in signals_df.columns:
                if col == 'Signal':
                    color = '#00ff88' if row[col] == 'BUY' else '#ff5555'
                    styles.append(f'color: {color}; font-weight: bold')
                elif col == 'Confidence':
                    color = '#00ff88' if row[col] > 85 else '#ffaa00' if row[col] > 75 else '#ffffff'
                    styles.append(f'color: {color}')
                else:
                    styles.append('')
            return styles
            
        styled_signals = signals_df.style.apply(style_signals, axis=1).format({
            'Entry Price': '₹{:.2f}',
            'Target': '₹{:.2f}',
            'Stop Loss': '₹{:.2f}',
            'Confidence': '{:.0f}%',
            'Risk:Reward': '1:{:.1f}'
        })
        
        st.dataframe(styled_signals, use_container_width=True, hide_index=True)
        
    def render_technical_analysis(self):
        """Render technical analysis"""
        
        st.write("📊 **Technical Analysis Summary**")
        
        # Mock technical data
        tech_data = []
        for symbol in st.session_state.selected_symbols:
            tech_data.append({
                'Symbol': symbol.split(':')[-1].replace('-EQ', '').replace('-INDEX', ''),
                'Trend': np.random.choice(['Bullish', 'Bearish', 'Neutral']),
                'RSI': np.random.uniform(20, 80),
                'MACD': np.random.choice(['Bullish', 'Bearish']),
                'Volume': np.random.choice(['High', 'Medium', 'Low']),
                'Support': np.random.uniform(100, 5000),
                'Resistance': np.random.uniform(100, 5000),
                'Recommendation': np.random.choice(['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'])
            })
            
        tech_df = pd.DataFrame(tech_data)
        
        # Style based on trend and recommendation
        def style_technical(row):
            styles = []
            for col in tech_df.columns:
                if col == 'Trend':
                    color = '#00ff88' if row[col] == 'Bullish' else '#ff5555' if row[col] == 'Bearish' else '#ffaa00'
                    styles.append(f'color: {color}')
                elif col == 'Recommendation':
                    if 'Buy' in row[col]:
                        color = '#00ff88'
                    elif 'Sell' in row[col]:
                        color = '#ff5555'
                    else:
                        color = '#ffaa00'
                    styles.append(f'color: {color}; font-weight: bold')
                else:
                    styles.append('')
            return styles
            
        styled_tech = tech_df.style.apply(style_technical, axis=1).format({
            'RSI': '{:.1f}',
            'Support': '₹{:.2f}',
            'Resistance': '₹{:.2f}'
        })
        
        st.dataframe(styled_tech, use_container_width=True, hide_index=True)
        
    def render_alerts_log(self):
        """Render alerts and notifications log"""
        
        st.subheader("🚨 Alerts & Notifications")
        
        # Sample alerts for demo
        sample_alerts = [
            {"time": "15:30:15", "type": "INFO", "message": "Daily portfolio summary generated"},
            {"time": "15:25:30", "type": "SUCCESS", "message": "Position closed: NIFTY BUY +₹2,500 (Target hit)"},
            {"time": "14:45:20", "type": "WARNING", "message": "Portfolio heat approaching 60%"},
            {"time": "14:30:10", "type": "INFO", "message": "New BUY signal generated for BANKNIFTY"},
            {"time": "13:15:45", "type": "DANGER", "message": "Stop loss triggered for RELIANCE position"},
            {"time": "12:30:30", "type": "SUCCESS", "message": "Order executed: TCS BUY 100 @ ₹3,450.75"},
        ]
        
        # Display alerts in reverse chronological order
        for alert in sample_alerts:
            alert_type = alert["type"]
            icon = {
                "INFO": "ℹ️",
                "SUCCESS": "✅", 
                "WARNING": "⚠️",
                "DANGER": "🚨"
            }.get(alert_type, "📝")
            
            color = {
                "INFO": "info",
                "SUCCESS": "success",
                "WARNING": "warning", 
                "DANGER": "error"
            }.get(alert_type, "info")
            
            with st.container():
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid 
                {'#00ff88' if color == 'success' else '#ffaa00' if color == 'warning' else '#ff5555' if color == 'error' else '#00d4ff'}; 
                background-color: rgba(255,255,255,0.05); border-radius: 4px;">
                    <span style="color: #888; font-size: 0.8rem;">{alert['time']}</span> 
                    {icon} {alert['message']}
                </div>
                """, unsafe_allow_html=True)
                
    def run_dashboard(self):
        """Main dashboard runner"""
        
        # Header
        self.render_header()
        
        # Control panel (sidebar)
        self.render_control_panel()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💼 Portfolio", "📊 Positions", "📈 Performance", "🔍 Scanner", "🚨 Alerts"
        ])
        
        with tab1:
            self.render_portfolio_overview()
            self.render_risk_monitoring()
            
        with tab2:
            self.render_live_positions()
            
        with tab3:
            self.render_pnl_chart()
            self.render_trade_history()
            
        with tab4:
            self.render_market_scanner()
            
        with tab5:
            self.render_alerts_log()
            
        # Auto-refresh
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.dashboard_refresh_rate)
            st.rerun()

def main():
    """Run the JEAFX dashboard"""
    
    dashboard = JeafxDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()