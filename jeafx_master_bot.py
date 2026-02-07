#!/usr/bin/env python3
"""
JEAFX MASTER TRADING BOT
Complete automated trading system integration

🤖 FEATURES:
- Complete JEAFX system integration
- Telegram bot interface
- Real-time monitoring
- Automated trading execution
- Risk management
- Performance tracking
- Alert system integration
- Multi-timeframe analysis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Telegram Bot
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import threading
import time
import schedule

# Import our systems
from jeafx_advanced_system import AdvancedJeafxSystem, AdvancedSignal, AdvancedZone
from jeafx_portfolio_manager import JeafxPortfolioManager, PortfolioState
from jeafx_risk_manager import JeafxRiskManager, RiskLevel
from jeafx_alert_system import JeafxAlertSystem, AlertLevel, AlertType, send_trading_alert, send_performance_alert, send_risk_alert

@dataclass
class BotConfig:
    """Bot configuration"""
    telegram_token: str
    authorized_users: List[int]
    trading_enabled: bool = True
    monitoring_enabled: bool = True
    alert_enabled: bool = True
    auto_start: bool = False
    update_interval: int = 300  # 5 minutes

class JeafxMasterBot:
    """
    JEAFX Master Trading Bot - Complete System Integration
    """
    
    def __init__(self, config_file: str = "bot_config.json"):
        # Load configuration
        self.config = self._load_bot_config(config_file)
        
        # Initialize all systems
        self.jeafx_system = AdvancedJeafxSystem()
        self.portfolio_manager = JeafxPortfolioManager()
        self.risk_manager = JeafxRiskManager()
        self.alert_system = JeafxAlertSystem()
        
        # Bot state
        self.is_running = False
        self.bot_start_time = datetime.now()
        
        # Performance tracking
        self.session_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'alerts_sent': 0
        }
        
        # Setup
        self._setup_logging()
        
        # Initialize Telegram bot if token provided
        if self.config.telegram_token:
            self._setup_telegram_bot()
        else:
            self.logger.warning("⚠️ No Telegram token - Bot interface disabled")
            
        self.logger.info("🤖 JEAFX Master Trading Bot Initialized")
        
    def _load_bot_config(self, config_file: str) -> BotConfig:
        """Load bot configuration"""
        
        default_config = {
            "telegram_token": "",
            "authorized_users": [],
            "trading_enabled": True,
            "monitoring_enabled": True,
            "alert_enabled": True,
            "auto_start": False,
            "update_interval": 300
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                
        return BotConfig(**default_config)
        
    def _setup_logging(self):
        """Setup logging system"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('jeafx_master_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('JEAFX_MASTER_BOT')
        
    def _setup_telegram_bot(self):
        """Setup Telegram bot"""
        
        self.application = Application.builder().token(self.config.telegram_token).build()
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("portfolio", self._cmd_portfolio))
        self.application.add_handler(CommandHandler("positions", self._cmd_positions))
        self.application.add_handler(CommandHandler("signals", self._cmd_signals))
        self.application.add_handler(CommandHandler("performance", self._cmd_performance))
        self.application.add_handler(CommandHandler("risk", self._cmd_risk))
        self.application.add_handler(CommandHandler("alerts", self._cmd_alerts))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        
        # Trading commands
        self.application.add_handler(CommandHandler("trading_start", self._cmd_trading_start))
        self.application.add_handler(CommandHandler("trading_stop", self._cmd_trading_stop))
        self.application.add_handler(CommandHandler("trading_pause", self._cmd_trading_pause))
        self.application.add_handler(CommandHandler("emergency_stop", self._cmd_emergency_stop))
        
        # Callback handler for inline keyboards
        self.application.add_handler(CallbackQueryHandler(self._button_callback))
        
    async def _is_authorized(self, update: Update) -> bool:
        """Check if user is authorized"""
        
        if not self.config.authorized_users:
            return True  # No restriction if no users specified
            
        user_id = update.effective_user.id
        return user_id in self.config.authorized_users
        
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        welcome_msg = """
🤖 **JEAFX Master Trading Bot**

Welcome to the complete automated trading system!

**Available Commands:**
📊 /status - System status
💼 /portfolio - Portfolio overview  
📈 /positions - Active positions
⚡ /signals - Recent signals
📊 /performance - Performance metrics
⚠️ /risk - Risk analysis
🚨 /alerts - Recent alerts
❓ /help - Command help

**Trading Controls:**
🚀 /trading_start - Start trading
🛑 /trading_stop - Stop trading  
⏸️ /trading_pause - Pause trading
🚨 /emergency_stop - Emergency stop

Ready to start automated trading! 🚀
"""
        
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')
        
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status command handler"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        # Get system status
        portfolio_status = self.portfolio_manager.get_portfolio_status()
        
        state_icon = {
            PortfolioState.ACTIVE: "🟢",
            PortfolioState.PAUSED: "🟡",
            PortfolioState.STOPPED: "🔴",
            PortfolioState.EMERGENCY: "🚨"
        }
        
        uptime = datetime.now() - self.bot_start_time
        uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
        
        status_msg = f"""
🤖 **JEAFX Master Bot Status**

**System Status:**
{state_icon.get(self.portfolio_manager.state, '⚪')} Trading: {self.portfolio_manager.state.value}
🔧 Bot Uptime: {uptime_str}
📊 Monitoring: {'✅' if self.config.monitoring_enabled else '❌'}
🚨 Alerts: {'✅' if self.config.alert_enabled else '❌'}

**Portfolio Status:**
💰 Total Value: ₹{portfolio_status['portfolio_metrics']['total_value']:,.0f}
📈 Return: {portfolio_status['portfolio_metrics']['total_return']:.2%}
💵 Cash: ₹{portfolio_status['portfolio_metrics']['cash_balance']:,.0f}
🎯 Positions: {portfolio_status['active_positions']}

**Session Stats:**
⚡ Signals: {self.session_stats['signals_generated']}
💼 Trades: {self.session_stats['trades_executed']}
💰 P&L: ₹{self.session_stats['total_pnl']:+,.0f}
🚨 Alerts: {self.session_stats['alerts_sent']}
"""
        
        # Create inline keyboard
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh")],
            [InlineKeyboardButton("📊 Portfolio", callback_data="show_portfolio"),
             InlineKeyboardButton("📈 Performance", callback_data="show_performance")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(status_msg, parse_mode='Markdown', reply_markup=reply_markup)
        
    async def _cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Portfolio command handler"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        status = self.portfolio_manager.get_portfolio_status()
        metrics = status['portfolio_metrics']
        
        portfolio_msg = f"""
💼 **Portfolio Overview**

**Portfolio Metrics:**
💰 Total Value: ₹{metrics['total_value']:,.0f}
💵 Cash Balance: ₹{metrics['cash_balance']:,.0f}
📊 Invested: ₹{metrics['total_value'] - metrics['cash_balance']:,.0f}
📈 Total Return: {metrics['total_return']:.2%}
📊 Daily Return: {metrics['daily_return']:.2%}

**P&L Breakdown:**
💚 Unrealized: ₹{metrics['unrealized_pnl']:+,.0f}
💙 Realized: ₹{metrics['realized_pnl']:+,.0f}

**Performance Metrics:**
🎯 Win Rate: {metrics['win_rate']:.1%}
⚡ Profit Factor: {metrics['profit_factor']:.2f}
📉 Max Drawdown: {metrics['max_drawdown']:.2%}
📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

**Position Summary:**
🎯 Active Positions: {metrics['active_positions']}
📊 Total Trades: {metrics['total_trades']}
"""
        
        await update.message.reply_text(portfolio_msg, parse_mode='Markdown')
        
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Positions command handler"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        active_positions = self.portfolio_manager.active_positions
        
        if not active_positions:
            await update.message.reply_text("📋 No active positions")
            return
            
        positions_msg = "📈 **Active Positions**\n\n"
        
        for pos_id, pos in active_positions.items():
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            pnl_icon = "💚" if unrealized_pnl > 0 else "📉" if unrealized_pnl < 0 else "⚪"
            
            symbol_name = pos['symbol'].split(':')[-1].replace('-EQ', '').replace('-INDEX', '')
            
            positions_msg += f"""
{pnl_icon} **{symbol_name}**
   Type: {pos['position_type']}
   Quantity: {pos['quantity']}
   Entry: ₹{pos['entry_price']:.2f}
   Current: ₹{pos['current_price']:.2f}
   P&L: ₹{unrealized_pnl:+,.0f}
   Value: ₹{pos['quantity'] * pos['current_price']:,.0f}

"""
        
        await update.message.reply_text(positions_msg, parse_mode='Markdown')
        
    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Signals command handler"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        # Get recent signals for watchlist symbols
        signals_msg = "⚡ **Recent Trading Signals**\n\n"
        
        watchlist = self.portfolio_manager.watchlist[:3]  # Limit to 3 symbols
        
        for symbol in watchlist:
            try:
                signals = self.jeafx_system.generate_trading_signals(symbol)
                
                if signals:
                    signal = signals[0]  # Most recent signal
                    
                    signal_icon = "🟢" if signal.signal_type == "BUY" else "🔴"
                    symbol_name = symbol.split(':')[-1].replace('-EQ', '').replace('-INDEX', '')
                    
                    signals_msg += f"""
{signal_icon} **{symbol_name} - {signal.signal_type}**
   Entry: ₹{signal.entry_price:.2f}
   Target: ₹{signal.target_1:.2f}
   Stop: ₹{signal.stop_loss:.2f}
   Confidence: {signal.confidence_score:.0f}%
   R:R: 1:{signal.risk_reward_ratio:.1f}

"""
                else:
                    symbol_name = symbol.split(':')[-1].replace('-EQ', '').replace('-INDEX', '')
                    signals_msg += f"⚪ **{symbol_name}** - No signals\n\n"
                    
            except Exception as e:
                self.logger.error(f"Signal generation error for {symbol}: {e}")
                
        await update.message.reply_text(signals_msg, parse_mode='Markdown')
        
    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performance command handler"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        # Get recent trades
        trade_history = self.portfolio_manager.trade_history[-10:]  # Last 10 trades
        
        if not trade_history:
            await update.message.reply_text("📊 No completed trades yet")
            return
            
        performance_msg = "📊 **Performance Summary**\n\n"
        
        # Calculate stats
        total_pnl = sum(trade['pnl'] for trade in trade_history)
        wins = [trade for trade in trade_history if trade['pnl'] > 0]
        win_rate = len(wins) / len(trade_history) if trade_history else 0
        
        avg_win = np.mean([trade['pnl'] for trade in wins]) if wins else 0
        losses = [trade for trade in trade_history if trade['pnl'] < 0]
        avg_loss = np.mean([trade['pnl'] for trade in losses]) if losses else 0
        
        performance_msg += f"""
**Overall Stats:**
💰 Total P&L: ₹{total_pnl:+,.0f}
🎯 Win Rate: {win_rate:.1%}
📊 Total Trades: {len(trade_history)}
💚 Avg Win: ₹{avg_win:+,.0f}
📉 Avg Loss: ₹{avg_loss:+,.0f}

**Recent Trades:**
"""
        
        # Show last 5 trades
        for trade in trade_history[-5:]:
            pnl_icon = "💚" if trade['pnl'] > 0 else "📉"
            symbol_name = trade['symbol'].split(':')[-1].replace('-EQ', '').replace('-INDEX', '')
            
            performance_msg += f"{pnl_icon} {symbol_name}: ₹{trade['pnl']:+,.0f} ({trade['pnl_percent']:+.1f}%)\n"
            
        await update.message.reply_text(performance_msg, parse_mode='Markdown')
        
    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Risk command handler"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        # Calculate risk metrics
        active_positions = len(self.portfolio_manager.active_positions)
        portfolio_heat = active_positions * 15  # Simplified calculation
        
        total_risk = sum(
            pos.get('risk_amount', 0) 
            for pos in self.portfolio_manager.active_positions.values()
        )
        
        portfolio_value = self.portfolio_manager._calculate_total_portfolio_value()
        risk_percentage = (total_risk / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        # Risk level indicators
        heat_icon = "🟢" if portfolio_heat < 30 else "🟡" if portfolio_heat < 60 else "🔴"
        risk_icon = "🟢" if risk_percentage < 2 else "🟡" if risk_percentage < 5 else "🔴"
        
        risk_msg = f"""
⚠️ **Risk Analysis**

**Portfolio Risk:**
🔥 Portfolio Heat: {heat_icon} {portfolio_heat}%
💸 Total Risk: {risk_icon} ₹{total_risk:,.0f} ({risk_percentage:.1f}%)
🎯 Active Positions: {active_positions}/5 max
💰 Portfolio Value: ₹{portfolio_value:,.0f}

**Risk Limits:**
📊 Max Position Risk: 2%
🔥 Max Portfolio Heat: 60%
🎯 Max Positions: 5
⚖️ Risk:Reward Min: 1:2

**Position Breakdown:**
"""
        
        for pos in self.portfolio_manager.active_positions.values():
            symbol_name = pos['symbol'].split(':')[-1].replace('-EQ', '').replace('-INDEX', '')
            pos_risk = pos.get('risk_amount', 0)
            pos_risk_pct = (pos_risk / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            risk_msg += f"   📊 {symbol_name}: ₹{pos_risk:,.0f} ({pos_risk_pct:.1f}%)\n"
            
        await update.message.reply_text(risk_msg, parse_mode='Markdown')
        
    async def _cmd_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Alerts command handler"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        # Get recent alerts
        recent_alerts = self.alert_system.get_alert_history(hours=6)
        
        if not recent_alerts:
            await update.message.reply_text("🚨 No recent alerts")
            return
            
        alerts_msg = "🚨 **Recent Alerts**\n\n"
        
        # Show last 10 alerts
        for alert in recent_alerts[:10]:
            level_icon = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️", 
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨"
            }.get(alert.level, "📝")
            
            time_str = alert.timestamp.strftime("%H:%M")
            alerts_msg += f"{level_icon} **{time_str}** - {alert.title}\n   {alert.message}\n\n"
            
        await update.message.reply_text(alerts_msg, parse_mode='Markdown')
        
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        
        help_msg = """
❓ **JEAFX Bot Help**

**Information Commands:**
📊 `/status` - System status & overview
💼 `/portfolio` - Portfolio metrics
📈 `/positions` - Active positions
⚡ `/signals` - Recent trading signals
📊 `/performance` - Trade performance
⚠️ `/risk` - Risk analysis
🚨 `/alerts` - Recent alerts

**Trading Control:**
🚀 `/trading_start` - Start automated trading
🛑 `/trading_stop` - Stop trading (keep positions)
⏸️ `/trading_pause` - Pause new trades
🚨 `/emergency_stop` - Close all positions immediately

**Bot Features:**
✅ Real-time portfolio monitoring
✅ Automated signal generation
✅ Risk management integration
✅ Performance tracking
✅ Multi-channel alerts
✅ Position management

**Safety Features:**
🛡️ Position size limits
🛡️ Maximum drawdown protection
🛡️ Portfolio heat monitoring
🛡️ Stop loss automation
🛡️ Risk-reward enforcement

Need help? The bot monitors markets 24/7 and sends alerts for important events.
"""
        
        await update.message.reply_text(help_msg, parse_mode='Markdown')
        
    async def _cmd_trading_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start trading command"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        if not self.config.trading_enabled:
            await update.message.reply_text("⚠️ Trading is disabled in configuration")
            return
            
        self.portfolio_manager.start_automation()
        
        await update.message.reply_text("""
🚀 **Trading Started**

✅ Portfolio automation activated
✅ Signal generation enabled
✅ Risk monitoring active
✅ Position management online

The system will now:
• Scan for trading opportunities
• Execute high-confidence signals
• Monitor existing positions
• Send alerts for important events

Happy trading! 📈
""", parse_mode='Markdown')
        
    async def _cmd_trading_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop trading command"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        self.portfolio_manager.stop_automation()
        
        await update.message.reply_text("""
🛑 **Trading Stopped**

✅ New signal execution disabled
✅ Existing positions maintained
✅ Monitoring continues
✅ Manual control available

No new trades will be executed, but:
• Existing positions remain open
• Stop losses remain active
• Monitoring continues
• Manual trading possible

Use /trading_start to resume.
""", parse_mode='Markdown')
        
    async def _cmd_trading_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause trading command"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        self.portfolio_manager.pause_automation()
        
        await update.message.reply_text("""
⏸️ **Trading Paused**

✅ Signal execution paused
✅ Position monitoring active
✅ Risk management active
✅ All systems online

Trading is temporarily paused:
• No new positions opened
• Existing positions monitored
• Stop losses active
• Can resume anytime

Use /trading_start to resume.
""", parse_mode='Markdown')
        
    async def _cmd_emergency_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop command"""
        
        if not await self._is_authorized(update):
            await update.message.reply_text("❌ Unauthorized access")
            return
            
        # Confirm emergency stop
        keyboard = [
            [InlineKeyboardButton("🚨 CONFIRM EMERGENCY STOP", callback_data="emergency_confirm")],
            [InlineKeyboardButton("❌ Cancel", callback_data="emergency_cancel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text("""
🚨 **EMERGENCY STOP CONFIRMATION**

⚠️ This will immediately:
• Close ALL open positions at market price
• Stop all trading automation
• Cancel pending orders
• Put system in emergency mode

This action cannot be undone!

Are you sure you want to proceed?
""", parse_mode='Markdown', reply_markup=reply_markup)
        
    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        
        query = update.callback_query
        await query.answer()
        
        if not await self._is_authorized(update):
            await query.edit_message_text("❌ Unauthorized access")
            return
            
        if query.data == "status_refresh":
            # Refresh status
            await self._cmd_status(update, context)
            
        elif query.data == "show_portfolio":
            await self._cmd_portfolio(update, context)
            
        elif query.data == "show_performance":
            await self._cmd_performance(update, context)
            
        elif query.data == "emergency_confirm":
            # Execute emergency stop
            self.portfolio_manager.emergency_stop()
            
            await query.edit_message_text("""
🚨 **EMERGENCY STOP EXECUTED**

✅ All positions closed
✅ Trading automation stopped
✅ System in emergency mode

Check your broker account for execution confirmations.
Use /status to see current state.
Use /trading_start to restart when ready.
""", parse_mode='Markdown')
            
        elif query.data == "emergency_cancel":
            await query.edit_message_text("❌ Emergency stop cancelled. System continues normally.")
            
    async def start_bot(self):
        """Start the Telegram bot"""
        
        if not hasattr(self, 'application'):
            self.logger.error("❌ Telegram bot not configured")
            return
            
        self.logger.info("🤖 Starting Telegram bot...")
        
        # Start the bot
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        # Auto-start trading if configured
        if self.config.auto_start:
            self.logger.info("🚀 Auto-starting trading system...")
            self.portfolio_manager.start_automation()
            
        self.is_running = True
        
        # Send startup notification
        if self.config.authorized_users:
            startup_msg = """
🚀 **JEAFX Master Bot Started**

✅ All systems online
✅ Monitoring active
✅ Ready for trading

Use /status to see system overview
Use /help for available commands

Happy trading! 📈
"""
            
            for user_id in self.config.authorized_users:
                try:
                    await self.application.bot.send_message(
                        chat_id=user_id,
                        text=startup_msg,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    self.logger.error(f"Failed to send startup message to {user_id}: {e}")
                    
        self.logger.info("✅ Telegram bot started successfully")
        
    async def stop_bot(self):
        """Stop the bot"""
        
        self.is_running = False
        
        if hasattr(self, 'application'):
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            
        # Stop all systems
        self.portfolio_manager.stop_automation()
        self.alert_system.stop()
        
        self.logger.info("🛑 JEAFX Master Bot Stopped")
        
    def run_background_monitoring(self):
        """Run background monitoring and maintenance"""
        
        def monitoring_loop():
            while self.is_running:
                try:
                    # Update session stats
                    self._update_session_stats()
                    
                    # Check for important alerts
                    self._check_important_alerts()
                    
                    # Performance milestone checks
                    self._check_performance_milestones()
                    
                    time.sleep(self.config.update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(60)
                    
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        
    def _update_session_stats(self):
        """Update session statistics"""
        
        # Update trade count
        self.session_stats['trades_executed'] = len(self.portfolio_manager.trade_history)
        
        # Update total P&L
        self.session_stats['total_pnl'] = sum(
            trade['pnl'] for trade in self.portfolio_manager.trade_history
        )
        
        # Update alert count
        recent_alerts = self.alert_system.get_alert_history(hours=24)
        self.session_stats['alerts_sent'] = len(recent_alerts)
        
    def _check_important_alerts(self):
        """Check for important alerts to send via Telegram"""
        
        # Get recent high-priority alerts
        recent_alerts = self.alert_system.get_alert_history(hours=1)
        
        important_alerts = [
            alert for alert in recent_alerts
            if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]
        ]
        
        for alert in important_alerts:
            # Send to Telegram users (if not already sent)
            asyncio.create_task(self._send_alert_to_telegram(alert))
            
    def _check_performance_milestones(self):
        """Check for performance milestones"""
        
        portfolio_status = self.portfolio_manager.get_portfolio_status()
        metrics = portfolio_status['portfolio_metrics']
        
        # Check for 5%, 10%, 15% returns
        milestones = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        for milestone in milestones:
            if metrics['total_return'] >= milestone:
                # Send milestone alert
                send_performance_alert(
                    self.alert_system,
                    f"Portfolio reached {milestone:.0%} return milestone! 🎉",
                    {
                        'total_return': metrics['total_return'],
                        'portfolio_value': metrics['total_value'],
                        'milestone': milestone
                    }
                )
                break
                
    async def _send_alert_to_telegram(self, alert):
        """Send alert to Telegram users"""
        
        if not hasattr(self, 'application') or not self.config.authorized_users:
            return
            
        level_emoji = {
            AlertLevel.INFO: 'ℹ️',
            AlertLevel.WARNING: '⚠️',
            AlertLevel.ERROR: '❌',
            AlertLevel.CRITICAL: '🚨'
        }
        
        emoji = level_emoji.get(alert.level, '📝')
        
        message = f"""
{emoji} **JEAFX Alert**

**{alert.title}**
{alert.message}

*{alert.timestamp.strftime('%H:%M:%S')} - {alert.level.value}*
"""
        
        for user_id in self.config.authorized_users:
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                self.logger.error(f"Failed to send alert to {user_id}: {e}")

def main():
    """Run the JEAFX Master Bot"""
    
    print("🤖 JEAFX MASTER TRADING BOT")
    print("="*50)
    
    # Initialize bot
    bot = JeafxMasterBot()
    
    # Demo mode (without Telegram)
    if not bot.config.telegram_token:
        print("\n📋 DEMO MODE - No Telegram Token")
        print("-"*35)
        
        # Start systems
        print("🚀 Starting portfolio automation...")
        bot.portfolio_manager.start_automation()
        
        # Start monitoring
        print("👁️ Starting background monitoring...")
        bot.run_background_monitoring()
        
        # Show status
        print("\n📊 SYSTEM STATUS")
        print("-"*20)
        
        status = bot.portfolio_manager.get_portfolio_status()
        print(f"💰 Portfolio Value: ₹{status['portfolio_metrics']['total_value']:,.0f}")
        print(f"📈 Return: {status['portfolio_metrics']['total_return']:.2%}")
        print(f"🎯 Active Positions: {status['active_positions']}")
        print(f"📊 State: {bot.portfolio_manager.state.value}")
        
        print("\n⏳ Running for 60 seconds...")
        time.sleep(60)
        
        # Show updated status
        status = bot.portfolio_manager.get_portfolio_status()
        print(f"\n📊 UPDATED STATUS")
        print(f"💰 Portfolio Value: ₹{status['portfolio_metrics']['total_value']:,.0f}")
        print(f"📈 Return: {status['portfolio_metrics']['total_return']:.2%}")
        print(f"🎯 Active Positions: {status['active_positions']}")
        
        # Stop systems
        bot.portfolio_manager.stop_automation()
        bot.alert_system.stop()
        
        print(f"\n✅ Demo Complete!")
        print(f"🎯 Ready for live trading with Telegram integration")
        
    else:
        # Full bot mode with Telegram
        print("\n🤖 STARTING TELEGRAM BOT")
        print("-"*30)
        
        async def run_bot():
            try:
                # Start background monitoring
                bot.run_background_monitoring()
                
                # Start Telegram bot
                await bot.start_bot()
                
                # Keep running
                while bot.is_running:
                    await asyncio.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n⏹️ Shutting down...")
                await bot.stop_bot()
                
        # Run the bot
        asyncio.run(run_bot())

if __name__ == "__main__":
    main()