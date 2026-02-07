#!/usr/bin/env python3
"""
JEAFX ALERT & NOTIFICATION SYSTEM
Multi-channel alerting with advanced filtering

🚨 FEATURES:
- Multiple alert channels (Console, File, Database, Email, Telegram)
- Smart alert filtering and throttling
- Risk-based alert prioritization
- Performance milestone notifications
- System health monitoring
- Custom alert rules engine
"""

import smtplib
import requests
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for email functionality
    MimeText = None
    MimeMultipart = None
import threading
import time
from collections import deque, defaultdict

class AlertLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    SYSTEM = "SYSTEM"
    TRADING = "TRADING"
    RISK = "RISK"
    PERFORMANCE = "PERFORMANCE"
    MARKET = "MARKET"

class AlertChannel(Enum):
    CONSOLE = "CONSOLE"
    FILE = "FILE"
    DATABASE = "DATABASE"
    EMAIL = "EMAIL"
    TELEGRAM = "TELEGRAM"
    WEBHOOK = "WEBHOOK"

@dataclass
class Alert:
    """Alert message structure"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str
    data: Dict = field(default_factory=dict)
    acknowledged: bool = False
    channels: List[AlertChannel] = field(default_factory=list)

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    condition: str  # Python expression
    level: AlertLevel
    alert_type: AlertType
    channels: List[AlertChannel]
    throttle_minutes: int = 0
    enabled: bool = True

class JeafxAlertSystem:
    """
    Comprehensive JEAFX Alert and Notification System
    """
    
    def __init__(self, config_file: str = "alert_config.json"):
        self.config = self._load_alert_config(config_file)
        
        # Alert storage
        self.alerts_history: deque = deque(maxlen=10000)
        self.active_alerts: Dict[str, Alert] = {}
        
        # Throttling
        self.throttle_tracking: Dict[str, datetime] = {}
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        self._initialize_default_rules()
        
        # Channel handlers
        self.channel_handlers = {
            AlertChannel.CONSOLE: self._handle_console_alert,
            AlertChannel.FILE: self._handle_file_alert,
            AlertChannel.DATABASE: self._handle_database_alert,
            AlertChannel.EMAIL: self._handle_email_alert,
            AlertChannel.TELEGRAM: self._handle_telegram_alert,
            AlertChannel.WEBHOOK: self._handle_webhook_alert
        }
        
        # Setup
        self._setup_database()
        self._setup_logging()
        
        # Background processing
        self.is_running = False
        self._start_background_processor()
        
        self.logger.info("🚨 JEAFX Alert System Initialized")
        
    def _load_alert_config(self, config_file: str) -> Dict:
        """Load alert configuration"""
        
        default_config = {
            "channels": {
                "console": {"enabled": True},
                "file": {"enabled": True, "log_file": "jeafx_alerts.log"},
                "database": {"enabled": True, "db_file": "jeafx_alerts.db"},
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "telegram": {
                    "enabled": False,
                    "bot_token": "",
                    "chat_id": ""
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "headers": {}
                }
            },
            "filtering": {
                "min_level": "INFO",
                "max_alerts_per_minute": 30,
                "throttle_duplicate_minutes": 5
            },
            "formatting": {
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
                "include_data": True,
                "compact_mode": False
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                
        return default_config
        
    def _setup_database(self):
        """Setup alert database"""
        
        if not self.config['channels']['database']['enabled']:
            return
            
        db_file = self.config['channels']['database']['db_file']
        self.alert_db = sqlite3.connect(db_file, check_same_thread=False)
        
        cursor = self.alert_db.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                level TEXT,
                alert_type TEXT,
                title TEXT,
                message TEXT,
                data TEXT,
                acknowledged BOOLEAN,
                channels TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_summary (
                date DATE PRIMARY KEY,
                total_alerts INTEGER,
                critical_alerts INTEGER,
                error_alerts INTEGER,
                warning_alerts INTEGER,
                info_alerts INTEGER,
                debug_alerts INTEGER
            )
        ''')
        
        self.alert_db.commit()
        
    def _setup_logging(self):
        """Setup logging system"""
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('JEAFX_ALERTS')
        
        # File handler for alerts
        if self.config['channels']['file']['enabled']:
            file_handler = logging.FileHandler(self.config['channels']['file']['log_file'])
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        
        default_rules = [
            AlertRule(
                rule_id="portfolio_drawdown",
                name="Portfolio Drawdown Alert",
                condition="portfolio_metrics.get('max_drawdown', 0) > 0.1",  # >10% drawdown
                level=AlertLevel.WARNING,
                alert_type=AlertType.RISK,
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE, AlertChannel.EMAIL],
                throttle_minutes=30
            ),
            AlertRule(
                rule_id="position_stop_loss",
                name="Stop Loss Hit",
                condition="trade_data.get('exit_reason') == 'STOP_LOSS'",
                level=AlertLevel.INFO,
                alert_type=AlertType.TRADING,
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE],
                throttle_minutes=0
            ),
            AlertRule(
                rule_id="high_profit_trade",
                name="High Profit Trade",
                condition="trade_data.get('pnl', 0) > 5000",
                level=AlertLevel.INFO,
                alert_type=AlertType.PERFORMANCE,
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE, AlertChannel.TELEGRAM],
                throttle_minutes=0
            ),
            AlertRule(
                rule_id="system_error",
                name="System Error",
                condition="system_data.get('error_count', 0) > 5",
                level=AlertLevel.ERROR,
                alert_type=AlertType.SYSTEM,
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE, AlertChannel.EMAIL],
                throttle_minutes=10
            ),
            AlertRule(
                rule_id="portfolio_milestone",
                name="Portfolio Milestone",
                condition="portfolio_metrics.get('total_return', 0) > 0.1",  # >10% return
                level=AlertLevel.INFO,
                alert_type=AlertType.PERFORMANCE,
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE, AlertChannel.TELEGRAM],
                throttle_minutes=60
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
            
    def _start_background_processor(self):
        """Start background alert processor"""
        
        self.is_running = True
        processor_thread = threading.Thread(target=self._background_processor, daemon=True)
        processor_thread.start()
        
    def _background_processor(self):
        """Background processor for alert maintenance"""
        
        while self.is_running:
            try:
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Update daily summaries
                self._update_daily_summary()
                
                # Process any pending alerts
                self._process_pending_alerts()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Background processor error: {e}")
                
    def create_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        title: str,
        message: str,
        data: Optional[Dict] = None,
        channels: Optional[List[AlertChannel]] = None
    ) -> Alert:
        """Create and send alert"""
        
        alert_id = f"{alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            level=level,
            alert_type=alert_type,
            title=title,
            message=message,
            data=data or {},
            channels=channels or self._get_default_channels(level)
        )
        
        # Check if alert should be throttled
        if self._should_throttle_alert(alert):
            self.logger.debug(f"Alert throttled: {alert_id}")
            return alert
            
        # Check alert level filtering
        if not self._should_send_alert(alert):
            self.logger.debug(f"Alert filtered: {alert_id}")
            return alert
            
        # Send alert through channels
        self._send_alert(alert)
        
        # Store alert
        self.alerts_history.append(alert)
        if level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]:
            self.active_alerts[alert_id] = alert
            
        return alert
        
    def _get_default_channels(self, level: AlertLevel) -> List[AlertChannel]:
        """Get default channels based on alert level"""
        
        channels = [AlertChannel.CONSOLE, AlertChannel.FILE, AlertChannel.DATABASE]
        
        if level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            if self.config['channels']['email']['enabled']:
                channels.append(AlertChannel.EMAIL)
                
        if level == AlertLevel.CRITICAL:
            if self.config['channels']['telegram']['enabled']:
                channels.append(AlertChannel.TELEGRAM)
                
        return channels
        
    def _should_throttle_alert(self, alert: Alert) -> bool:
        """Check if alert should be throttled"""
        
        # Check global rate limit
        current_time = datetime.now()
        recent_alerts = [
            a for a in self.alerts_history 
            if (current_time - a.timestamp).total_seconds() < 60
        ]
        
        if len(recent_alerts) >= self.config['filtering']['max_alerts_per_minute']:
            return True
            
        # Check duplicate throttling
        throttle_key = f"{alert.alert_type.value}_{alert.title}"
        if throttle_key in self.throttle_tracking:
            last_sent = self.throttle_tracking[throttle_key]
            throttle_minutes = self.config['filtering']['throttle_duplicate_minutes']
            
            if (current_time - last_sent).total_seconds() < throttle_minutes * 60:
                return True
                
        self.throttle_tracking[throttle_key] = current_time
        return False
        
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert meets sending criteria"""
        
        min_level = AlertLevel(self.config['filtering']['min_level'])
        alert_levels = {
            AlertLevel.DEBUG: 0,
            AlertLevel.INFO: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.ERROR: 3,
            AlertLevel.CRITICAL: 4
        }
        
        return alert_levels[alert.level] >= alert_levels[min_level]
        
    def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        
        for channel in alert.channels:
            try:
                if channel in self.channel_handlers:
                    self.channel_handlers[channel](alert)
                    
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel.value}: {e}")
                
    def _handle_console_alert(self, alert: Alert):
        """Handle console alert"""
        
        if not self.config['channels']['console']['enabled']:
            return
            
        # Format message with colors
        level_colors = {
            AlertLevel.DEBUG: '\033[36m',      # Cyan
            AlertLevel.INFO: '\033[32m',       # Green
            AlertLevel.WARNING: '\033[33m',    # Yellow
            AlertLevel.ERROR: '\033[31m',      # Red
            AlertLevel.CRITICAL: '\033[35m'    # Magenta
        }
        
        level_icons = {
            AlertLevel.DEBUG: '🔧',
            AlertLevel.INFO: 'ℹ️',
            AlertLevel.WARNING: '⚠️',
            AlertLevel.ERROR: '❌',
            AlertLevel.CRITICAL: '🚨'
        }
        
        color = level_colors.get(alert.level, '')
        icon = level_icons.get(alert.level, '')
        reset = '\033[0m'
        
        timestamp = alert.timestamp.strftime(self.config['formatting']['timestamp_format'])
        
        console_msg = f"{color}{icon} [{timestamp}] {alert.level.value} - {alert.title}: {alert.message}{reset}"
        
        print(console_msg)
        
    def _handle_file_alert(self, alert: Alert):
        """Handle file alert"""
        
        if not self.config['channels']['file']['enabled']:
            return
            
        timestamp = alert.timestamp.strftime(self.config['formatting']['timestamp_format'])
        
        log_msg = f"[{timestamp}] {alert.level.value} - {alert.alert_type.value} - {alert.title}: {alert.message}"
        
        if self.config['formatting']['include_data'] and alert.data:
            log_msg += f" | Data: {json.dumps(alert.data, default=str)}"
            
        self.logger.info(log_msg)
        
    def _handle_database_alert(self, alert: Alert):
        """Handle database alert"""
        
        if not self.config['channels']['database']['enabled']:
            return
            
        cursor = self.alert_db.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (alert_id, timestamp, level, alert_type, title, 
                              message, data, acknowledged, channels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id, alert.timestamp, alert.level.value, alert.alert_type.value,
            alert.title, alert.message, json.dumps(alert.data, default=str),
            alert.acknowledged, json.dumps([c.value for c in alert.channels])
        ))
        
        self.alert_db.commit()
        
    def _handle_email_alert(self, alert: Alert):
        """Handle email alert"""
        
        email_config = self.config['channels']['email']
        
        if not email_config['enabled'] or not email_config['recipients'] or not MimeText:
            return
            
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"JEAFX Alert: {alert.title}"
            
            # Create body
            body = f"""
JEAFX Trading System Alert

Level: {alert.level.value}
Type: {alert.alert_type.value}
Time: {alert.timestamp.strftime(self.config['formatting']['timestamp_format'])}

Title: {alert.title}
Message: {alert.message}

"""
            
            if alert.data:
                body += f"\nAdditional Data:\n{json.dumps(alert.data, indent=2, default=str)}\n"
                
            body += "\n-- \nJEAFX Trading System\nAutomated Alert"
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.sendmail(email_config['username'], email_config['recipients'], msg.as_string())
            server.quit()
            
            self.logger.debug(f"Email alert sent: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
            
    def _handle_telegram_alert(self, alert: Alert):
        """Handle Telegram alert"""
        
        telegram_config = self.config['channels']['telegram']
        
        if not telegram_config['enabled'] or not telegram_config['bot_token']:
            return
            
        try:
            # Format message
            level_emoji = {
                AlertLevel.DEBUG: '🔧',
                AlertLevel.INFO: 'ℹ️',
                AlertLevel.WARNING: '⚠️',
                AlertLevel.ERROR: '❌',
                AlertLevel.CRITICAL: '🚨'
            }
            
            emoji = level_emoji.get(alert.level, '📝')
            timestamp = alert.timestamp.strftime("%H:%M:%S")
            
            message = f"{emoji} *JEAFX Alert*\n\n"
            message += f"*Level:* {alert.level.value}\n"
            message += f"*Type:* {alert.alert_type.value}\n"
            message += f"*Time:* {timestamp}\n\n"
            message += f"*{alert.title}*\n{alert.message}"
            
            if alert.data and self.config['formatting']['include_data']:
                message += f"\n\n```json\n{json.dumps(alert.data, indent=2, default=str)}\n```"
                
            # Send via Telegram Bot API
            url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
            
            payload = {
                'chat_id': telegram_config['chat_id'],
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.debug(f"Telegram alert sent: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Telegram alert failed: {e}")
            
    def _handle_webhook_alert(self, alert: Alert):
        """Handle webhook alert"""
        
        webhook_config = self.config['channels']['webhook']
        
        if not webhook_config['enabled'] or not webhook_config['url']:
            return
            
        try:
            payload = {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'alert_type': alert.alert_type.value,
                'title': alert.title,
                'message': alert.message,
                'data': alert.data
            }
            
            headers = {'Content-Type': 'application/json'}
            headers.update(webhook_config.get('headers', {}))
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.debug(f"Webhook alert sent: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
            
    def _cleanup_old_alerts(self):
        """Clean up old acknowledged alerts"""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove old acknowledged alerts
        to_remove = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.acknowledged and alert.timestamp < cutoff_time
        ]
        
        for alert_id in to_remove:
            del self.active_alerts[alert_id]
            
    def _update_daily_summary(self):
        """Update daily alert summary"""
        
        if not self.config['channels']['database']['enabled']:
            return
            
        today = datetime.now().date()
        
        # Count alerts by level for today
        today_alerts = [
            a for a in self.alerts_history
            if a.timestamp.date() == today
        ]
        
        if not today_alerts:
            return
            
        level_counts = defaultdict(int)
        for alert in today_alerts:
            level_counts[alert.level.value] += 1
            
        cursor = self.alert_db.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO alert_summary 
            (date, total_alerts, critical_alerts, error_alerts, 
             warning_alerts, info_alerts, debug_alerts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            today,
            len(today_alerts),
            level_counts.get('CRITICAL', 0),
            level_counts.get('ERROR', 0),
            level_counts.get('WARNING', 0),
            level_counts.get('INFO', 0),
            level_counts.get('DEBUG', 0)
        ))
        
        self.alert_db.commit()
        
    def _process_pending_alerts(self):
        """Process any pending alert processing"""
        
        # Evaluate custom alert rules
        self._evaluate_alert_rules()
        
    def _evaluate_alert_rules(self):
        """Evaluate custom alert rules against current data"""
        
        # This would be called with current system data
        # For demo purposes, we'll skip the actual evaluation
        pass
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert"""
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            
            # Update in database
            if self.config['channels']['database']['enabled']:
                cursor = self.alert_db.cursor()
                cursor.execute(
                    'UPDATE alerts SET acknowledged = ? WHERE alert_id = ?',
                    (True, alert_id)
                )
                self.alert_db.commit()
                
            self.logger.info(f"Alert acknowledged: {alert_id}")
            return True
            
        return False
        
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active (unacknowledged) alerts"""
        
        return [alert for alert in self.active_alerts.values() if not alert.acknowledged]
        
    def get_alert_history(self, hours: int = 24, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get alert history for specified period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = [
            alert for alert in self.alerts_history
            if alert.timestamp >= cutoff_time
        ]
        
        if level:
            history = [alert for alert in history if alert.level == level]
            
        return sorted(history, key=lambda x: x.timestamp, reverse=True)
        
    def get_alert_statistics(self, days: int = 7) -> Dict:
        """Get alert statistics for specified period"""
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        period_alerts = [
            alert for alert in self.alerts_history
            if alert.timestamp >= cutoff_time
        ]
        
        if not period_alerts:
            return {}
            
        # Count by level
        level_counts = defaultdict(int)
        for alert in period_alerts:
            level_counts[alert.level.value] += 1
            
        # Count by type
        type_counts = defaultdict(int)
        for alert in period_alerts:
            type_counts[alert.alert_type.value] += 1
            
        # Daily distribution
        daily_counts = defaultdict(int)
        for alert in period_alerts:
            daily_counts[alert.timestamp.date().isoformat()] += 1
            
        return {
            'total_alerts': len(period_alerts),
            'active_alerts': len(self.get_active_alerts()),
            'level_distribution': dict(level_counts),
            'type_distribution': dict(type_counts),
            'daily_distribution': dict(daily_counts),
            'avg_daily_alerts': len(period_alerts) / days,
            'period_start': cutoff_time.isoformat(),
            'period_end': datetime.now().isoformat()
        }
        
    def test_alert_channels(self) -> Dict[str, bool]:
        """Test all configured alert channels"""
        
        results = {}
        
        test_alert = Alert(
            alert_id="TEST_ALERT",
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            alert_type=AlertType.SYSTEM,
            title="Alert System Test",
            message="This is a test alert to verify all channels are working correctly.",
            data={"test": True}
        )
        
        for channel in AlertChannel:
            try:
                if channel in self.channel_handlers:
                    # Check if channel is enabled
                    channel_config = self.config['channels'].get(channel.value.lower(), {})
                    
                    if channel_config.get('enabled', False) or channel == AlertChannel.CONSOLE:
                        self.channel_handlers[channel](test_alert)
                        results[channel.value] = True
                    else:
                        results[channel.value] = False  # Disabled
                else:
                    results[channel.value] = False  # Not implemented
                    
            except Exception as e:
                self.logger.error(f"Channel test failed for {channel.value}: {e}")
                results[channel.value] = False
                
        return results
        
    def stop(self):
        """Stop the alert system"""
        
        self.is_running = False
        
        if hasattr(self, 'alert_db'):
            self.alert_db.close()
            
        self.logger.info("🚨 JEAFX Alert System Stopped")

# Convenience functions for common alerts
def send_trading_alert(alert_system: JeafxAlertSystem, message: str, data: Optional[Dict] = None):
    """Send a trading-related alert"""
    
    return alert_system.create_alert(
        level=AlertLevel.INFO,
        alert_type=AlertType.TRADING,
        title="Trading Event",
        message=message,
        data=data
    )
    
def send_risk_alert(alert_system: JeafxAlertSystem, message: str, level: AlertLevel = AlertLevel.WARNING, data: Optional[Dict] = None):
    """Send a risk-related alert"""
    
    return alert_system.create_alert(
        level=level,
        alert_type=AlertType.RISK,
        title="Risk Alert",
        message=message,
        data=data
    )
    
def send_performance_alert(alert_system: JeafxAlertSystem, message: str, data: Optional[Dict] = None):
    """Send a performance-related alert"""
    
    return alert_system.create_alert(
        level=AlertLevel.INFO,
        alert_type=AlertType.PERFORMANCE,
        title="Performance Milestone",
        message=message,
        data=data
    )
    
def send_system_alert(alert_system: JeafxAlertSystem, message: str, level: AlertLevel = AlertLevel.ERROR, data: Optional[Dict] = None):
    """Send a system-related alert"""
    
    return alert_system.create_alert(
        level=level,
        alert_type=AlertType.SYSTEM,
        title="System Alert",
        message=message,
        data=data
    )

def main():
    """Demo the alert system"""
    
    print("🚨 JEAFX ALERT SYSTEM - DEMO")
    print("="*50)
    
    # Initialize alert system
    alert_system = JeafxAlertSystem()
    
    # Test all channels
    print("\n🧪 TESTING ALERT CHANNELS")
    print("-"*35)
    
    test_results = alert_system.test_alert_channels()
    
    for channel, result in test_results.items():
        status_icon = "✅" if result else "❌"
        print(f"   {status_icon} {channel}: {'Working' if result else 'Failed/Disabled'}")
        
    # Demo different alert types
    print(f"\n📢 SENDING DEMO ALERTS")
    print("-"*30)
    
    # Trading alerts
    send_trading_alert(alert_system, "New BUY signal generated for NIFTY", {
        "symbol": "NSE:NIFTY50-INDEX",
        "signal_type": "BUY", 
        "confidence": 85.5,
        "entry_price": 19500
    })
    
    send_trading_alert(alert_system, "Position closed with profit", {
        "symbol": "NSE:BANKNIFTY-INDEX",
        "pnl": 2500,
        "pnl_percent": 2.8
    })
    
    # Risk alerts
    send_risk_alert(alert_system, "Portfolio heat approaching limit", AlertLevel.WARNING, {
        "current_heat": 75,
        "limit": 80,
        "active_positions": 4
    })
    
    # Performance alerts
    send_performance_alert(alert_system, "Portfolio reached 10% return milestone!", {
        "total_return": 10.2,
        "portfolio_value": 110200,
        "days_to_achieve": 45
    })
    
    # System alerts
    send_system_alert(alert_system, "High memory usage detected", AlertLevel.WARNING, {
        "memory_usage": 85.5,
        "available_memory": "2.1 GB"
    })
    
    time.sleep(2)  # Let alerts process
    
    # Show statistics
    print(f"\n📊 ALERT STATISTICS")
    print("-"*25)
    
    stats = alert_system.get_alert_statistics(days=1)
    
    print(f"   📈 Total Alerts: {stats.get('total_alerts', 0)}")
    print(f"   🔴 Active Alerts: {stats.get('active_alerts', 0)}")
    print(f"   📊 Alert Types: {stats.get('type_distribution', {})}")
    print(f"   ⚠️ Alert Levels: {stats.get('level_distribution', {})}")
    
    # Show active alerts
    active_alerts = alert_system.get_active_alerts()
    
    if active_alerts:
        print(f"\n🔴 ACTIVE ALERTS")
        print("-"*20)
        
        for alert in active_alerts:
            level_icon = {
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌", 
                AlertLevel.CRITICAL: "🚨"
            }.get(alert.level, "📝")
            
            print(f"   {level_icon} [{alert.timestamp.strftime('%H:%M:%S')}] {alert.title}")
            
    # Stop the system
    alert_system.stop()
    
    print(f"\n✅ Alert System Demo Complete!")

if __name__ == "__main__":
    main()