# Polymarket Spike Bot

A production-ready Python trading bot for Polymarket using a **Mean Reversion / Spike Hunting** strategy.

## Strategy Overview

The bot monitors prediction market prices in real-time and detects sudden price spikes:

- **Spike UP (Pump)**: When price increases rapidly (e.g., +3% in 2 seconds), the bot places a SELL limit order, expecting the price to revert downward.
- **Spike DOWN (Dump)**: When price drops rapidly (e.g., -3% in 2 seconds), the bot places a BUY limit order, expecting the price to bounce back.

This exploits the tendency of overreactions in prediction markets to correct themselves.

## Architecture

```
spike-bot-polymarket/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management (pydantic-settings)
│   ├── logger.py          # Structured logging with rotation
│   ├── market_connector.py # Polymarket API & WebSocket
│   ├── risk_manager.py    # Risk controls & position management
│   ├── strategy_engine.py # Mean reversion strategy logic
│   └── main.py            # Entry point & orchestration
├── logs/                  # Log files (auto-created)
├── requirements.txt       # Python dependencies
├── env.example            # Example environment configuration
├── .gitignore
└── README.md
```

## Features

- **100% Asynchronous**: Built with `asyncio` and `aiohttp` for non-blocking I/O
- **Real-time Data**: WebSocket connection for live order book updates
- **Risk Management**:
  - Position sizing limits
  - Hard stop-loss per position
  - Daily loss limits
  - Trade cooldowns
  - Volatility circuit breakers
- **Paper Trading Mode**: Test strategies without risking real funds
- **Robust Error Handling**: Exponential backoff, auto-reconnection
- **Structured Logging**: File rotation, JSON format option

## Prerequisites

- Python 3.10 or higher
- A Polymarket account with funds
- Your Ethereum private key (for signing transactions)

## Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd spike-bot-polymarket

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp env.example .env

# Edit with your settings
nano .env  # or use your preferred editor
```

### 3. Get Your Polymarket Private Key

#### Option A: Export from MetaMask

1. Open MetaMask
2. Click the three dots menu → Account Details
3. Click "Export Private Key"
4. Enter your password
5. Copy the private key (without 0x prefix)

#### Option B: Create a Dedicated Trading Wallet

For security, it's recommended to create a separate wallet for trading:

1. Create a new wallet in MetaMask
2. Export its private key
3. Transfer only trading funds to this wallet

#### Important Security Notes

- **NEVER** share your private key with anyone
- **NEVER** commit your `.env` file to git
- Consider using a hardware wallet for large amounts
- The private key has full control of your wallet funds

### 4. Run the Bot

```bash
# Start in paper trading mode (default)
python -m src.main

# Or run directly
python src/main.py
```

### 5. Go Live (After Testing!)

Once you've verified the bot works correctly in paper trading:

1. Edit `.env`:
   ```
   TRADING_PAPER_TRADING=false
   ```

2. Start with small amounts:
   ```
   TRADING_ORDER_AMOUNT_USDC=10.0
   ```

3. Monitor logs closely for the first few hours

## Configuration Reference

### Trading Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_PAPER_TRADING` | `true` | Simulate trades without real execution |
| `TRADING_SPIKE_THRESHOLD_PERCENT` | `3.0` | % change to trigger spike detection |
| `TRADING_SPIKE_WINDOW_SECONDS` | `2.0` | Time window for spike detection |
| `TRADING_ORDER_AMOUNT_USDC` | `10.0` | Amount per trade in USDC |
| `TRADING_TAKE_PROFIT_PERCENT` | `2.0` | Take profit threshold |

### Risk Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `RISK_MAX_PORTFOLIO_PERCENT` | `5.0` | Max % of portfolio per trade |
| `RISK_STOP_LOSS_PERCENT` | `5.0` | Stop loss % per position |
| `RISK_DAILY_LOSS_LIMIT_USDC` | `100.0` | Max daily loss before stopping |
| `RISK_COOLDOWN_SECONDS` | `30.0` | Min time between trades per market |
| `RISK_MAX_TRADES_PER_HOUR` | `10` | Rate limit on trades |

## Running on a VPS (24/7 Operation)

### Using systemd (Recommended)

Create `/etc/systemd/system/spike-bot.service`:

```ini
[Unit]
Description=Polymarket Spike Bot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/spike-bot-polymarket
Environment=PATH=/path/to/spike-bot-polymarket/venv/bin
ExecStart=/path/to/spike-bot-polymarket/venv/bin/python -m src.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable spike-bot
sudo systemctl start spike-bot
sudo journalctl -u spike-bot -f  # View logs
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env .

CMD ["python", "-m", "src.main"]
```

```bash
docker build -t spike-bot .
docker run -d --name spike-bot --restart always spike-bot
```

## Monitoring

### Log Files

Logs are written to `logs/bot.log` with automatic rotation:

```bash
# Follow live logs
tail -f logs/bot.log

# Search for errors
grep ERROR logs/bot.log

# Check trade signals
grep "Spike detected" logs/bot.log
```

### Health Checks

The bot logs a health status every minute:

```
2024-01-15 10:30:00 | INFO | Health check | positions=2 daily_pnl=$45.23 signals=15 orders=8
```

## Risk Disclaimer

**Trading involves significant risk of loss.**

- This bot is provided as-is, without warranty
- Past performance does not guarantee future results
- Only trade with funds you can afford to lose
- Always start with paper trading
- Monitor the bot regularly
- The author is not responsible for any losses

## Troubleshooting

### "Private key is required"

Make sure your `.env` file has `POLY_PRIVATE_KEY` set correctly.

### WebSocket disconnections

The bot automatically reconnects. If issues persist, check:
- Your internet connection
- Polymarket API status
- Firewall settings

### No trades being placed

Check:
1. Paper trading mode is as expected
2. Target markets are valid and active
3. Risk limits aren't being hit (daily loss, cooldown, etc.)
4. Log file for "Trade rejected by risk manager" messages

## Development

### Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/
```

### Code Structure

- `config.py`: All configuration via pydantic-settings
- `market_connector.py`: API communication layer
- `risk_manager.py`: All risk checks centralized
- `strategy_engine.py`: Pure strategy logic
- `main.py`: Orchestration only

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

---

**Built with caution. Trade with discipline.**
