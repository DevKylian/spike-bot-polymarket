#!/usr/bin/env python3
"""
Simple WebSocket test to verify Polymarket connection.
"""

import asyncio
import json
import aiohttp


async def test_websocket(token_id: str):
    """Test WebSocket connection and subscription."""
    ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    print(f"Connecting to {ws_url}...")

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            print("✅ Connected!")

            # Subscribe to the token
            subscribe_msg = {
                "assets_ids": [token_id],
                "type": "market",
            }
            await ws.send_json(subscribe_msg)
            print(f"✅ Subscribed to token: {token_id[:30]}...")

            # Listen for messages
            print("\n📡 Waiting for price updates (Ctrl+C to stop)...\n")

            msg_count = 0
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if not msg.data.strip():
                        continue

                    try:
                        data = json.loads(msg.data)
                        msg_count += 1

                        # Print the message type and summary
                        if isinstance(data, list):
                            for item in data:
                                print_event(item, msg_count)
                        else:
                            print_event(data, msg_count)

                    except json.JSONDecodeError:
                        print(f"⚠️  Invalid JSON: {msg.data[:50]}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"❌ WebSocket error: {ws.exception()}")
                    break


def print_event(data: dict, count: int):
    """Print a WebSocket event."""
    event_type = data.get("event_type") or data.get("type", "unknown")

    if event_type == "book":
        asset_id = data.get("asset_id", "")[:20]
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Handle both list and dict formats
        best_bid = 0
        best_ask = 0

        if bids:
            if isinstance(bids[0], (list, tuple)):
                best_bid = float(bids[0][0])
            elif isinstance(bids[0], dict):
                best_bid = float(bids[0].get("price", 0))

        if asks:
            if isinstance(asks[0], (list, tuple)):
                best_ask = float(asks[0][0])
            elif isinstance(asks[0], dict):
                best_ask = float(asks[0].get("price", 0))

        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        print(f"[{count}] 📊 BOOK | Bid: {best_bid:.4f} | Ask: {best_ask:.4f} | Mid: {mid:.4f}")

    elif event_type == "price_change":
        price = data.get("price", 0)
        print(f"[{count}] 💰 PRICE | {price}")

    elif event_type == "last_trade_price":
        price = data.get("price", 0)
        print(f"[{count}] 🔄 TRADE | {price}")

    else:
        print(f"[{count}] 📨 {event_type}: {str(data)[:100]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_websocket.py <token_id>")
        print("\nExample:")
        print('  python test_websocket.py "35512084566810379638827951412033582773795360811750408621002106372583271690440"')
        sys.exit(1)

    token_id = sys.argv[1]

    try:
        asyncio.run(test_websocket(token_id))
    except KeyboardInterrupt:
        print("\n\nStopped.")
