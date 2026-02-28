"""Tests for data adapters."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from crypto_mm_research.adapters.csv_stub import CSVStubAdapter
from crypto_mm_research.data.events import L2BookSnapshotEvent, TradeEvent


class TestCSVStubAdapter:
    """Tests for CSVStubAdapter."""
    
    def test_required_columns(self):
        ""Test getting required columns."""
        adapter = CSVStubAdapter()
        cols = adapter.get_required_columns()
        
        assert "event_ts" in cols
        assert "type" in cols
        assert "symbol" in cols
        assert "price" in cols
    
    def test_parse_book_row(self):
        ""Test parsing book row."""
        adapter = CSVStubAdapter(book_levels=3)
        
        row = pd.Series({
            "event_ts": pd.Timestamp("2024-01-01 12:00:00"),
            "type": "book",
            "symbol": "BTC-USDT",
            "bid_price_0": 50000.0,
            "bid_size_0": 1.5,
            "bid_price_1": 49999.0,
            "bid_size_1": 2.0,
            "ask_price_0": 50001.0,
            "ask_size_0": 1.0,
        })
        
        event = adapter._parse_book_row(row)
        
        assert isinstance(event, L2BookSnapshotEvent)
        assert event.symbol == "BTC-USDT"
        assert event.best_bid == 50000.0
        assert event.best_ask == 50001.0
    
    def test_parse_trade_row(self):
        ""Test parsing trade row."""
        adapter = CSVStubAdapter()
        
        row = pd.Series({
            "event_ts": pd.Timestamp("2024-01-01 12:00:00"),
            "type": "trade",
            "symbol": "BTC-USDT",
            "price": 50000.5,
            "size": 0.5,
            "side": "buy",
        })
        
        event = adapter._parse_trade_row(row)
        
        assert isinstance(event, TradeEvent)
        assert event.symbol == "BTC-USDT"
        assert event.price == 50000.5
        assert event.size == 0.5
    
    def test_symbol_mapping(self):
        ""Test symbol normalization."""
        adapter = CSVStubAdapter(
            symbol_mapping={"btcusdt": "BTC-USDT"}
        )
        
        normalized = adapter.normalize_symbol("btcusdt")
        assert normalized == "BTC-USDT"
        
        unchanged = adapter.normalize_symbol("ETH-USDT")
        assert unchanged == "ETH-USDT"


class TestExampleData:
    """Tests for example data file."""
    
    def test_example_data_exists(self):
        ""Test that example data file exists."""
        example_path = Path("data/examples/sample_data.csv")
        assert example_path.exists()
    
    def test_example_data_loadable(self):
        ""Test that example data can be loaded."""
        example_path = Path("data/examples/sample_data.csv")
        
        df = pd.read_csv(example_path)
        
        assert "event_ts" in df.columns
        assert "type" in df.columns
        assert len(df) > 0
    
    def test_adapter_with_example_data(self):
        ""Test adapter can load example data."""
        example_path = Path("data/examples/sample_data.csv")
        
        adapter = CSVStubAdapter()
        
        books = list(adapter.load_books(example_path))
        trades = list(adapter.load_trades(example_path))
        
        assert len(books) > 0 or len(trades) > 0
