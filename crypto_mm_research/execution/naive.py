"""Naive execution model - simplified fill logic."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from crypto_mm_research.execution.base import (
    ExecutionModel, Order, FillEvent, OrderStatus, TimeInForce,
)


class NaiveExecutionModel(ExecutionModel):
    """Simplified execution model."""
    
    def submit_order(self, order: Order, event_ts: datetime) -> List[FillEvent]:
        order.submit_ts = event_ts
        order.active_ts = event_ts
        
        if order.time_in_force == TimeInForce.IOC:
            order.status = OrderStatus.CANCELED
        else:
            order.status = OrderStatus.ACTIVE
        
        return []
    
    def cancel_order(self, order: Order, event_ts: datetime) -> bool:
        if not order.is_active:
            return False
        
        order.cancel_ts = event_ts
        order.cancel_effective_ts = event_ts
        order.status = OrderStatus.CANCELED
        
        return True
    
    def replace_order(
        self,
        old_order: Order,
        new_price: float,
        new_size: float,
        event_ts: datetime,
    ) -> Optional[Order]:
        if not self.cancel_order(old_order, event_ts):
            return None
        
        new_order = Order(
            client_order_id=f"{old_order.client_order_id}_R",
            symbol=old_order.symbol,
            side=old_order.side,
            price=self._round_price(new_price),
            size=self._round_size(new_size),
            time_in_force=old_order.time_in_force,
            submit_ts=event_ts,
            active_ts=event_ts,
            status=OrderStatus.ACTIVE,
        )
        
        return new_order
    
    def on_book_update(
        self,
        orders: Dict[str, Order],
        book,
        event_ts: datetime,
    ) -> List[FillEvent]:
        fills = []
        
        for order in list(orders.values()):
            if not order.is_active:
                continue
            
            fill = self._check_fill(order, book, event_ts)
            if fill:
                fills.append(fill)
                order.update_fill(fill.size, fill.price, event_ts)
                
                if order.is_done:
                    del orders[order.client_order_id]
        
        return fills
    
    def on_trade(
        self,
        orders: Dict[str, Order],
        trade,
        event_ts: datetime,
    ) -> List[FillEvent]:
        fills = []
        
        for order in list(orders.values()):
            if not order.is_active:
                continue
            
            fill = self._check_trade_fill(order, trade, event_ts)
            if fill:
                fills.append(fill)
                order.update_fill(fill.size, fill.price, event_ts)
                
                if order.is_done:
                    del orders[order.client_order_id]
        
        return fills
    
    def _check_fill(
        self,
        order: Order,
        book,
        event_ts: datetime,
    ) -> Optional[FillEvent]:
        if order.side == "buy":
            if order.price >= book.best_ask:
                is_maker = order.price < book.best_ask
                return FillEvent(
                    fill_id=self._generate_fill_id(),
                    order_id=order.exchange_order_id or order.client_order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    fill_ts=event_ts,
                    arrival_ts=event_ts,
                    price=book.best_ask if is_maker else order.price,
                    size=order.remaining_size,
                    side="buy",
                    is_maker=is_maker,
                    fee=self.compute_fee(order.remaining_size * book.best_ask, is_maker),
                    mid_at_fill=book.mid_price,
                    spread_at_fill=book.spread,
                )
        else:
            if order.price <= book.best_bid:
                is_maker = order.price > book.best_bid
                return FillEvent(
                    fill_id=self._generate_fill_id(),
                    order_id=order.exchange_order_id or order.client_order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    fill_ts=event_ts,
                    arrival_ts=event_ts,
                    price=book.best_bid if is_maker else order.price,
                    size=order.remaining_size,
                    side="sell",
                    is_maker=is_maker,
                    fee=self.compute_fee(order.remaining_size * book.best_bid, is_maker),
                    mid_at_fill=book.mid_price,
                    spread_at_fill=book.spread,
                )
        
        return None
    
    def _check_trade_fill(
        self,
        order: Order,
        trade,
        event_ts: datetime,
    ) -> Optional[FillEvent]:
        if order.side == "buy" and trade.side.value < 0:
            if trade.price <= order.price:
                return FillEvent(
                    fill_id=self._generate_fill_id(),
                    order_id=order.exchange_order_id or order.client_order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    fill_ts=event_ts,
                    arrival_ts=event_ts,
                    price=trade.price,
                    size=order.remaining_size,
                    side="buy",
                    is_maker=True,
                    fee=self.compute_fee(order.remaining_size * trade.price, True),
                    mid_at_fill=trade.price,
                    spread_at_fill=0.0,
                )
        elif order.side == "sell" and trade.side.value > 0:
            if trade.price >= order.price:
                return FillEvent(
                    fill_id=self._generate_fill_id(),
                    order_id=order.exchange_order_id or order.client_order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    fill_ts=event_ts,
                    arrival_ts=event_ts,
                    price=trade.price,
                    size=order.remaining_size,
                    side="sell",
                    is_maker=True,
                    fee=self.compute_fee(order.remaining_size * trade.price, True),
                    mid_at_fill=trade.price,
                    spread_at_fill=0.0,
                )
        
        return None
