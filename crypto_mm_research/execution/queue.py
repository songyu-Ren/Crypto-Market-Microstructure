"""Queue-based execution model with realistic queue position and partial fills."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from crypto_mm_research.execution.base import (
    ExecutionModel, Order, FillEvent, OrderStatus, TimeInForce,
)


@dataclass
class QueuePosition:
    """Track queue position for an order."""
    
    order_id: str
    price: float
    side: str
    size: float
    queue_ahead: float
    queue_behind: float = 0.0
    filled_size: float = 0.0
    
    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size
    
    @property
    def queue_position_pct(self) -> float:
        total = self.queue_ahead + self.size + self.queue_behind
        if total == 0:
            return 0.0
        return self.queue_ahead / total
    
    def consume_queue(self, amount: float) -> Tuple[float, float]:
        if amount <= self.queue_ahead:
            self.queue_ahead -= amount
            return 0.0, 0.0
        
        remaining = amount - self.queue_ahead
        self.queue_ahead = 0.0
        
        fill_size = min(remaining, self.remaining_size)
        self.filled_size += fill_size
        
        remaining_after = remaining - fill_size
        
        return fill_size, max(0.0, remaining_after)


class QueueExecutionModel(ExecutionModel):
    """Queue-based execution model with realistic fill dynamics."""
    
    def __init__(
        self,
        tick_size: float = 0.1,
        lot_size: float = 0.001,
        maker_fee_rate: float = 0.0002,
        taker_fee_rate: float = 0.0005,
        queue_ahead_ratio: float = 0.5,
        allow_partial_fills: bool = True,
        trade_through_enabled: bool = True,
    ) -> None:
        super().__init__(tick_size, lot_size, maker_fee_rate, taker_fee_rate)
        
        self.queue_ahead_ratio = queue_ahead_ratio
        self.allow_partial_fills = allow_partial_fills
        self.trade_through_enabled = trade_through_enabled
        
        self._queue_positions: Dict[str, QueuePosition] = {}
        self._last_book = None
    
    def submit_order(self, order: Order, event_ts: datetime) -> List[FillEvent]:
        order.submit_ts = event_ts
        order.active_ts = event_ts
        
        if self._last_book is not None:
            self._estimate_queue_position(order)
        
        if order.time_in_force == TimeInForce.IOC:
            if order.client_order_id in self._queue_positions:
                qp = self._queue_positions[order.client_order_id]
                if qp.queue_ahead == 0:
                    order.status = OrderStatus.ACTIVE
                else:
                    order.status = OrderStatus.CANCELED
                    del self._queue_positions[order.client_order_id]
            else:
                order.status = OrderStatus.CANCELED
        else:
            order.status = OrderStatus.ACTIVE
        
        return []
    
    def _estimate_queue_position(self, order: Order) -> None:
        book = self._last_book
        
        if order.side == "buy":
            if abs(order.price - book.best_bid) < self.tick_size / 2:
                visible_size = book.bids[0, 1] if book.bids.size > 0 else 0
                queue_ahead = visible_size * self.queue_ahead_ratio
            elif order.price > book.best_bid:
                queue_ahead = 0.0
            else:
                queue_ahead = self._estimate_depth_queue(order.price, book.bids)
        else:
            if abs(order.price - book.best_ask) < self.tick_size / 2:
                visible_size = book.asks[0, 1] if book.asks.size > 0 else 0
                queue_ahead = visible_size * self.queue_ahead_ratio
            elif order.price < book.best_ask:
                queue_ahead = 0.0
            else:
                queue_ahead = self._estimate_depth_queue(order.price, book.asks)
        
        qp = QueuePosition(
            order_id=order.client_order_id,
            price=order.price,
            side=order.side,
            size=order.size,
            queue_ahead=queue_ahead,
        )
        
        self._queue_positions[order.client_order_id] = qp
    
    def _estimate_depth_queue(self, price: float, levels: np.ndarray) -> float:
        total_ahead = 0.0
        for i in range(len(levels)):
            if abs(levels[i, 0] - price) < self.tick_size / 2:
                return total_ahead * self.queue_ahead_ratio
            total_ahead += levels[i, 1]
        return total_ahead * self.queue_ahead_ratio
    
    def cancel_order(self, order: Order, event_ts: datetime) -> bool:
        if not order.is_active:
            return False
        
        order.cancel_ts = event_ts
        order.cancel_effective_ts = event_ts
        order.status = OrderStatus.CANCELED
        
        if order.client_order_id in self._queue_positions:
            del self._queue_positions[order.client_order_id]
        
        return True
    
    def replace_order(
        self,
        old_order: Order,
        new_price: float,
        new_size: float,
        event_ts: datetime,
    ) -> Optional[Order]:
        if not old_order.is_active:
            return None
        
        old_order.status = OrderStatus.PENDING_REPLACE
        
        new_order = Order(
            client_order_id=f"{old_order.client_order_id}_R{event_ts.timestamp():.0f}",
            symbol=old_order.symbol,
            side=old_order.side,
            price=self._round_price(new_price),
            size=self._round_size(new_size),
            time_in_force=old_order.time_in_force,
            submit_ts=event_ts,
            status=OrderStatus.PENDING_SUBMIT,
        )
        
        old_order.status = OrderStatus.CANCELED
        if old_order.client_order_id in self._queue_positions:
            del self._queue_positions[old_order.client_order_id]
        
        new_order.active_ts = event_ts
        new_order.status = OrderStatus.ACTIVE
        self._estimate_queue_position(new_order)
        
        return new_order
    
    def on_book_update(
        self,
        orders: Dict[str, Order],
        book,
        event_ts: datetime,
    ) -> List[FillEvent]:
        self._last_book = book
        fills = []
        
        for order in orders.values():
            if order.is_active and order.client_order_id in self._queue_positions:
                self._update_queue_position(order, book)
        
        for order in list(orders.values()):
            if not order.is_active:
                continue
            
            fill = self._check_crossed_fill(order, book, event_ts)
            if fill:
                fills.append(fill)
                order.update_fill(fill.size, fill.price, event_ts)
                
                if order.client_order_id in self._queue_positions:
                    del self._queue_positions[order.client_order_id]
                
                if order.is_done:
                    del orders[order.client_order_id]
                continue
            
            if order.client_order_id in self._queue_positions:
                qp = self._queue_positions[order.client_order_id]
                
                if qp.queue_ahead == 0 and qp.remaining_size > 0:
                    fill = self._try_queue_fill(order, qp, book, event_ts)
                    if fill:
                        fills.append(fill)
                        order.update_fill(fill.size, fill.price, event_ts)
                        
                        if qp.remaining_size <= 1e-9:
                            del self._queue_positions[order.client_order_id]
                            if order.is_done:
                                del orders[order.client_order_id]
        
        return fills
    
    def _update_queue_position(self, order: Order, book) -> None:
        qp = self._queue_positions.get(order.client_order_id)
        if qp is None:
            return
        
        if order.side == "buy":
            if book.bids.size > 0 and abs(book.bids[0, 0] - order.price) < self.tick_size / 2:
                visible_size = book.bids[0, 1]
                consumed = max(0, qp.queue_ahead - visible_size * self.queue_ahead_ratio)
                qp.queue_ahead = max(0, qp.queue_ahead - consumed * 0.5)
        else:
            if book.asks.size > 0 and abs(book.asks[0, 0] - order.price) < self.tick_size / 2:
                visible_size = book.asks[0, 1]
                consumed = max(0, qp.queue_ahead - visible_size * self.queue_ahead_ratio)
                qp.queue_ahead = max(0, qp.queue_ahead - consumed * 0.5)
    
    def _check_crossed_fill(
        self,
        order: Order,
        book,
        event_ts: datetime,
    ) -> Optional[FillEvent]:
        if order.side == "buy" and order.price >= book.best_ask:
            is_maker = False
            fill_size = order.remaining_size
            
            if not self.allow_partial_fills:
                if book.asks[0, 1] < fill_size:
                    return None
            
            fill_size = min(fill_size, book.asks[0, 1])
            
            return FillEvent(
                fill_id=self._generate_fill_id(),
                order_id=order.exchange_order_id or order.client_order_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                fill_ts=event_ts,
                arrival_ts=event_ts,
                price=book.best_ask,
                size=fill_size,
                side="buy",
                is_maker=is_maker,
                fee=self.compute_fee(fill_size * book.best_ask, is_maker),
                mid_at_fill=book.mid_price,
                spread_at_fill=book.spread,
                queue_ahead_before=0.0,
                queue_ahead_after=0.0,
            )
        
        elif order.side == "sell" and order.price <= book.best_bid:
            is_maker = False
            fill_size = order.remaining_size
            
            if not self.allow_partial_fills:
                if book.bids[0, 1] < fill_size:
                    return None
            
            fill_size = min(fill_size, book.bids[0, 1])
            
            return FillEvent(
                fill_id=self._generate_fill_id(),
                order_id=order.exchange_order_id or order.client_order_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                fill_ts=event_ts,
                arrival_ts=event_ts,
                price=book.best_bid,
                size=fill_size,
                side="sell",
                is_maker=is_maker,
                fee=self.compute_fee(fill_size * book.best_bid, is_maker),
                mid_at_fill=book.mid_price,
                spread_at_fill=book.spread,
                queue_ahead_before=0.0,
                queue_ahead_after=0.0,
            )
        
        return None
    
    def _try_queue_fill(
        self,
        order: Order,
        qp: QueuePosition,
        book,
        event_ts: datetime,
    ) -> Optional[FillEvent]:
        if qp.queue_ahead > 0:
            return None
        
        fill_size = qp.remaining_size
        if self.allow_partial_fills:
            fill_size = min(fill_size, order.size * 0.5)
        
        fill_size = self._round_size(fill_size)
        if fill_size <= 0:
            return None
        
        qp.filled_size += fill_size
        
        return FillEvent(
            fill_id=self._generate_fill_id(),
            order_id=order.exchange_order_id or order.client_order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            fill_ts=event_ts,
            arrival_ts=event_ts,
            price=order.price,
            size=fill_size,
            side=order.side,
            is_maker=True,
            fee=self.compute_fee(fill_size * order.price, True),
            mid_at_fill=book.mid_price,
            spread_at_fill=book.spread,
            queue_ahead_before=qp.queue_ahead,
            queue_ahead_after=0.0,
        )
    
    def on_trade(
        self,
        orders: Dict[str, Order],
        trade,
        event_ts: datetime,
    ) -> List[FillEvent]:
        fills = []
        
        if not self.trade_through_enabled:
            return fills
        
        for order in list(orders.values()):
            if not order.is_active:
                continue
            
            fill = self._check_trade_fill(order, trade, event_ts)
            if fill:
                fills.append(fill)
                order.update_fill(fill.size, fill.price, event_ts)
                
                if order.client_order_id in self._queue_positions:
                    qp = self._queue_positions[order.client_order_id]
                    qp.filled_size += fill.size
                    if qp.remaining_size <= 1e-9:
                        del self._queue_positions[order.client_order_id]
                
                if order.is_done:
                    del orders[order.client_order_id]
        
        return fills
    
    def _check_trade_fill(
        self,
        order: Order,
        trade,
        event_ts: datetime,
    ) -> Optional[FillEvent]:
        if order.side == "buy" and trade.side.value < 0:
            if trade.price <= order.price:
                qp = self._queue_positions.get(order.client_order_id)
                if qp and qp.queue_ahead > 0:
                    fill_size, remaining = qp.consume_queue(trade.size)
                    if fill_size > 0:
                        return FillEvent(
                            fill_id=self._generate_fill_id(),
                            order_id=order.exchange_order_id or order.client_order_id,
                            client_order_id=order.client_order_id,
                            symbol=order.symbol,
                            fill_ts=event_ts,
                            arrival_ts=event_ts,
                            price=order.price,
                            size=fill_size,
                            side="buy",
                            is_maker=True,
                            fee=self.compute_fee(fill_size * order.price, True),
                            mid_at_fill=trade.price,
                            spread_at_fill=0.0,
                            queue_ahead_before=qp.queue_ahead + fill_size,
                            queue_ahead_after=qp.queue_ahead,
                        )
                    return None
                
                fill_size = min(order.remaining_size, trade.size)
                if self.allow_partial_fills:
                    fill_size = self._round_size(fill_size)
                
                return FillEvent(
                    fill_id=self._generate_fill_id(),
                    order_id=order.exchange_order_id or order.client_order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    fill_ts=event_ts,
                    arrival_ts=event_ts,
                    price=order.price,
                    size=fill_size,
                    side="buy",
                    is_maker=True,
                    fee=self.compute_fee(fill_size * order.price, True),
                    mid_at_fill=trade.price,
                    spread_at_fill=0.0,
                )
        
        elif order.side == "sell" and trade.side.value > 0:
            if trade.price >= order.price:
                qp = self._queue_positions.get(order.client_order_id)
                if qp and qp.queue_ahead > 0:
                    fill_size, remaining = qp.consume_queue(trade.size)
                    if fill_size > 0:
                        return FillEvent(
                            fill_id=self._generate_fill_id(),
                            order_id=order.exchange_order_id or order.client_order_id,
                            client_order_id=order.client_order_id,
                            symbol=order.symbol,
                            fill_ts=event_ts,
                            arrival_ts=event_ts,
                            price=order.price,
                            size=fill_size,
                            side="sell",
                            is_maker=True,
                            fee=self.compute_fee(fill_size * order.price, True),
                            mid_at_fill=trade.price,
                            spread_at_fill=0.0,
                            queue_ahead_before=qp.queue_ahead + fill_size,
                            queue_ahead_after=qp.queue_ahead,
                        )
                    return None
                
                fill_size = min(order.remaining_size, trade.size)
                if self.allow_partial_fills:
                    fill_size = self._round_size(fill_size)
                
                return FillEvent(
                    fill_id=self._generate_fill_id(),
                    order_id=order.exchange_order_id or order.client_order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    fill_ts=event_ts,
                    arrival_ts=event_ts,
                    price=order.price,
                    size=fill_size,
                    side="sell",
                    is_maker=True,
                    fee=self.compute_fee(fill_size * order.price, True),
                    mid_at_fill=trade.price,
                    spread_at_fill=0.0,
                )
        
        return None
