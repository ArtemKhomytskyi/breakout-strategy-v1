from entry_exit import Bar, PositionDirection
from typing import Optional


class StopLossManager:
    """
    Менеджер стоп-лосса.

    Рассчитывает стоп-лосс при входе в позицию и проверяет его срабатывание на каждом баре.
    Поддерживает три режима:
    - fixed: фиксированный процент от цены входа
    - structural: за последним swing-уровнем + буфер
    - bos: за экстремумом сигнальной свечи (Break of Structure) + буфер

    Стоп-лосс фиксируется один раз при входе и дальше не меняется.
    """

    VALID_MODES = {"fixed", "structural", "bos"}

    def __init__(
        self,
        mode: str = "fixed",
        fixed_pct: float = 0.01,
        buffer_pct: float = 0.001,
    ):
        """
        Инициализация менеджера стоп-лосса.

        Args:
            mode: Режим расчёта SL ("fixed" | "structural" | "bos")
            fixed_pct: Процент от цены входа для режима fixed (например, 0.01 = 1%)
            buffer_pct: Буфер в процентах от цены входа для режимов structural и bos

        Raises:
            ValueError: Если передан некорректный mode или отрицательные/нулевые значения параметров
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Неверный mode: '{mode}'. Допустимые: {', '.join(self.VALID_MODES)}"
            )
        if fixed_pct <= 0:
            raise ValueError(f"fixed_pct должен быть > 0, получено {fixed_pct}")
        if buffer_pct < 0:
            raise ValueError(f"buffer_pct не может быть отрицательным, получено {buffer_pct}")

        self.mode = mode
        self.fixed_pct = fixed_pct
        self.buffer_pct = buffer_pct
        self.reset()

    def reset(self) -> None:
        """
        Сброс состояния менеджера.

        Вызывается после закрытия позиции, чтобы подготовить менеджер к новому входу.
        """
        self.direction: Optional[PositionDirection] = None
        self.entry_price: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.active = False

    def on_entry(
        self,
        direction: PositionDirection,
        entry_price: float,
        *,
        last_swing_high: Optional[float] = None,
        last_swing_low: Optional[float] = None,
        signal_bar: Optional[Bar] = None,
    ) -> float:
        """
        Рассчитывает и фиксирует стоп-лосс при входе в позицию.

        Args:
            direction: Направление позиции (PositionDirection.LONG или .SHORT)
            entry_price: Цена входа в позицию
            last_swing_high: Цена последнего swing high (обязательно для mode="structural")
            last_swing_low: Цена последнего swing low (обязательно для mode="structural")
            signal_bar: Сигнальная свеча BOS (обязательно для mode="bos")

        Returns:
            float: Рассчитанная цена стоп-лосса

        Raises:
            RuntimeError: Если менеджер уже активен (не вызван reset())
            ValueError: Если для выбранного режима не переданы обязательные параметры
            ValueError: Если entry_price <= 0
        """
        if self.active:
            raise RuntimeError(
                "StopLossManager уже активен! Вызови reset() перед новым входом."
            )

        if entry_price <= 0:
            raise ValueError(f"entry_price должен быть > 0, получено {entry_price}")

        self.direction = direction
        self.entry_price = entry_price

        if self.mode == "fixed":
            self.stop_price = self._fixed_sl()

        elif self.mode == "structural":
            if last_swing_high is None or last_swing_low is None:
                raise ValueError(
                    "Для режима 'structural' обязательно передать last_swing_high и last_swing_low!"
                )
            self.stop_price = self._structural_sl(last_swing_high, last_swing_low)

        elif self.mode == "bos":
            if signal_bar is None:
                raise ValueError("Для режима 'bos' обязательно передать signal_bar!")
            self.stop_price = self._bos_sl(signal_bar)

        self.active = True
        return self.stop_price

    def should_exit(self, bar: Bar) -> bool:
        """
        Проверяет, сработал ли стоп-лосс на текущем баре.

        Args:
            bar: Текущая свеча (Bar)

        Returns:
            bool: True, если цена бара пробила стоп-лосс, иначе False
        """
        if not self.active:
            return False

        if self.direction == PositionDirection.LONG:
            return bar.low <= self.stop_price  # type: ignore

        if self.direction == PositionDirection.SHORT:
            return bar.high >= self.stop_price  # type: ignore

        return False

    def _fixed_sl(self) -> float:
        """
        Рассчитывает стоп-лосс для режима fixed.

        Returns:
            float: Цена стоп-лосса
        """
        if self.direction == PositionDirection.LONG:
            return self.entry_price * (1 - self.fixed_pct)

        return self.entry_price * (1 + self.fixed_pct)

    def _structural_sl(
        self,
        last_swing_high: float,
        last_swing_low: float,
    ) -> float:
        """
        Рассчитывает стоп-лосс для режима structural (за swing-уровнем).

        Args:
            last_swing_high: Цена последнего swing high
            last_swing_low: Цена последнего swing low

        Returns:
            float: Цена стоп-лосса с учётом буфера
        """
        buffer = self.entry_price * self.buffer_pct

        if self.direction == PositionDirection.LONG:
            return last_swing_low - buffer

        return last_swing_high + buffer

    def _bos_sl(self, signal_bar: Bar) -> float:
        """
        Рассчитывает стоп-лосс для режима bos (за экстремумом сигнальной свечи).

        Args:
            signal_bar: Сигнальная свеча Break of Structure

        Returns:
            float: Цена стоп-лосса с учётом буфера
        """
        buffer = self.entry_price * self.buffer_pct

        if self.direction == PositionDirection.LONG:
            return signal_bar.low - buffer

        return signal_bar.high + buffer