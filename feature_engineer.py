import numpy as np


class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def calculate_sma(self, length):
        self.df[f'sma_{length}'] = self.df['price'].rolling(window=length).mean()

    def calculate_ema(self, length):
        self.df[f'ema_{length}'] = self.df['price'].ewm(span=length, adjust=False).mean()

    def calculate_rsi(self, length):
        delta = self.df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        self.df[f'rsi_{length}'] = 100 - (100 / (1 + rs))

    def calculate_atr(self, length):
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['price'].shift())
        low_close = np.abs(self.df['low'] - self.df['price'].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        self.df[f'atr_{length}'] = tr.rolling(window=length).mean()

    def calculate_bollinger_bands(self, length):
        sma = self.df['price'].rolling(window=length).mean()
        std = self.df['price'].rolling(window=length).std()
        self.df[f'bb_upper_{length}'] = sma + (std * 2)
        self.df[f'bb_lower_{length}'] = sma - (std * 2)

    def calculate_stochastic_oscillator(self, length):
        low_min = self.df['low'].rolling(window=length).min()
        high_max = self.df['high'].rolling(window=length).max()
        self.df[f'stoch_{length}'] = 100 * ((self.df['price'] - low_min) / (high_max - low_min))

    def calculate_momentum(self, length):
        self.df[f'momentum_{length}'] = self.df['price'].diff(periods=length)

    def calculate_adx(self, length):
        plus_dm = self.df['high'].diff()
        minus_dm = self.df['low'].diff()
        tr = self.df[['high', 'low', 'price']].max(axis=1) - self.df[['high', 'low', 'price']].min(axis=1)
        atr = tr.rolling(window=length).mean()

        plus_di = 100 * (plus_dm.rolling(window=length).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=length).mean() / atr)
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        self.df[f'adx_{length}'] = dx.rolling(window=length).mean()

    def calculate_cci(self, length):
        tp = (self.df['high'] + self.df['low'] + self.df['price']) / 3
        ma = tp.rolling(window=length).mean()
        md = tp.rolling(window=length).apply(lambda x: np.fabs(x - x.mean()).mean())
        self.df[f'cci_{length}'] = (tp - ma) / (0.015 * md)

    def calculate_volatility(self, length):
        self.df[f'volatility_{length}'] = self.df['price'].rolling(window=length).std()

    def calculate_roc(self, length):
        self.df[f'roc_{length}'] = self.df['price'].pct_change(periods=length) * 100

    def calculate_williams_r(self, length):
        high_max = self.df['high'].rolling(window=length).max()
        low_min = self.df['low'].rolling(window=length).min()
        self.df[f'williams_r_{length}'] = (high_max - self.df['price']) / (high_max - low_min) * -100

    def add_lags(self, lag_value):
        for lag in range(1, lag_value + 1):
            self.df[f'lag_{lag}'] = self.df['price'].shift(lag)

    def calculate_price_to_sma_ratio(self, sma_lengths):
        for length in sma_lengths:
            self.df[f'price_to_sma_{length}'] = self.df['price'] / self.df[f'sma_{length}']

    def calculate_price_spikes(self, threshold):
        self.df['price_change'] = self.df['price'].pct_change()
        self.df['price_spike'] = self.df['price_change'].apply(lambda x: 1 if abs(x) > threshold else 0)

    def calculate_combinations(self, method, **param_ranges):
        param_names = param_ranges.keys()
        param_values = product(*param_ranges.values())

        combinations = []
        for values in param_values:
            params = dict(zip(param_names, values))
            method(**params)
            combinations.append((params, self.df.copy()))
        
        return combinations

    def advanced_features(self, sma_lengths=[], ema_lengths=[], rsi_lengths=[], atr_lengths=[], bb_lengths=[], macd_combinations=None,
                          stoch_lengths=[], momentum_lengths=[], adx_lengths=[], cci_lengths=[], obv=False, volatility_lengths=[],
                          roc_lengths=[], williams_r_lengths=[], lag_value=1, price_spike_threshold=0.05):
        self.df['return'] = self.df['price'].pct_change()

        for sma_length in sma_lengths:
            self.calculate_sma(sma_length)

        self.calculate_price_to_sma_ratio(sma_lengths)

        for ema_length in ema_lengths:
            self.calculate_ema(ema_length)

        for rsi_length in rsi_lengths:
            self.calculate_rsi(rsi_length)

        for atr_length in atr_lengths:
            self.calculate_atr(atr_length)

        for bb_length in bb_lengths:
            self.calculate_bollinger_bands(bb_length)

        if macd_combinations:
            slow_values = macd_combinations.get('slow_values', [26])
            fast_values = macd_combinations.get('fast_values', [12])
            signal_values = macd_combinations.get('signal_values', [9])
            macd_combinations = self.calculate_combinations(self._calculate_macd, slow=slow_values, fast=fast_values, signal=signal_values)

        for stoch_length in stoch_lengths:
            self.calculate_stochastic_oscillator(stoch_length)

        for momentum_length in momentum_lengths:
            self.calculate_momentum(momentum_length)

        for adx_length in adx_lengths:
            self.calculate_adx(adx_length)

        for cci_length in cci_lengths:
            self.calculate_cci(cci_length)

        if obv:
            self.calculate_obv()

        for volatility_length in volatility_lengths:
            self.calculate_volatility(volatility_length)

        for roc_length in roc_lengths:
            self.calculate_roc(roc_length)

        for williams_r_length in williams_r_lengths:
            self.calculate_williams_r(williams_r_length)

        self.add_lags(lag_value)
        self.calculate_price_spikes(price_spike_threshold)

        self.df.dropna(inplace=True)
        return self.df

    def _calculate_macd(self, slow=26, fast=12, signal=9):
        fast_ema = self.df['price'].ewm(span=fast, min_periods=fast).mean()
        slow_ema = self.df['price'].ewm(span=slow, min_periods=slow).mean()
        self.df['macd'] = fast_ema - slow_ema
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, min_periods=signal).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']