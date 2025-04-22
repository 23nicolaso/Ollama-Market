import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import random

# Define the possible states for each economic factor
class MarketTrend(Enum):
    BEAR = 0
    NEUTRAL = 1
    BULL = 2

class InflationState(Enum):
    LOW = 0
    MODERATE = 1
    HIGH = 2

class InterestRateState(Enum):
    LOW = 0
    MODERATE = 1
    HIGH = 2

class UnemploymentState(Enum):
    LOW = 0
    MODERATE = 1
    HIGH = 2

class EconomicGrowthState(Enum):
    RECESSION = 0
    SLOW = 1
    MODERATE = 2
    RAPID = 3

class Sector(Enum):
    TECHNOLOGY = 0
    HEALTHCARE = 1
    FINANCIAL = 2
    ENERGY = 3
    CONSUMER = 4
    INDUSTRIAL = 5

@dataclass
class Asset:
    name: str
    sector: Sector
    beta: float  # Market sensitivity
    dividend_yield: float
    growth_rate: float
    base_price: float
    volatility: float  # Asset-specific volatility

class EconomyState:
    def __init__(self):
        self.market_trend = MarketTrend.NEUTRAL
        self.inflation = InflationState.MODERATE
        self.interest_rate = InterestRateState.MODERATE
        self.unemployment = UnemploymentState.MODERATE
        self.economic_growth = EconomicGrowthState.MODERATE
        self.day = 0
        
        # Sector performance modifiers (relative to market)
        self.sector_performance = {
            Sector.TECHNOLOGY: 0.0,
            Sector.HEALTHCARE: 0.0,
            Sector.FINANCIAL: 0.0,
            Sector.ENERGY: 0.0,
            Sector.CONSUMER: 0.0,
            Sector.INDUSTRIAL: 0.0
        }
        
        # History tracking
        self.history = {
            'day': [],
            'market_trend': [],
            'inflation': [],
            'interest_rate': [],
            'unemployment': [],
            'economic_growth': [],
            'market_return': []
        }
        
        for sector in Sector:
            self.history[f'sector_{sector.name}'] = []

class MarkovModel:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.state = EconomyState()
        self.assets = []
        self.market_base_return = 0.08  # 8% annual expected return
        self.market_volatility = 0.15   # 15% annual volatility
        
        # Initialize transition matrices
        self._initialize_transition_matrices()
        
        # Initialize correlation matrix for economic factors
        self._initialize_correlation_matrix()
    
    def _initialize_correlation_matrix(self):
        """Initialize correlation matrix between economic variables"""
        # This correlation matrix represents relationships between:
        # [Market Trend, Inflation, Interest Rate, Unemployment, Economic Growth]
        # Values range from -1 (perfectly negative correlation) to 1 (perfectly positive correlation)
        self.correlation_matrix = np.array([
            # Mkt_Trend Inflation Int_Rate Unemp    Econ_Growth
            [1.0,      0.1,      -0.2,    -0.6,     0.7],    # Market Trend
            [0.1,      1.0,      0.7,     -0.3,     0.0],    # Inflation
            [-0.2,     0.7,      1.0,     0.1,      -0.3],   # Interest Rate
            [-0.6,     -0.3,     0.1,     1.0,      -0.8],   # Unemployment
            [0.7,      0.0,      -0.3,    -0.8,     1.0]     # Economic Growth
        ])
        
    def _initialize_transition_matrices(self):
        # Market trend transition probabilities
        self.market_trend_matrix = np.array([
            [0.70, 0.25, 0.05],  # BEAR -> BEAR, NEUTRAL, BULL
            [0.20, 0.60, 0.20],  # NEUTRAL -> BEAR, NEUTRAL, BULL
            [0.05, 0.25, 0.70]   # BULL -> BEAR, NEUTRAL, BULL
        ])
        
        # Inflation state transition probabilities
        self.inflation_matrix = np.array([
            [0.70, 0.25, 0.05],  # LOW -> LOW, MODERATE, HIGH
            [0.15, 0.70, 0.15],  # MODERATE -> LOW, MODERATE, HIGH
            [0.10, 0.30, 0.60]   # HIGH -> LOW, MODERATE, HIGH
        ])
        
        # Interest rate transition probabilities (influenced by inflation and economic growth)
        self.interest_rate_matrix = {
            # Matrices for each inflation state
            InflationState.LOW: np.array([
                [0.80, 0.18, 0.02],  # LOW -> LOW, MODERATE, HIGH
                [0.30, 0.65, 0.05],  # MODERATE -> LOW, MODERATE, HIGH
                [0.15, 0.35, 0.50]   # HIGH -> LOW, MODERATE, HIGH
            ]),
            InflationState.MODERATE: np.array([
                [0.60, 0.35, 0.05],  # LOW -> LOW, MODERATE, HIGH
                [0.15, 0.70, 0.15],  # MODERATE -> LOW, MODERATE, HIGH
                [0.05, 0.35, 0.60]   # HIGH -> LOW, MODERATE, HIGH
            ]),
            InflationState.HIGH: np.array([
                [0.40, 0.40, 0.20],  # LOW -> LOW, MODERATE, HIGH
                [0.10, 0.50, 0.40],  # MODERATE -> LOW, MODERATE, HIGH
                [0.05, 0.15, 0.80]   # HIGH -> LOW, MODERATE, HIGH
            ])
        }
        
        # Unemployment transition probabilities (influenced by economic growth - Okun's Law)
        self.unemployment_matrix = {
            # Matrices for each economic growth state
            EconomicGrowthState.RECESSION: np.array([
                [0.30, 0.40, 0.30],  # LOW -> LOW, MODERATE, HIGH
                [0.05, 0.40, 0.55],  # MODERATE -> LOW, MODERATE, HIGH
                [0.01, 0.24, 0.75]   # HIGH -> LOW, MODERATE, HIGH
            ]),
            EconomicGrowthState.SLOW: np.array([
                [0.50, 0.40, 0.10],  # LOW -> LOW, MODERATE, HIGH
                [0.20, 0.60, 0.20],  # MODERATE -> LOW, MODERATE, HIGH
                [0.05, 0.45, 0.50]   # HIGH -> LOW, MODERATE, HIGH
            ]),
            EconomicGrowthState.MODERATE: np.array([
                [0.70, 0.25, 0.05],  # LOW -> LOW, MODERATE, HIGH
                [0.30, 0.60, 0.10],  # MODERATE -> LOW, MODERATE, HIGH
                [0.15, 0.55, 0.30]   # HIGH -> LOW, MODERATE, HIGH
            ]),
            EconomicGrowthState.RAPID: np.array([
                [0.85, 0.14, 0.01],  # LOW -> LOW, MODERATE, HIGH
                [0.45, 0.50, 0.05],  # MODERATE -> LOW, MODERATE, HIGH
                [0.30, 0.60, 0.10]   # HIGH -> LOW, MODERATE, HIGH
            ])
        }
        
        # Economic growth transition probabilities
        self.economic_growth_matrix = np.array([
            [0.50, 0.35, 0.14, 0.01],  # RECESSION -> RECESSION, SLOW, MODERATE, RAPID
            [0.15, 0.50, 0.30, 0.05],  # SLOW -> RECESSION, SLOW, MODERATE, RAPID
            [0.08, 0.20, 0.60, 0.12],  # MODERATE -> RECESSION, SLOW, MODERATE, RAPID
            [0.05, 0.25, 0.50, 0.20]   # RAPID -> RECESSION, SLOW, MODERATE, RAPID
        ])
        
        # Sector performance modifiers based on economic conditions
        self.sector_modifiers = {
            # Technology sector modifiers
            Sector.TECHNOLOGY: {
                'market_trend': {MarketTrend.BEAR: -0.3, MarketTrend.NEUTRAL: 0.0, MarketTrend.BULL: 0.5},
                'interest_rate': {InterestRateState.LOW: 0.3, InterestRateState.MODERATE: 0.0, InterestRateState.HIGH: -0.4},
                'economic_growth': {
                    EconomicGrowthState.RECESSION: -0.5, 
                    EconomicGrowthState.SLOW: -0.1, 
                    EconomicGrowthState.MODERATE: 0.2, 
                    EconomicGrowthState.RAPID: 0.5
                }
            },
            # Healthcare sector modifiers
            Sector.HEALTHCARE: {
                'market_trend': {MarketTrend.BEAR: -0.1, MarketTrend.NEUTRAL: 0.0, MarketTrend.BULL: 0.2},
                'interest_rate': {InterestRateState.LOW: 0.1, InterestRateState.MODERATE: 0.0, InterestRateState.HIGH: -0.1},
                'economic_growth': {
                    EconomicGrowthState.RECESSION: 0.1, 
                    EconomicGrowthState.SLOW: 0.0, 
                    EconomicGrowthState.MODERATE: 0.0, 
                    EconomicGrowthState.RAPID: 0.1
                }
            },
            # Financial sector modifiers
            Sector.FINANCIAL: {
                'market_trend': {MarketTrend.BEAR: -0.3, MarketTrend.NEUTRAL: 0.0, MarketTrend.BULL: 0.3},
                'interest_rate': {InterestRateState.LOW: -0.2, InterestRateState.MODERATE: 0.1, InterestRateState.HIGH: 0.3},
                'economic_growth': {
                    EconomicGrowthState.RECESSION: -0.4, 
                    EconomicGrowthState.SLOW: -0.1, 
                    EconomicGrowthState.MODERATE: 0.2, 
                    EconomicGrowthState.RAPID: 0.4
                }
            },
            # Energy sector modifiers
            Sector.ENERGY: {
                'market_trend': {MarketTrend.BEAR: -0.2, MarketTrend.NEUTRAL: 0.0, MarketTrend.BULL: 0.2},
                'interest_rate': {InterestRateState.LOW: 0.0, InterestRateState.MODERATE: 0.0, InterestRateState.HIGH: -0.1},
                'economic_growth': {
                    EconomicGrowthState.RECESSION: -0.3, 
                    EconomicGrowthState.SLOW: -0.1, 
                    EconomicGrowthState.MODERATE: 0.1, 
                    EconomicGrowthState.RAPID: 0.3
                },
                'inflation': {InflationState.LOW: -0.1, InflationState.MODERATE: 0.0, InflationState.HIGH: 0.3}
            },
            # Consumer sector modifiers
            Sector.CONSUMER: {
                'market_trend': {MarketTrend.BEAR: -0.2, MarketTrend.NEUTRAL: 0.0, MarketTrend.BULL: 0.2},
                'interest_rate': {InterestRateState.LOW: 0.2, InterestRateState.MODERATE: 0.0, InterestRateState.HIGH: -0.2},
                'economic_growth': {
                    EconomicGrowthState.RECESSION: -0.3, 
                    EconomicGrowthState.SLOW: -0.1, 
                    EconomicGrowthState.MODERATE: 0.1, 
                    EconomicGrowthState.RAPID: 0.3
                },
                'unemployment': {UnemploymentState.LOW: 0.2, UnemploymentState.MODERATE: 0.0, UnemploymentState.HIGH: -0.3}
            },
            # Industrial sector modifiers
            Sector.INDUSTRIAL: {
                'market_trend': {MarketTrend.BEAR: -0.3, MarketTrend.NEUTRAL: 0.0, MarketTrend.BULL: 0.3},
                'interest_rate': {InterestRateState.LOW: 0.1, InterestRateState.MODERATE: 0.0, InterestRateState.HIGH: -0.2},
                'economic_growth': {
                    EconomicGrowthState.RECESSION: -0.4, 
                    EconomicGrowthState.SLOW: -0.2, 
                    EconomicGrowthState.MODERATE: 0.2, 
                    EconomicGrowthState.RAPID: 0.4
                }
            }
        }
    
    def add_asset(self, asset: Asset):
        """Add an asset to the simulation"""
        self.assets.append(asset)
        return len(self.assets) - 1  # Return the index of the added asset
    
    def get_economy_state(self) -> Dict:
        """Get the current state of the economy"""
        return {
            'day': self.state.day,
            'market_trend': self.state.market_trend,
            'inflation': self.state.inflation,
            'interest_rate': self.state.interest_rate,
            'unemployment': self.state.unemployment,
            'economic_growth': self.state.economic_growth,
            'sector_performance': {sector.name: perf for sector, perf in self.state.sector_performance.items()}
        }
    
    def get_noisy_state(self, noise_level=0.01) -> Dict:
        """Get a noisy version of the economy state"""
        # This emits state information with some random noise added
        true_state = self.get_economy_state()
        
        # Add noise to sector performance
        noisy_sector_performance = {}
        for sector_name, perf in true_state['sector_performance'].items():
            noise = np.random.normal(0, noise_level)
            noisy_sector_performance[sector_name] = perf + noise
        
        # States might randomly shift to adjacent states with some probability
        def noisy_enum_state(current_state, enum_class, noise_prob=0.15):
            values = list(enum_class)
            current_idx = values.index(current_state)
            
            if random.random() < noise_prob:
                # Move one step in either direction with equal probability
                possible_moves = []
                if current_idx > 0:
                    possible_moves.append(-1)  # Can move left
                if current_idx < len(values) - 1:
                    possible_moves.append(1)   # Can move right
                
                if possible_moves:
                    move = random.choice(possible_moves)
                    return values[current_idx + move]
            
            return current_state
        
        return {
            'day': true_state['day'],
            'market_trend': noisy_enum_state(true_state['market_trend'], MarketTrend),
            'inflation': noisy_enum_state(true_state['inflation'], InflationState),
            'interest_rate': noisy_enum_state(true_state['interest_rate'], InterestRateState),
            'unemployment': noisy_enum_state(true_state['unemployment'], UnemploymentState),
            'economic_growth': noisy_enum_state(true_state['economic_growth'], EconomicGrowthState),
            'sector_performance': noisy_sector_performance
        }
    
    def _apply_correlations(self):
        """Apply the correlation matrix to make transitions more realistic"""
        # Get current state indices for all factors
        current_states = [
            self.state.market_trend.value,
            self.state.inflation.value,
            self.state.interest_rate.value,
            self.state.unemployment.value,
            self.state.economic_growth.value
        ]
        
        # Define weights for how much to bias the transitions
        bias_strength = 0.2
        
        # For each variable, adjust transitions based on correlations
        for i in range(5):  # 5 economic variables
            # Skip if this is the variable being updated
            if i == 0:  # Market trend
                matrix = self.market_trend_matrix
                num_states = len(MarketTrend)
            elif i == 1:  # Inflation
                matrix = self.inflation_matrix
                num_states = len(InflationState)
            elif i == 2:  # Interest Rate
                continue  # Interest rate is handled separately via its own mechanism
            elif i == 3:  # Unemployment
                continue  # Unemployment is handled separately via Okun's Law
            elif i == 4:  # Economic Growth
                matrix = self.economic_growth_matrix
                num_states = len(EconomicGrowthState)
            
            # Calculate correlation-based bias for transition
            bias = np.zeros(num_states)
            
            for j in range(5):  # Influence from all 5 variables
                if i == j:
                    continue  # Skip self-correlation
                
                correlation = self.correlation_matrix[i, j]
                
                # Apply correlation effect
                if correlation > 0:
                    # Positive correlation: bias towards the same direction
                    if current_states[j] < num_states - 1:  # If not at max
                        bias[current_states[j]+1] += correlation * bias_strength
                    if current_states[j] > 0:  # If not at min
                        bias[current_states[j]-1] -= correlation * bias_strength
                else:  # Negative correlation
                    # Negative correlation: bias towards the opposite direction
                    if current_states[j] < num_states - 1:  # If not at max
                        bias[current_states[j]+1] -= -correlation * bias_strength
                    if current_states[j] > 0:  # If not at min
                        bias[current_states[j]-1] += -correlation * bias_strength
            
            # Apply the bias to transition matrix
            if i == 0:  # Market trend
                row = self.market_trend_matrix[current_states[i]]
                biased_row = np.clip(row + bias, 0.01, 0.99)
                biased_row /= biased_row.sum()  # Normalize
                self.market_trend_matrix[current_states[i]] = biased_row
            elif i == 1:  # Inflation
                row = self.inflation_matrix[current_states[i]]
                biased_row = np.clip(row + bias, 0.01, 0.99)
                biased_row /= biased_row.sum()  # Normalize
                self.inflation_matrix[current_states[i]] = biased_row
            elif i == 4:  # Economic Growth
                row = self.economic_growth_matrix[current_states[i]]
                biased_row = np.clip(row + bias[:len(row)], 0.01, 0.99)
                biased_row /= biased_row.sum()  # Normalize
                self.economic_growth_matrix[current_states[i]] = biased_row
    
    def _calculate_daily_market_return(self) -> float:
        """Calculate the daily market return based on current economic conditions"""
        # Convert annual return and volatility to daily values
        daily_return_base = self.market_base_return / 252  # Trading days in a year
        daily_volatility = self.market_volatility / np.sqrt(252)
        
        # Adjust base return based on market trend
        trend_modifier = {
            MarketTrend.BEAR: -0.01,
            MarketTrend.NEUTRAL: 0.0,
            MarketTrend.BULL: 0.01
        }[self.state.market_trend]
        
        # Adjust base return based on economic growth
        growth_modifier = {
            EconomicGrowthState.RECESSION: -0.005,
            EconomicGrowthState.SLOW: -0.002,
            EconomicGrowthState.MODERATE: 0.002,
            EconomicGrowthState.RAPID: 0.005
        }[self.state.economic_growth]
        
        # Adjust base return based on interest rate (inverse relationship)
        interest_modifier = {
            InterestRateState.LOW: 0.001,
            InterestRateState.MODERATE: 0.0,
            InterestRateState.HIGH: -0.002
        }[self.state.interest_rate]
        
        # Inflation effect
        inflation_modifier = {
            InflationState.LOW: 0.0005,
            InflationState.MODERATE: 0.0,
            InflationState.HIGH: -0.001
        }[self.state.inflation]
        
        # Unemployment effect
        unemployment_modifier = {
            UnemploymentState.LOW: 0.0005,
            UnemploymentState.MODERATE: 0.0,
            UnemploymentState.HIGH: -0.001
        }[self.state.unemployment]
        
        # Calculate adjusted expected return
        adjusted_daily_return = (daily_return_base + trend_modifier + growth_modifier + 
                                interest_modifier + inflation_modifier + unemployment_modifier)
        
        # Add random normal noise scaled by volatility
        return np.random.normal(adjusted_daily_return, daily_volatility)
    
    def _update_sector_performance(self, market_return: float):
        """Update sector performance based on economic conditions and market return"""
        for sector in Sector:
            # Base sector return is related to market return
            base_sector_return = market_return
            
            # Apply modifiers based on economic conditions
            modifiers = self.sector_modifiers[sector]
            
            # Default modifiers (all sectors respond to these)
            trend_effect = modifiers['market_trend'][self.state.market_trend]
            interest_effect = modifiers['interest_rate'][self.state.interest_rate]
            growth_effect = modifiers['economic_growth'][self.state.economic_growth]
            
            # Additional modifiers (sector-specific)
            total_effect = trend_effect + interest_effect + growth_effect
            
            # Add inflation effect for sectors that respond to it
            if 'inflation' in modifiers:
                inflation_effect = modifiers['inflation'][self.state.inflation]
                total_effect += inflation_effect
            
            # Add unemployment effect for sectors that respond to it
            if 'unemployment' in modifiers:
                unemployment_effect = modifiers['unemployment'][self.state.unemployment]
                total_effect += unemployment_effect
            
            # Add sector-specific random noise
            sector_volatility = 0.003  # Additional sector-specific daily volatility
            sector_noise = np.random.normal(0, sector_volatility)
            
            # Update sector performance
            self.state.sector_performance[sector] = base_sector_return + total_effect/100 + sector_noise
    
    def _apply_phillips_curve(self):
        """Apply Phillips Curve relationship (inflation vs unemployment)"""
        # Phillips Curve suggests an inverse relationship between inflation and unemployment
        # If inflation is high and unemployment is low, pressure to increase unemployment
        # If inflation is low and unemployment is high, pressure to decrease unemployment
        
        inflation_level = self.state.inflation.value
        unemployment_level = self.state.unemployment.value
        
        # Calculate Phillips curve pressure (simplified)
        phillips_diff = inflation_level - unemployment_level
        
        # Apply pressure to transition matrices
        if phillips_diff > 0:  # High inflation relative to unemployment
            # Increase chance of higher unemployment in next transition
            self.unemployment_matrix[self.state.economic_growth] *= 0.9
            # Increase probability of higher unemployment
            for i in range(len(UnemploymentState)):
                if i < len(UnemploymentState) - 1:
                    self.unemployment_matrix[self.state.economic_growth][i, i+1] *= 1.5
            
            row_sums = self.unemployment_matrix[self.state.economic_growth].sum(axis=1, keepdims=True)
            self.unemployment_matrix[self.state.economic_growth] /= row_sums
            
            # Reduce chance of higher inflation
            self.inflation_matrix *= 0.9
            # Increase probability of lower inflation
            for i in range(len(InflationState)):
                if i > 0:
                    self.inflation_matrix[i, i-1] *= 1.5
                    
            row_sums = self.inflation_matrix.sum(axis=1, keepdims=True)
            self.inflation_matrix /= row_sums
            
        elif phillips_diff < 0:  # Low inflation relative to unemployment
            # Increase chance of lower unemployment
            self.unemployment_matrix[self.state.economic_growth] *= 0.9
            # Increase probability of lower unemployment
            for i in range(len(UnemploymentState)):
                if i > 0:
                    self.unemployment_matrix[self.state.economic_growth][i, i-1] *= 1.5
                    
            row_sums = self.unemployment_matrix[self.state.economic_growth].sum(axis=1, keepdims=True)
            self.unemployment_matrix[self.state.economic_growth] /= row_sums
            
            # Increase chance of higher inflation
            self.inflation_matrix *= 0.9
            # Increase probability of higher inflation
            for i in range(len(InflationState)):
                if i < len(InflationState) - 1:
                    self.inflation_matrix[i, i+1] *= 1.5
                    
            row_sums = self.inflation_matrix.sum(axis=1, keepdims=True)
            self.inflation_matrix /= row_sums
    
    def _apply_okuns_law(self):
        """Apply Okun's Law (relationship between GDP growth and unemployment)"""
        # Okun's Law suggests that unemployment decreases when GDP growth is above trend
        # and increases when GDP growth is below trend
        
        growth_level = self.state.economic_growth.value
        unemployment_level = self.state.unemployment.value
        
        # Calculate Okun's Law effect
        # Higher growth should lead to lower unemployment and vice versa
        okun_effect = 1.0 - (growth_level / (len(EconomicGrowthState) - 1))
        
        # Adjust unemployment transition probabilities based on Okun's effect
        for growth_state in EconomicGrowthState:
            # Apply stronger effect to the current economic growth state
            if growth_state == self.state.economic_growth:
                multiplier = 1.5  # Stronger effect for current state
            else:
                multiplier = 0.5  # Weaker effect for other states
                
            matrix = self.unemployment_matrix[growth_state]
            
            # If okun_effect > 0.5, bias towards higher unemployment
            # If okun_effect < 0.5, bias towards lower unemployment
            if okun_effect > 0.5:  # Slow growth -> higher unemployment
                for i in range(len(UnemploymentState)):
                    if i < len(UnemploymentState) - 1:
                        matrix[i, i+1] *= 1.0 + (okun_effect - 0.5) * multiplier
            else:  # Fast growth -> lower unemployment
                for i in range(len(UnemploymentState)):
                    if i > 0:
                        matrix[i, i-1] *= 1.0 + (0.5 - okun_effect) * multiplier
            
            # Normalize the matrix rows
            row_sums = matrix.sum(axis=1, keepdims=True)
            self.unemployment_matrix[growth_state] = matrix / row_sums

    def _transition_state(self):
        """Update all economic states based on transition matrices"""
        # Apply correlations to bias the transition matrices
        self._apply_correlations()
        
        # Apply Okun's Law effects
        self._apply_okuns_law()
        
        # Apply Phillips Curve effects
        self._apply_phillips_curve()
        
        # Transition market trend
        current_market_idx = self.state.market_trend.value
        next_market_idx = np.random.choice(len(MarketTrend), p=self.market_trend_matrix[current_market_idx])
        self.state.market_trend = MarketTrend(next_market_idx)
        
        # Transition economic growth
        current_growth_idx = self.state.economic_growth.value
        next_growth_idx = np.random.choice(len(EconomicGrowthState), p=self.economic_growth_matrix[current_growth_idx])
        self.state.economic_growth = EconomicGrowthState(next_growth_idx)
        
        # Transition unemployment (influenced by economic growth via Okun's Law)
        current_unemployment_idx = self.state.unemployment.value
        next_unemployment_idx = np.random.choice(
            len(UnemploymentState), 
            p=self.unemployment_matrix[self.state.economic_growth][current_unemployment_idx]
        )
        self.state.unemployment = UnemploymentState(next_unemployment_idx)
        
        # Transition inflation
        current_inflation_idx = self.state.inflation.value
        next_inflation_idx = np.random.choice(len(InflationState), p=self.inflation_matrix[current_inflation_idx])
        self.state.inflation = InflationState(next_inflation_idx)
        
        # Transition interest rate (influenced by inflation)
        current_interest_idx = self.state.interest_rate.value
        next_interest_idx = np.random.choice(
            len(InterestRateState), 
            p=self.interest_rate_matrix[self.state.inflation][current_interest_idx]
        )
        self.state.interest_rate = InterestRateState(next_interest_idx)        

    def update_on_sentiment(self, sentiment: float, importance: int):
        """Update economy state based on sentiment (0 to 1) and importance (1-10)"""
        if not 0 <= sentiment <= 1:
            raise ValueError("Sentiment must be between 0 and 1")
        if not 1 <= importance <= 10:
            raise ValueError("Importance must be between 1 and 10")

        # Extreme cases: importance 10 with very low/high sentiment
        if importance == 10:
            if sentiment <= 0.1:
                # Seriously bad update
                self.state.market_trend = MarketTrend.BEAR
                self.state.economic_growth = EconomicGrowthState.RECESSION
                self.state.unemployment = UnemploymentState.HIGH
                self.state.inflation = InflationState.HIGH
                self.state.interest_rate = InterestRateState.HIGH
                for sector in Sector:
                    self.state.sector_performance[sector] -= 0.05  # Significant sector downturn
            elif sentiment >= 0.9:
                # Seriously good update
                self.state.market_trend = MarketTrend.BULL
                self.state.economic_growth = EconomicGrowthState.RAPID
                self.state.unemployment = UnemploymentState.LOW
                self.state.inflation = InflationState.LOW
                self.state.interest_rate = InterestRateState.LOW
                for sector in Sector:
                    self.state.sector_performance[sector] += 0.05  # Significant sector upturn
            return

        # Normal case: slight adjustments
        # Scale impact based on importance
        impact = (sentiment - 0.5) * (importance / 10) * 0.2  # Max adjustment of 0.2 at importance 10

        # Adjust transition matrices
        def adjust_matrix(matrix, direction, strength):
            adjusted = matrix.copy()
            for i in range(adjusted.shape[0]):
                if direction > 0:  # Bias towards higher states
                    if i < adjusted.shape[1] - 1:
                        adjusted[i, i+1] += strength
                        adjusted[i, i] -= strength
                else:  # Bias towards lower states
                    if i > 0:
                        adjusted[i, i-1] += strength
                        adjusted[i, i] -= strength
                adjusted[i] = np.clip(adjusted[i], 0.01, 0.99)
                adjusted[i] /= adjusted[i].sum()  # Normalize
            return adjusted

        # Update transition matrices
        strength = abs(impact)
        direction = 1 if impact > 0 else -1

        self.market_trend_matrix = adjust_matrix(self.market_trend_matrix, direction, strength)
        self.inflation_matrix = adjust_matrix(self.inflation_matrix, direction, strength)
        self.economic_growth_matrix = adjust_matrix(self.economic_growth_matrix, direction, strength)
        
        for inflation_state in self.interest_rate_matrix:
            self.interest_rate_matrix[inflation_state] = adjust_matrix(
                self.interest_rate_matrix[inflation_state], direction, strength
            )
        
        for growth_state in self.unemployment_matrix:
            self.unemployment_matrix[growth_state] = adjust_matrix(
                self.unemployment_matrix[growth_state], -direction, strength  # Inverse for unemployment
            )

        # Adjust sector performance
        for sector in Sector:
            self.state.sector_performance[sector] += impact

    def simulate_day(self):
        """Simulate one day in the market"""
        self.state.day += 1
        
        # Transition to new economic states
        self._transition_state()
        
        # Calculate market return for the day
        market_return = self._calculate_daily_market_return()
        
        # Update sector performance
        self._update_sector_performance(market_return)
        
        return market_return