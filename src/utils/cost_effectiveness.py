"""
Cost-Effectiveness Analysis Utilities

This module provides utility functions for cost-effectiveness analysis
including efficiency frontier calculation, dominance analysis, and
cost-effectiveness metrics.

Author: Sanjay Basu, MD, PhD
Institution: University of California San Francisco / Waymark Care
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CostEffectivenessResult:
    """Cost-effectiveness analysis result"""
    intervention: str
    total_costs: float
    total_qalys: float
    incremental_costs: float
    incremental_qalys: float
    icer: float
    dominated: bool
    extendedly_dominated: bool

class CostEffectivenessAnalyzer:
    """Analyzer for cost-effectiveness results"""
    
    def __init__(self, simulation_results: Dict):
        """
        Initialize analyzer with simulation results
        
        Args:
            simulation_results: Dictionary of simulation results by intervention
        """
        self.results = simulation_results
        self.baseline_result = simulation_results.get('Baseline')
        
    def calculate_efficiency_frontier(self) -> List[Dict]:
        """
        Calculate efficiency frontier excluding dominated interventions
        
        Returns:
            List of interventions on the efficiency frontier
        """
        # Prepare data for analysis
        interventions_data = []
        
        for intervention_name, result in self.results.items():
            if intervention_name == 'Baseline':
                continue
                
            interventions_data.append({
                'intervention': intervention_name,
                'total_costs': result.total_costs,
                'total_qalys': result.total_qalys,
                'incremental_costs': result.incremental_costs,
                'incremental_qalys': result.incremental_qalys,
                'icer': result.icer_per_qaly
            })
        
        # Sort by total costs
        interventions_data.sort(key=lambda x: x['total_costs'])
        
        # Add baseline
        baseline_data = {
            'intervention': 'Baseline',
            'total_costs': self.baseline_result.total_costs,
            'total_qalys': self.baseline_result.total_qalys,
            'incremental_costs': 0,
            'incremental_qalys': 0,
            'icer': 0
        }
        
        all_interventions = [baseline_data] + interventions_data
        
        # Identify dominated interventions
        frontier_interventions = self._identify_frontier(all_interventions)
        
        return frontier_interventions
    
    def _identify_frontier(self, interventions: List[Dict]) -> List[Dict]:
        """Identify interventions on the efficiency frontier"""
        
        # Sort by total costs
        interventions.sort(key=lambda x: x['total_costs'])
        
        frontier = []
        
        for i, intervention in enumerate(interventions):
            dominated = False
            extendedly_dominated = False
            
            # Check for simple dominance
            for j, other in enumerate(interventions):
                if i != j:
                    if (other['total_costs'] <= intervention['total_costs'] and 
                        other['total_qalys'] >= intervention['total_qalys'] and
                        (other['total_costs'] < intervention['total_costs'] or 
                         other['total_qalys'] > intervention['total_qalys'])):
                        dominated = True
                        break
            
            if not dominated:
                # Check for extended dominance
                extendedly_dominated = self._check_extended_dominance(intervention, interventions)
            
            intervention['dominated'] = dominated
            intervention['extendedly_dominated'] = extendedly_dominated
            
            if not dominated and not extendedly_dominated:
                frontier.append(intervention)
        
        return frontier
    
    def _check_extended_dominance(self, intervention: Dict, all_interventions: List[Dict]) -> bool:
        """Check if intervention is extendedly dominated"""
        
        # Extended dominance occurs when a linear combination of two other interventions
        # provides better value than the intervention in question
        
        for i, int1 in enumerate(all_interventions):
            for j, int2 in enumerate(all_interventions):
                if i >= j or int1 == intervention or int2 == intervention:
                    continue
                
                # Check if intervention lies above the line connecting int1 and int2
                if (int1['total_costs'] < intervention['total_costs'] < int2['total_costs'] and
                    int1['total_qalys'] < int2['total_qalys']):
                    
                    # Calculate interpolated QALY at intervention's cost
                    cost_ratio = ((intervention['total_costs'] - int1['total_costs']) / 
                                 (int2['total_costs'] - int1['total_costs']))
                    
                    interpolated_qalys = (int1['total_qalys'] + 
                                        cost_ratio * (int2['total_qalys'] - int1['total_qalys']))
                    
                    if interpolated_qalys > intervention['total_qalys']:
                        return True
        
        return False
    
    def calculate_net_monetary_benefit(self, willingness_to_pay: float) -> Dict[str, float]:
        """
        Calculate net monetary benefit for each intervention
        
        Args:
            willingness_to_pay: Willingness to pay threshold per QALY
            
        Returns:
            Dictionary of net monetary benefits by intervention
        """
        nmb_results = {}
        
        for intervention_name, result in self.results.items():
            if intervention_name == 'Baseline':
                nmb_results[intervention_name] = 0  # Baseline reference
            else:
                nmb = (result.incremental_qalys * willingness_to_pay) - result.incremental_costs
                nmb_results[intervention_name] = nmb
        
        return nmb_results
    
    def rank_interventions_by_nmb(self, willingness_to_pay: float) -> List[Tuple[str, float]]:
        """
        Rank interventions by net monetary benefit
        
        Args:
            willingness_to_pay: Willingness to pay threshold per QALY
            
        Returns:
            List of (intervention_name, nmb) tuples sorted by NMB descending
        """
        nmb_results = self.calculate_net_monetary_benefit(willingness_to_pay)
        
        # Sort by NMB descending
        ranked = sorted(nmb_results.items(), key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def calculate_cost_effectiveness_acceptability(self, 
                                                 wtp_range: List[float]) -> pd.DataFrame:
        """
        Calculate cost-effectiveness acceptability across WTP range
        
        Args:
            wtp_range: List of willingness to pay thresholds
            
        Returns:
            DataFrame with acceptability probabilities
        """
        acceptability_data = []
        
        for wtp in wtp_range:
            nmb_results = self.calculate_net_monetary_benefit(wtp)
            
            # Find intervention with highest NMB
            best_intervention = max(nmb_results.items(), key=lambda x: x[1])
            
            for intervention_name, nmb in nmb_results.items():
                acceptability_data.append({
                    'WTP_Threshold': wtp,
                    'Intervention': intervention_name,
                    'Net_Monetary_Benefit': nmb,
                    'Optimal': intervention_name == best_intervention[0],
                    'Cost_Effective': nmb > 0
                })
        
        return pd.DataFrame(acceptability_data)
    
    def calculate_value_of_information(self) -> Dict[str, float]:
        """
        Calculate expected value of perfect information (EVPI)
        
        Returns:
            Dictionary with EVPI estimates
        """
        # Simplified EVPI calculation
        # In practice, this would require probabilistic sensitivity analysis
        
        evpi_results = {}
        
        # Calculate EVPI at different WTP thresholds
        wtp_thresholds = [4000, 12000, 50000, 100000]
        
        for wtp in wtp_thresholds:
            nmb_results = self.calculate_net_monetary_benefit(wtp)
            
            # EVPI is the difference between perfect information and current best decision
            max_nmb = max(nmb_results.values())
            current_best_nmb = max(nmb_results.values())  # Assuming perfect information for now
            
            evpi = max_nmb - current_best_nmb
            evpi_results[f'EVPI_at_WTP_{wtp}'] = evpi
        
        return evpi_results
    
    def generate_cost_effectiveness_summary(self) -> pd.DataFrame:
        """Generate comprehensive cost-effectiveness summary table"""
        
        summary_data = []
        
        # Add baseline
        summary_data.append({
            'Intervention': 'Baseline',
            'Total_Costs': self.baseline_result.total_costs,
            'Total_QALYs': self.baseline_result.total_qalys,
            'Incremental_Costs': 0,
            'Incremental_QALYs': 0,
            'ICER_per_QALY': 0,
            'Dominated': False,
            'Cost_Effective_1x_GDP': True,
            'Cost_Effective_3x_GDP': True
        })
        
        # Add interventions
        for intervention_name, result in self.results.items():
            if intervention_name == 'Baseline':
                continue
            
            # Check cost-effectiveness at different thresholds
            gdp_per_capita = 4000  # Approximate LMIC average
            ce_1x_gdp = result.icer_per_qaly <= gdp_per_capita
            ce_3x_gdp = result.icer_per_qaly <= 3 * gdp_per_capita
            
            summary_data.append({
                'Intervention': intervention_name,
                'Total_Costs': result.total_costs,
                'Total_QALYs': result.total_qalys,
                'Incremental_Costs': result.incremental_costs,
                'Incremental_QALYs': result.incremental_qalys,
                'ICER_per_QALY': result.icer_per_qaly,
                'Dominated': result.icer_per_qaly == float('inf'),
                'Cost_Effective_1x_GDP': ce_1x_gdp,
                'Cost_Effective_3x_GDP': ce_3x_gdp
            })
        
        return pd.DataFrame(summary_data)
    
    def calculate_budget_impact(self, target_population: int, 
                              coverage_rate: float, 
                              time_horizon: int = 5) -> Dict[str, Dict]:
        """
        Calculate budget impact for different interventions
        
        Args:
            target_population: Size of target population
            coverage_rate: Proportion of population covered (0-1)
            time_horizon: Time horizon for budget impact (years)
            
        Returns:
            Dictionary with budget impact results by intervention
        """
        budget_impact_results = {}
        
        covered_population = int(target_population * coverage_rate)
        
        for intervention_name, result in self.results.items():
            if intervention_name == 'Baseline':
                continue
            
            # Calculate annual costs
            annual_incremental_cost = result.incremental_costs / 10  # 10-year model
            
            # Total budget impact
            annual_budget_impact = annual_incremental_cost * covered_population
            total_budget_impact = annual_budget_impact * time_horizon
            
            # Health benefits
            total_qalys_gained = result.incremental_qalys * covered_population
            
            # Cost savings from complications prevented (simplified)
            complication_cost_savings = total_qalys_gained * 500  # Estimated savings per QALY
            
            net_budget_impact = total_budget_impact - complication_cost_savings
            
            budget_impact_results[intervention_name] = {
                'annual_budget_impact': annual_budget_impact,
                'total_budget_impact': total_budget_impact,
                'total_qalys_gained': total_qalys_gained,
                'complication_cost_savings': complication_cost_savings,
                'net_budget_impact': net_budget_impact,
                'cost_per_person_covered': annual_incremental_cost,
                'qalys_per_person_covered': result.incremental_qalys
            }
        
        return budget_impact_results
    
    def identify_optimal_portfolio(self, budget_constraint: float) -> List[str]:
        """
        Identify optimal intervention portfolio given budget constraint
        
        Args:
            budget_constraint: Available budget
            
        Returns:
            List of interventions in optimal portfolio
        """
        # Simple greedy algorithm based on cost-effectiveness ratios
        interventions = []
        
        for intervention_name, result in self.results.items():
            if intervention_name == 'Baseline':
                continue
            
            interventions.append({
                'name': intervention_name,
                'cost': result.incremental_costs,
                'qalys': result.incremental_qalys,
                'icer': result.icer_per_qaly
            })
        
        # Sort by ICER (most cost-effective first)
        interventions.sort(key=lambda x: x['icer'])
        
        # Select interventions within budget
        selected_interventions = []
        remaining_budget = budget_constraint
        
        for intervention in interventions:
            if intervention['cost'] <= remaining_budget and intervention['icer'] != float('inf'):
                selected_interventions.append(intervention['name'])
                remaining_budget -= intervention['cost']
        
        return selected_interventions

def calculate_confidence_intervals(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for a list of values
    
    Args:
        values: List of values
        confidence_level: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)
    
    return lower_bound, upper_bound

def format_currency(value: float, currency: str = "Int$") -> str:
    """Format currency values for display"""
    if abs(value) >= 1e6:
        return f"{currency}{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{currency}{value/1e3:.0f}K"
    else:
        return f"{currency}{value:.0f}"

def format_icer(icer: float, currency: str = "Int$") -> str:
    """Format ICER values for display"""
    if icer == float('inf'):
        return "Dominated"
    elif icer < 0:
        return "Cost-saving"
    else:
        return f"{format_currency(icer, currency)}/QALY"

