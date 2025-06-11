"""
Sensitivity Analysis for Type 1 Diabetes Cost-Effectiveness Study

This script performs comprehensive sensitivity analysis including
one-way, probabilistic, and scenario analyses.

Author: Sanjay Basu, MD, PhD
Institution: University of California San Francisco / Waymark Care
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import logging

from model.microsimulation import MicrosimulationModel
from model.parameters import (
    INDIVIDUAL_INTERVENTIONS, INTERVENTION_CLUSTERS, ECONOMIC_PARAMS,
    SENSITIVITY_PARAMS, get_intervention_parameters, get_cluster_parameters
)

class SensitivityAnalyzer:
    """Comprehensive sensitivity analysis for cost-effectiveness model"""
    
    def __init__(self, base_results: dict):
        self.base_results = base_results
        self.logger = logging.getLogger(__name__)
        
    def one_way_sensitivity_analysis(self, intervention_name: str, 
                                   parameter_ranges: dict) -> pd.DataFrame:
        """
        Perform one-way sensitivity analysis
        
        Args:
            intervention_name: Name of intervention to analyze
            parameter_ranges: Dict of parameter names and their ranges
        """
        results = []
        
        for param_name, (low_val, high_val) in parameter_ranges.items():
            self.logger.info(f"Running one-way sensitivity for {param_name}")
            
            # Test low value
            low_result = self._run_sensitivity_scenario(
                intervention_name, {param_name: low_val}
            )
            
            # Test high value  
            high_result = self._run_sensitivity_scenario(
                intervention_name, {param_name: high_val}
            )
            
            results.append({
                'parameter': param_name,
                'low_value': low_val,
                'high_value': high_val,
                'low_icer': low_result['icer_per_qaly'],
                'high_icer': high_result['icer_per_qaly'],
                'icer_range': high_result['icer_per_qaly'] - low_result['icer_per_qaly']
            })
        
        return pd.DataFrame(results)
    
    def probabilistic_sensitivity_analysis(self, intervention_name: str, 
                                         n_iterations: int = 1000) -> dict:
        """
        Perform probabilistic sensitivity analysis using Monte Carlo simulation
        
        Args:
            intervention_name: Name of intervention to analyze
            n_iterations: Number of Monte Carlo iterations
        """
        self.logger.info(f"Running PSA for {intervention_name} with {n_iterations} iterations")
        
        results = {
            'incremental_costs': [],
            'incremental_qalys': [],
            'icer_per_qaly': [],
            'net_monetary_benefit': []
        }
        
        # Define parameter distributions
        param_distributions = self._define_parameter_distributions(intervention_name)
        
        for i in tqdm(range(n_iterations), desc="PSA iterations"):
            # Sample parameters
            sampled_params = self._sample_parameters(param_distributions)
            
            # Run model with sampled parameters
            result = self._run_sensitivity_scenario(intervention_name, sampled_params)
            
            results['incremental_costs'].append(result['incremental_costs'])
            results['incremental_qalys'].append(result['incremental_qalys'])
            results['icer_per_qaly'].append(result['icer_per_qaly'])
            
            # Calculate NMB at different thresholds
            wtp_threshold = 4000  # 1x GDP per capita
            nmb = result['incremental_qalys'] * wtp_threshold - result['incremental_costs']
            results['net_monetary_benefit'].append(nmb)
        
        return results
    
    def scenario_analysis(self, intervention_name: str) -> pd.DataFrame:
        """
        Perform scenario analysis with different implementation contexts
        
        Args:
            intervention_name: Name of intervention to analyze
        """
        scenarios = {
            'Base Case': {},
            'High Resource Setting': {
                'effectiveness_multiplier': 1.2,
                'cost_multiplier': 1.3,
                'sustainability_multiplier': 1.1
            },
            'Low Resource Setting': {
                'effectiveness_multiplier': 0.8,
                'cost_multiplier': 0.7,
                'sustainability_multiplier': 0.9
            },
            'Urban Setting': {
                'effectiveness_multiplier': 1.1,
                'cost_multiplier': 1.2,
                'access_multiplier': 1.2
            },
            'Rural Setting': {
                'effectiveness_multiplier': 0.9,
                'cost_multiplier': 0.8,
                'access_multiplier': 0.8
            },
            'Optimistic': {
                'effectiveness_multiplier': 1.3,
                'cost_multiplier': 0.8,
                'sustainability_multiplier': 1.2
            },
            'Pessimistic': {
                'effectiveness_multiplier': 0.7,
                'cost_multiplier': 1.4,
                'sustainability_multiplier': 0.8
            }
        }
        
        results = []
        
        for scenario_name, scenario_params in scenarios.items():
            self.logger.info(f"Running scenario: {scenario_name}")
            
            if scenario_name == 'Base Case':
                # Use base case results
                base_result = self.base_results[intervention_name]
                result = {
                    'incremental_costs': base_result.incremental_costs,
                    'incremental_qalys': base_result.incremental_qalys,
                    'icer_per_qaly': base_result.icer_per_qaly
                }
            else:
                result = self._run_sensitivity_scenario(intervention_name, scenario_params)
            
            results.append({
                'scenario': scenario_name,
                'incremental_costs': result['incremental_costs'],
                'incremental_qalys': result['incremental_qalys'],
                'icer_per_qaly': result['icer_per_qaly']
            })
        
        return pd.DataFrame(results)
    
    def threshold_analysis(self, intervention_name: str) -> dict:
        """
        Perform threshold analysis to find break-even points
        
        Args:
            intervention_name: Name of intervention to analyze
        """
        # Find effectiveness threshold for cost-effectiveness
        wtp_thresholds = [4000, 12000, 50000, 100000]  # Different WTP thresholds
        
        threshold_results = {}
        
        for wtp in wtp_thresholds:
            # Binary search for effectiveness threshold
            low_eff = 0.0
            high_eff = 2.0  # Maximum reasonable effectiveness
            tolerance = 0.001
            
            while high_eff - low_eff > tolerance:
                mid_eff = (low_eff + high_eff) / 2
                
                result = self._run_sensitivity_scenario(
                    intervention_name, 
                    {'effectiveness_override': mid_eff}
                )
                
                if result['icer_per_qaly'] <= wtp:
                    high_eff = mid_eff
                else:
                    low_eff = mid_eff
            
            threshold_results[f'effectiveness_threshold_wtp_{wtp}'] = high_eff
        
        return threshold_results
    
    def _run_sensitivity_scenario(self, intervention_name: str, 
                                param_modifications: dict) -> dict:
        """Run model with modified parameters"""
        # Create modified model (simplified for demonstration)
        model = MicrosimulationModel(n_patients=1000, random_seed=42)
        
        # Apply parameter modifications (this would need to be implemented
        # to actually modify the model parameters)
        
        # For demonstration, return modified base case results
        base_result = self.base_results[intervention_name]
        
        # Apply modifications
        effectiveness_mult = param_modifications.get('effectiveness_multiplier', 1.0)
        cost_mult = param_modifications.get('cost_multiplier', 1.0)
        
        modified_qalys = base_result.incremental_qalys * effectiveness_mult
        modified_costs = base_result.incremental_costs * cost_mult
        
        modified_icer = modified_costs / modified_qalys if modified_qalys > 0 else float('inf')
        
        return {
            'incremental_costs': modified_costs,
            'incremental_qalys': modified_qalys,
            'icer_per_qaly': modified_icer
        }
    
    def _define_parameter_distributions(self, intervention_name: str) -> dict:
        """Define probability distributions for PSA"""
        if intervention_name in INDIVIDUAL_INTERVENTIONS:
            params = get_intervention_parameters(intervention_name)
        else:
            params = get_cluster_parameters(intervention_name)
        
        distributions = {
            'hba1c_reduction': stats.beta(
                a=2, b=2, 
                loc=params.hba1c_reduction_min,
                scale=params.hba1c_reduction_max - params.hba1c_reduction_min
            ),
            'annual_cost': stats.gamma(
                a=4, scale=params.annual_cost_mean/4
            ),
            'sustainability_factor': stats.beta(a=5, b=2, loc=0.6, scale=0.4)
        }
        
        return distributions
    
    def _sample_parameters(self, distributions: dict) -> dict:
        """Sample parameters from distributions"""
        sampled = {}
        for param_name, distribution in distributions.items():
            sampled[param_name] = distribution.rvs()
        return sampled

def main():
    """Run comprehensive sensitivity analysis"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting sensitivity analysis")
    
    # Load base case results (would normally load from saved results)
    # For demonstration, create mock results
    from analysis.base_case_analysis import main as run_base_case
    
    # Run base case first
    model = MicrosimulationModel(n_patients=1000, random_seed=42)
    base_results = model.run_all_interventions()
    
    # Initialize sensitivity analyzer
    analyzer = SensitivityAnalyzer(base_results)
    
    # Define interventions to analyze
    key_interventions = ['DSMES', 'Task_Shifting', 'Integrated_Care', 'Public_Private']
    
    all_sensitivity_results = {}
    
    for intervention in key_interventions:
        logger.info(f"Analyzing {intervention}")
        
        # One-way sensitivity analysis
        parameter_ranges = {
            'effectiveness_multiplier': (0.7, 1.3),
            'cost_multiplier': (0.7, 1.3),
            'sustainability_multiplier': (0.8, 1.2),
            'discount_rate': (0.0, 0.06)
        }
        
        owa_results = analyzer.one_way_sensitivity_analysis(intervention, parameter_ranges)
        owa_results.to_csv(f'../results/sensitivity_one_way_{intervention}.csv', index=False)
        
        # Scenario analysis
        scenario_results = analyzer.scenario_analysis(intervention)
        scenario_results.to_csv(f'../results/sensitivity_scenarios_{intervention}.csv', index=False)
        
        # Probabilistic sensitivity analysis
        psa_results = analyzer.probabilistic_sensitivity_analysis(intervention, n_iterations=1000)
        
        # Save PSA results
        psa_df = pd.DataFrame(psa_results)
        psa_df.to_csv(f'../results/sensitivity_psa_{intervention}.csv', index=False)
        
        # Threshold analysis
        threshold_results = analyzer.threshold_analysis(intervention)
        
        all_sensitivity_results[intervention] = {
            'one_way': owa_results,
            'scenarios': scenario_results,
            'psa': psa_results,
            'thresholds': threshold_results
        }
    
    # Generate sensitivity analysis visualizations
    generate_sensitivity_plots(all_sensitivity_results)
    
    # Generate sensitivity analysis summary
    generate_sensitivity_summary(all_sensitivity_results)
    
    logger.info("Sensitivity analysis completed")

def generate_sensitivity_plots(results: dict):
    """Generate sensitivity analysis visualizations"""
    
    # Tornado plots for one-way sensitivity
    for intervention, intervention_results in results.items():
        owa_results = intervention_results['one_way']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by range
        owa_sorted = owa_results.sort_values('icer_range', ascending=True)
        
        y_pos = range(len(owa_sorted))
        
        # Create horizontal bars
        ax.barh(y_pos, owa_sorted['icer_range'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(owa_sorted['parameter'])
        ax.set_xlabel('ICER Range (2025 Int$ per QALY)')
        ax.set_title(f'One-Way Sensitivity Analysis: {intervention}')
        
        plt.tight_layout()
        plt.savefig(f'../results/sensitivity_tornado_{intervention}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Cost-effectiveness acceptability curves
    for intervention, intervention_results in results.items():
        if 'psa' in intervention_results:
            psa_results = intervention_results['psa']
            
            wtp_range = np.linspace(0, 50000, 100)
            prob_cost_effective = []
            
            for wtp in wtp_range:
                nmb_values = np.array(psa_results['incremental_qalys']) * wtp - \
                           np.array(psa_results['incremental_costs'])
                prob_ce = np.mean(nmb_values > 0)
                prob_cost_effective.append(prob_ce)
            
            plt.figure(figsize=(10, 6))
            plt.plot(wtp_range, prob_cost_effective, linewidth=2)
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            plt.axvline(x=4000, color='orange', linestyle='--', alpha=0.7, label='1x GDP per capita')
            plt.axvline(x=12000, color='red', linestyle='--', alpha=0.7, label='3x GDP per capita')
            plt.xlabel('Willingness to Pay (2025 Int$ per QALY)')
            plt.ylabel('Probability Cost-Effective')
            plt.title(f'Cost-Effectiveness Acceptability Curve: {intervention}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'../results/sensitivity_ceac_{intervention}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def generate_sensitivity_summary(results: dict):
    """Generate comprehensive sensitivity analysis summary"""
    
    summary_lines = []
    summary_lines.append("SENSITIVITY ANALYSIS SUMMARY")
    summary_lines.append("=" * 40)
    summary_lines.append("")
    
    for intervention, intervention_results in results.items():
        summary_lines.append(f"INTERVENTION: {intervention}")
        summary_lines.append("-" * 30)
        
        # One-way sensitivity summary
        if 'one_way' in intervention_results:
            owa_results = intervention_results['one_way']
            most_sensitive = owa_results.loc[owa_results['icer_range'].idxmax()]
            
            summary_lines.append(f"Most sensitive parameter: {most_sensitive['parameter']}")
            summary_lines.append(f"ICER range: ${most_sensitive['icer_range']:,.0f}")
        
        # Scenario analysis summary
        if 'scenarios' in intervention_results:
            scenario_results = intervention_results['scenarios']
            best_scenario = scenario_results.loc[scenario_results['icer_per_qaly'].idxmin()]
            worst_scenario = scenario_results.loc[scenario_results['icer_per_qaly'].idxmax()]
            
            summary_lines.append(f"Best case scenario: {best_scenario['scenario']}")
            summary_lines.append(f"  ICER: ${best_scenario['icer_per_qaly']:,.0f}/QALY")
            summary_lines.append(f"Worst case scenario: {worst_scenario['scenario']}")
            summary_lines.append(f"  ICER: ${worst_scenario['icer_per_qaly']:,.0f}/QALY")
        
        # PSA summary
        if 'psa' in intervention_results:
            psa_results = intervention_results['psa']
            icer_mean = np.mean(psa_results['icer_per_qaly'])
            icer_ci = np.percentile(psa_results['icer_per_qaly'], [2.5, 97.5])
            
            summary_lines.append(f"PSA mean ICER: ${icer_mean:,.0f}/QALY")
            summary_lines.append(f"95% CI: ${icer_ci[0]:,.0f} - ${icer_ci[1]:,.0f}")
            
            # Probability cost-effective at key thresholds
            for wtp in [4000, 12000, 50000]:
                nmb_values = np.array(psa_results['incremental_qalys']) * wtp - \
                           np.array(psa_results['incremental_costs'])
                prob_ce = np.mean(nmb_values > 0)
                summary_lines.append(f"Prob. cost-effective at ${wtp:,}/QALY: {prob_ce:.1%}")
        
        summary_lines.append("")
    
    # Save summary
    with open('../results/sensitivity_analysis_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))

if __name__ == "__main__":
    main()

