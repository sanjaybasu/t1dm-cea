"""
Base Case Analysis for Type 1 Diabetes Cost-Effectiveness Study

This script runs the base case analysis for all interventions and clusters,
generating the main cost-effectiveness results for the study.

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
from datetime import datetime
import logging

from model.microsimulation import MicrosimulationModel
from model.parameters import INDIVIDUAL_INTERVENTIONS, INTERVENTION_CLUSTERS, ECONOMIC_PARAMS
from utils.cost_effectiveness import CostEffectivenessAnalyzer
from visualization.plots import create_cost_effectiveness_plane, create_tornado_plot

def main():
    """Run base case cost-effectiveness analysis"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('results/base_case_analysis.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting base case cost-effectiveness analysis")
    logger.info(f"Analysis parameters:")
    logger.info(f"  - Time horizon: {ECONOMIC_PARAMS['time_horizon']} years")
    logger.info(f"  - Discount rate: {ECONOMIC_PARAMS['discount_rate']*100}%")
    logger.info(f"  - Perspective: Health system (WHO CHOICE)")
    logger.info(f"  - Currency: 2025 International Dollars")
    
    # Initialize model
    n_patients = 1000  # Smaller sample for faster demo
    model = MicrosimulationModel(n_patients=n_patients, random_seed=42)
    
    logger.info(f"Simulating {n_patients} patients per intervention")
    
    # Run all simulations
    all_results = model.run_all_interventions()
    
    # Generate cost-effectiveness summary
    ce_summary = model.get_cost_effectiveness_summary()
    
    # Save summary results
    ce_summary.to_csv('results/base_case_cost_effectiveness_summary.csv', index=False)
    logger.info("Cost-effectiveness summary saved to base_case_cost_effectiveness_summary.csv")
    
    # Print key results
    print("\n" + "="*80)
    print("BASE CASE COST-EFFECTIVENESS RESULTS")
    print("="*80)
    print(f"{'Intervention':<30} {'Inc. Cost':<12} {'Inc. QALY':<12} {'ICER/QALY':<15}")
    print("-"*80)
    
    for _, row in ce_summary.iterrows():
        if row['Intervention'] != 'Baseline':
            icer_display = f"${row['ICER_per_QALY']:,.0f}" if row['ICER_per_QALY'] != float('inf') else "Dominated"
            print(f"{row['Intervention']:<30} ${row['Incremental_Costs']:<11,.0f} "
                  f"{row['Incremental_QALYs']:<11.3f} {icer_display:<15}")
    
    print("\n" + "="*80)
    print("INTERVENTION CLUSTER RESULTS")
    print("="*80)
    
    cluster_results = ce_summary[ce_summary['Intervention'].isin(INTERVENTION_CLUSTERS.keys())]
    for _, row in cluster_results.iterrows():
        icer_display = f"${row['ICER_per_QALY']:,.0f}" if row['ICER_per_QALY'] != float('inf') else "Dominated"
        print(f"{row['Intervention']:<30}")
        print(f"  Incremental Cost: ${row['Incremental_Costs']:,.0f}")
        print(f"  Incremental QALYs: {row['Incremental_QALYs']:.3f}")
        print(f"  ICER per QALY: {icer_display}")
        print()
    
    # Cost-effectiveness thresholds analysis
    print("COST-EFFECTIVENESS AT DIFFERENT THRESHOLDS")
    print("="*60)
    
    # WHO thresholds (1x and 3x GDP per capita)
    # Using global average GDP per capita for LMICs (~$4,000)
    gdp_per_capita = 4000
    thresholds = {
        "Highly cost-effective (1x GDP)": gdp_per_capita,
        "Cost-effective (3x GDP)": 3 * gdp_per_capita,
        "International threshold ($50k)": 50000,
        "High-income threshold ($100k)": 100000
    }
    
    for threshold_name, threshold_value in thresholds.items():
        print(f"\n{threshold_name}: ${threshold_value:,}")
        cost_effective_interventions = ce_summary[
            (ce_summary['ICER_per_QALY'] <= threshold_value) & 
            (ce_summary['ICER_per_QALY'] > 0)
        ]
        
        if len(cost_effective_interventions) > 0:
            for _, row in cost_effective_interventions.iterrows():
                if row['Intervention'] != 'Baseline':
                    print(f"  ✓ {row['Intervention']}: ${row['ICER_per_QALY']:,.0f}/QALY")
        else:
            print("  No interventions cost-effective at this threshold")
    
    # Export detailed patient-level results
    model.export_detailed_results('results/base_case_patient_level_results.csv')
    
    # Create cost-effectiveness analyzer for additional analyses
    analyzer = CostEffectivenessAnalyzer(all_results)
    
    # Generate efficiency frontier
    frontier_results = analyzer.calculate_efficiency_frontier()
    frontier_df = pd.DataFrame(frontier_results)
    frontier_df.to_csv('results/base_case_efficiency_frontier.csv', index=False)
    
    # Calculate net monetary benefits
    nmb_results = {}
    for threshold_value in [gdp_per_capita, 3*gdp_per_capita, 50000]:
        nmb = model.calculate_net_monetary_benefit(threshold_value)
        nmb_results[f"NMB_at_{threshold_value}"] = nmb
    
    nmb_df = pd.DataFrame(nmb_results)
    nmb_df.to_csv('results/base_case_net_monetary_benefits.csv')
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Cost-effectiveness plane
    fig_ce_plane = create_cost_effectiveness_plane(ce_summary, gdp_per_capita)
    fig_ce_plane.savefig('results/base_case_cost_effectiveness_plane.png', 
                        dpi=300, bbox_inches='tight')
    plt.close(fig_ce_plane)
    
    # Results by intervention type
    individual_results = ce_summary[ce_summary['Intervention'].isin(INDIVIDUAL_INTERVENTIONS.keys())]
    cluster_results = ce_summary[ce_summary['Intervention'].isin(INTERVENTION_CLUSTERS.keys())]
    
    # Bar chart of ICERs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Individual interventions
    individual_results_filtered = individual_results[individual_results['ICER_per_QALY'] != float('inf')]
    if len(individual_results_filtered) > 0:
        ax1.bar(range(len(individual_results_filtered)), 
                individual_results_filtered['ICER_per_QALY'])
        ax1.set_xticks(range(len(individual_results_filtered)))
        ax1.set_xticklabels(individual_results_filtered['Intervention'], rotation=45, ha='right')
        ax1.set_ylabel('ICER per QALY (2025 Int$)')
        ax1.set_title('Individual Interventions')
        ax1.axhline(y=gdp_per_capita, color='red', linestyle='--', label='1x GDP per capita')
        ax1.axhline(y=3*gdp_per_capita, color='orange', linestyle='--', label='3x GDP per capita')
        ax1.legend()
    
    # Intervention clusters
    cluster_results_filtered = cluster_results[cluster_results['ICER_per_QALY'] != float('inf')]
    if len(cluster_results_filtered) > 0:
        ax2.bar(range(len(cluster_results_filtered)), 
                cluster_results_filtered['ICER_per_QALY'])
        ax2.set_xticks(range(len(cluster_results_filtered)))
        ax2.set_xticklabels(cluster_results_filtered['Intervention'], rotation=45, ha='right')
        ax2.set_ylabel('ICER per QALY (2025 Int$)')
        ax2.set_title('Intervention Clusters')
        ax2.axhline(y=gdp_per_capita, color='red', linestyle='--', label='1x GDP per capita')
        ax2.axhline(y=3*gdp_per_capita, color='orange', linestyle='--', label='3x GDP per capita')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/base_case_icer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    baseline_result = all_results['Baseline']
    print(f"Baseline scenario (10-year outcomes per patient):")
    print(f"  Total costs: ${baseline_result.total_costs/baseline_result.n_patients:,.0f}")
    print(f"  Total QALYs: {baseline_result.total_qalys/baseline_result.n_patients:.2f}")
    print(f"  Total DALYs: {baseline_result.total_dalys/baseline_result.n_patients:.2f}")
    print(f"  Life years: {baseline_result.life_years/baseline_result.n_patients:.2f}")
    
    # Most cost-effective interventions
    cost_effective_at_1x_gdp = ce_summary[
        (ce_summary['ICER_per_QALY'] <= gdp_per_capita) & 
        (ce_summary['ICER_per_QALY'] > 0)
    ].sort_values('ICER_per_QALY')
    
    if len(cost_effective_at_1x_gdp) > 0:
        print(f"\nMost cost-effective interventions (≤${gdp_per_capita:,}/QALY):")
        for _, row in cost_effective_at_1x_gdp.iterrows():
            print(f"  1. {row['Intervention']}: ${row['ICER_per_QALY']:,.0f}/QALY")
    
    # Highest impact interventions (by QALYs gained)
    highest_impact = ce_summary[ce_summary['Intervention'] != 'Baseline'].sort_values(
        'Incremental_QALYs', ascending=False
    ).head(3)
    
    print(f"\nHighest impact interventions (by QALYs gained):")
    for _, row in highest_impact.iterrows():
        print(f"  {row['Intervention']}: {row['Incremental_QALYs']:.3f} QALYs gained")
    
    # Generate final summary report
    generate_summary_report(ce_summary, all_results, gdp_per_capita)
    
    logger.info("Base case analysis completed successfully")
    logger.info("Results saved to results/ directory")

def generate_summary_report(ce_summary: pd.DataFrame, all_results: dict, gdp_per_capita: float):
    """Generate a comprehensive summary report"""
    
    report_lines = []
    report_lines.append("COST-EFFECTIVENESS ANALYSIS SUMMARY REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Model: Type 1 Diabetes Microsimulation")
    report_lines.append(f"Perspective: Health System (WHO CHOICE)")
    report_lines.append(f"Time Horizon: {ECONOMIC_PARAMS['time_horizon']} years")
    report_lines.append(f"Discount Rate: {ECONOMIC_PARAMS['discount_rate']*100}%")
    report_lines.append(f"Currency: 2025 International Dollars")
    report_lines.append("")
    
    # Key findings
    report_lines.append("KEY FINDINGS")
    report_lines.append("-" * 20)
    
    # Most cost-effective
    cost_effective = ce_summary[
        (ce_summary['ICER_per_QALY'] <= gdp_per_capita) & 
        (ce_summary['ICER_per_QALY'] > 0)
    ].sort_values('ICER_per_QALY')
    
    if len(cost_effective) > 0:
        best_intervention = cost_effective.iloc[0]
        report_lines.append(f"Most cost-effective intervention: {best_intervention['Intervention']}")
        report_lines.append(f"  ICER: ${best_intervention['ICER_per_QALY']:,.0f} per QALY")
        report_lines.append(f"  Incremental QALYs: {best_intervention['Incremental_QALYs']:.3f}")
        report_lines.append(f"  Incremental costs: ${best_intervention['Incremental_Costs']:,.0f}")
    
    # Highest impact
    highest_impact = ce_summary[ce_summary['Intervention'] != 'Baseline'].sort_values(
        'Incremental_QALYs', ascending=False
    ).iloc[0]
    
    report_lines.append(f"\nHighest impact intervention: {highest_impact['Intervention']}")
    report_lines.append(f"  QALYs gained: {highest_impact['Incremental_QALYs']:.3f}")
    report_lines.append(f"  ICER: ${highest_impact['ICER_per_QALY']:,.0f} per QALY")
    
    # Cluster performance
    cluster_results = ce_summary[ce_summary['Intervention'].isin(INTERVENTION_CLUSTERS.keys())]
    if len(cluster_results) > 0:
        report_lines.append(f"\nIntervention Cluster Performance:")
        for _, row in cluster_results.iterrows():
            ce_status = "Highly cost-effective" if row['ICER_per_QALY'] <= gdp_per_capita else \
                       "Cost-effective" if row['ICER_per_QALY'] <= 3*gdp_per_capita else \
                       "Not cost-effective"
            report_lines.append(f"  {row['Intervention']}: {ce_status}")
            report_lines.append(f"    ICER: ${row['ICER_per_QALY']:,.0f}/QALY")
    
    # Policy recommendations
    report_lines.append(f"\nPOLICY RECOMMENDATIONS")
    report_lines.append("-" * 25)
    
    if len(cost_effective) > 0:
        report_lines.append("Recommended for immediate implementation:")
        for _, row in cost_effective.iterrows():
            report_lines.append(f"  • {row['Intervention']} (${row['ICER_per_QALY']:,.0f}/QALY)")
    
    # Interventions cost-effective at 3x GDP
    cost_effective_3x = ce_summary[
        (ce_summary['ICER_per_QALY'] <= 3*gdp_per_capita) & 
        (ce_summary['ICER_per_QALY'] > gdp_per_capita)
    ]
    
    if len(cost_effective_3x) > 0:
        report_lines.append("\nRecommended with adequate resources:")
        for _, row in cost_effective_3x.iterrows():
            report_lines.append(f"  • {row['Intervention']} (${row['ICER_per_QALY']:,.0f}/QALY)")
    
    # Save report
    with open('results/base_case_summary_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

if __name__ == "__main__":
    main()

