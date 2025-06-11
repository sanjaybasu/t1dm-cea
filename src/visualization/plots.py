"""
Visualization Functions for Cost-Effectiveness Analysis

This module provides functions for creating publication-quality plots
and visualizations for cost-effectiveness analysis results.

Author: Sanjay Basu, MD, PhD
Institution: University of California San Francisco / Waymark Care
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib.patches as patches

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_cost_effectiveness_plane(ce_summary: pd.DataFrame, 
                                  wtp_threshold: float,
                                  title: str = "Cost-Effectiveness Plane") -> plt.Figure:
    """
    Create cost-effectiveness plane plot
    
    Args:
        ce_summary: DataFrame with cost-effectiveness results
        wtp_threshold: Willingness to pay threshold
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out baseline and infinite ICERs
    plot_data = ce_summary[
        (ce_summary['Intervention'] != 'Baseline') & 
        (ce_summary['ICER_per_QALY'] != float('inf'))
    ].copy()
    
    # Create scatter plot
    scatter = ax.scatter(
        plot_data['Incremental_QALYs'], 
        plot_data['Incremental_Costs'],
        s=100, 
        alpha=0.7,
        c=range(len(plot_data)),
        cmap='viridis'
    )
    
    # Add intervention labels
    for idx, row in plot_data.iterrows():
        ax.annotate(
            row['Intervention'], 
            (row['Incremental_QALYs'], row['Incremental_Costs']),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=9,
            ha='left'
        )
    
    # Add WTP threshold line
    max_qaly = plot_data['Incremental_QALYs'].max() * 1.1
    x_threshold = np.linspace(0, max_qaly, 100)
    y_threshold = x_threshold * wtp_threshold
    
    ax.plot(x_threshold, y_threshold, 'r--', linewidth=2, 
            label=f'WTP Threshold: ${wtp_threshold:,.0f}/QALY')
    
    # Add quadrant labels
    ax.text(0.02, 0.98, 'More Effective\nMore Costly', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax.text(0.02, 0.02, 'Less Effective\nMore Costly', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    ax.text(0.98, 0.02, 'Less Effective\nLess Costly', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    ax.text(0.98, 0.98, 'More Effective\nLess Costly', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Incremental QALYs', fontsize=12)
    ax.set_ylabel('Incremental Costs (2025 Int$)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add origin lines
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    return fig

def create_tornado_plot(sensitivity_results: pd.DataFrame, 
                       intervention_name: str,
                       base_case_icer: float) -> plt.Figure:
    """
    Create tornado plot for one-way sensitivity analysis
    
    Args:
        sensitivity_results: DataFrame with sensitivity analysis results
        intervention_name: Name of intervention
        base_case_icer: Base case ICER value
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by range (largest impact first)
    sorted_results = sensitivity_results.sort_values('icer_range', ascending=True)
    
    y_pos = np.arange(len(sorted_results))
    
    # Calculate deviations from base case
    low_deviation = sorted_results['low_icer'] - base_case_icer
    high_deviation = sorted_results['high_icer'] - base_case_icer
    
    # Create horizontal bars
    bars_low = ax.barh(y_pos, low_deviation, height=0.6, 
                      color='lightcoral', alpha=0.7, label='Low Value')
    bars_high = ax.barh(y_pos, high_deviation, height=0.6, 
                       color='lightblue', alpha=0.7, label='High Value')
    
    # Add parameter labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_results['parameter'])
    
    # Add base case line
    ax.axvline(x=0, color='black', linewidth=2, label='Base Case')
    
    # Formatting
    ax.set_xlabel('Change in ICER from Base Case (2025 Int$ per QALY)', fontsize=12)
    ax.set_title(f'One-Way Sensitivity Analysis: {intervention_name}', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (low, high) in enumerate(zip(low_deviation, high_deviation)):
        ax.text(low - abs(low)*0.1, i, f'${sorted_results.iloc[i]["low_icer"]:,.0f}', 
                ha='right', va='center', fontsize=9)
        ax.text(high + abs(high)*0.1, i, f'${sorted_results.iloc[i]["high_icer"]:,.0f}', 
                ha='left', va='center', fontsize=9)
    
    return fig

def create_acceptability_curve(psa_results: Dict, 
                             intervention_name: str,
                             wtp_range: Optional[List[float]] = None) -> plt.Figure:
    """
    Create cost-effectiveness acceptability curve
    
    Args:
        psa_results: Dictionary with PSA results
        intervention_name: Name of intervention
        wtp_range: Range of WTP thresholds to plot
        
    Returns:
        matplotlib Figure object
    """
    if wtp_range is None:
        wtp_range = np.linspace(0, 100000, 200)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate probability of cost-effectiveness at each WTP threshold
    prob_cost_effective = []
    
    incremental_costs = np.array(psa_results['incremental_costs'])
    incremental_qalys = np.array(psa_results['incremental_qalys'])
    
    for wtp in wtp_range:
        nmb = incremental_qalys * wtp - incremental_costs
        prob_ce = np.mean(nmb > 0)
        prob_cost_effective.append(prob_ce)
    
    # Plot curve
    ax.plot(wtp_range, prob_cost_effective, linewidth=3, color='blue')
    
    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
               label='50% Probability')
    ax.axvline(x=4000, color='orange', linestyle='--', alpha=0.7, 
               label='1x GDP per capita')
    ax.axvline(x=12000, color='red', linestyle='--', alpha=0.7, 
               label='3x GDP per capita')
    
    # Formatting
    ax.set_xlabel('Willingness to Pay (2025 Int$ per QALY)', fontsize=12)
    ax.set_ylabel('Probability Cost-Effective', fontsize=12)
    ax.set_title(f'Cost-Effectiveness Acceptability Curve: {intervention_name}', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.ticklabel_format(style='plain', axis='x')
    
    return fig

def create_efficiency_frontier(frontier_results: List[Dict]) -> plt.Figure:
    """
    Create efficiency frontier plot
    
    Args:
        frontier_results: List of interventions on efficiency frontier
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    costs = [result['total_costs'] for result in frontier_results]
    qalys = [result['total_qalys'] for result in frontier_results]
    names = [result['intervention'] for result in frontier_results]
    
    # Plot frontier
    ax.plot(qalys, costs, 'o-', linewidth=2, markersize=8, color='blue')
    
    # Add intervention labels
    for i, name in enumerate(names):
        ax.annotate(name, (qalys[i], costs[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, ha='left')
    
    # Formatting
    ax.set_xlabel('Total QALYs', fontsize=12)
    ax.set_ylabel('Total Costs (2025 Int$)', fontsize=12)
    ax.set_title('Efficiency Frontier', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig

def create_budget_impact_plot(budget_impact_results: Dict[str, Dict],
                            time_horizon: int = 5) -> plt.Figure:
    """
    Create budget impact visualization
    
    Args:
        budget_impact_results: Dictionary with budget impact results
        time_horizon: Time horizon for analysis
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    interventions = list(budget_impact_results.keys())
    
    # Annual budget impact
    annual_impacts = [budget_impact_results[intervention]['annual_budget_impact'] 
                     for intervention in interventions]
    
    bars1 = ax1.bar(range(len(interventions)), annual_impacts, color='skyblue', alpha=0.7)
    ax1.set_xticks(range(len(interventions)))
    ax1.set_xticklabels(interventions, rotation=45, ha='right')
    ax1.set_ylabel('Annual Budget Impact (2025 Int$)', fontsize=12)
    ax1.set_title('Annual Budget Impact by Intervention', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, annual_impacts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${value/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
    
    # Cost per QALY gained vs QALYs gained
    qalys_gained = [budget_impact_results[intervention]['total_qalys_gained'] 
                   for intervention in interventions]
    cost_per_qaly = [budget_impact_results[intervention]['total_budget_impact'] / 
                    budget_impact_results[intervention]['total_qalys_gained']
                    for intervention in interventions]
    
    scatter = ax2.scatter(qalys_gained, cost_per_qaly, s=100, alpha=0.7, c=range(len(interventions)))
    
    for i, intervention in enumerate(interventions):
        ax2.annotate(intervention, (qalys_gained[i], cost_per_qaly[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left')
    
    ax2.set_xlabel('Total QALYs Gained', fontsize=12)
    ax2.set_ylabel('Cost per QALY Gained (2025 Int$)', fontsize=12)
    ax2.set_title('Cost per QALY vs Total QALYs Gained', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_intervention_comparison_heatmap(ce_summary: pd.DataFrame) -> plt.Figure:
    """
    Create heatmap comparing interventions across multiple metrics
    
    Args:
        ce_summary: DataFrame with cost-effectiveness results
        
    Returns:
        matplotlib Figure object
    """
    # Prepare data for heatmap
    interventions = ce_summary[ce_summary['Intervention'] != 'Baseline']['Intervention'].tolist()
    
    # Normalize metrics for comparison (0-1 scale)
    metrics_data = []
    
    for intervention in interventions:
        row_data = ce_summary[ce_summary['Intervention'] == intervention].iloc[0]
        
        # Calculate normalized scores (higher is better)
        qaly_score = row_data['Incremental_QALYs'] / ce_summary['Incremental_QALYs'].max()
        
        # Cost score (lower cost is better, so invert)
        cost_score = 1 - (row_data['Incremental_Costs'] / ce_summary['Incremental_Costs'].max())
        
        # ICER score (lower ICER is better, so invert)
        max_finite_icer = ce_summary[ce_summary['ICER_per_QALY'] != float('inf')]['ICER_per_QALY'].max()
        if row_data['ICER_per_QALY'] == float('inf'):
            icer_score = 0
        else:
            icer_score = 1 - (row_data['ICER_per_QALY'] / max_finite_icer)
        
        metrics_data.append([qaly_score, cost_score, icer_score])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    
    heatmap_data = np.array(metrics_data)
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Health Benefit\n(QALYs)', 'Cost Efficiency\n(Lower Cost)', 'Cost-Effectiveness\n(Lower ICER)'])
    ax.set_yticks(range(len(interventions)))
    ax.set_yticklabels(interventions)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Score (0=Worst, 1=Best)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(interventions)):
        for j in range(3):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Intervention Comparison Across Key Metrics', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_summary_dashboard(ce_summary: pd.DataFrame, 
                           budget_impact_results: Dict,
                           psa_results: Dict) -> plt.Figure:
    """
    Create comprehensive dashboard with multiple visualizations
    
    Args:
        ce_summary: DataFrame with cost-effectiveness results
        budget_impact_results: Dictionary with budget impact results
        psa_results: Dictionary with PSA results
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Cost-effectiveness plane
    ax1 = fig.add_subplot(gs[0, 0])
    plot_data = ce_summary[
        (ce_summary['Intervention'] != 'Baseline') & 
        (ce_summary['ICER_per_QALY'] != float('inf'))
    ]
    ax1.scatter(plot_data['Incremental_QALYs'], plot_data['Incremental_Costs'])
    ax1.set_xlabel('Incremental QALYs')
    ax1.set_ylabel('Incremental Costs')
    ax1.set_title('Cost-Effectiveness Plane')
    ax1.grid(True, alpha=0.3)
    
    # 2. ICER comparison
    ax2 = fig.add_subplot(gs[0, 1])
    icer_data = ce_summary[
        (ce_summary['Intervention'] != 'Baseline') & 
        (ce_summary['ICER_per_QALY'] != float('inf'))
    ]
    bars = ax2.bar(range(len(icer_data)), icer_data['ICER_per_QALY'])
    ax2.set_xticks(range(len(icer_data)))
    ax2.set_xticklabels(icer_data['Intervention'], rotation=45, ha='right')
    ax2.set_ylabel('ICER per QALY')
    ax2.set_title('ICER Comparison')
    ax2.axhline(y=4000, color='red', linestyle='--', alpha=0.7)
    
    # 3. Health benefits
    ax3 = fig.add_subplot(gs[0, 2])
    qaly_data = ce_summary[ce_summary['Intervention'] != 'Baseline']
    ax3.bar(range(len(qaly_data)), qaly_data['Incremental_QALYs'])
    ax3.set_xticks(range(len(qaly_data)))
    ax3.set_xticklabels(qaly_data['Intervention'], rotation=45, ha='right')
    ax3.set_ylabel('Incremental QALYs')
    ax3.set_title('Health Benefits')
    
    # 4. Budget impact
    ax4 = fig.add_subplot(gs[1, :])
    if budget_impact_results:
        interventions = list(budget_impact_results.keys())
        annual_impacts = [budget_impact_results[intervention]['annual_budget_impact'] 
                         for intervention in interventions]
        ax4.bar(interventions, annual_impacts)
        ax4.set_ylabel('Annual Budget Impact')
        ax4.set_title('Budget Impact by Intervention')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # 5. Uncertainty analysis
    ax5 = fig.add_subplot(gs[2, :])
    if psa_results:
        # Plot distribution of ICERs
        icers = psa_results.get('icer_per_qaly', [])
        if icers:
            ax5.hist(icers, bins=50, alpha=0.7, edgecolor='black')
            ax5.axvline(x=np.mean(icers), color='red', linestyle='--', 
                       label=f'Mean: ${np.mean(icers):,.0f}')
            ax5.set_xlabel('ICER per QALY')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Distribution of ICERs (Probabilistic Sensitivity Analysis)')
            ax5.legend()
    
    plt.suptitle('Cost-Effectiveness Analysis Dashboard', fontsize=16, fontweight='bold')
    
    return fig

