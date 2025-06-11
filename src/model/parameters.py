"""
Model Parameters for Type 1 Diabetes Cost-Effectiveness Analysis

This module contains all model parameters derived from systematic review evidence
and expert validation through the Delphi process.

Author: Sanjay Basu, MD, PhD
Institution: University of California San Francisco / Waymark Care
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class InterventionParameters:
    """Parameters for individual interventions"""
    name: str
    hba1c_reduction_mean: float  # Mean HbA1c reduction (percentage points)
    hba1c_reduction_std: float   # Standard deviation of HbA1c reduction
    hba1c_reduction_min: float   # Minimum HbA1c reduction
    hba1c_reduction_max: float   # Maximum HbA1c reduction
    annual_cost_mean: float      # Mean annual cost (2025 International $)
    annual_cost_std: float       # Standard deviation of annual cost
    implementation_cost: float   # One-time implementation cost
    sustainability_factor: float # Factor for effect sustainability over time
    
@dataclass
class ClusterParameters:
    """Parameters for intervention clusters"""
    name: str
    components: List[str]        # List of component intervention names
    hba1c_reduction_mean: float  # Mean HbA1c reduction (percentage points)
    hba1c_reduction_std: float   # Standard deviation of HbA1c reduction
    hba1c_reduction_min: float   # Minimum HbA1c reduction
    hba1c_reduction_max: float   # Maximum HbA1c reduction
    annual_cost_mean: float      # Mean annual cost (2025 International $)
    annual_cost_std: float       # Standard deviation of annual cost
    implementation_cost: float   # One-time implementation cost
    synergy_factor: float        # Factor for synergistic effects beyond individual components

# Individual Intervention Parameters (derived from systematic review)
INDIVIDUAL_INTERVENTIONS = {
    'DSMES': InterventionParameters(
        name='Diabetes Self-Management Education & Support',
        hba1c_reduction_mean=0.725,  # Mean of 0.45-1.0% range
        hba1c_reduction_std=0.183,   # SD to capture range
        hba1c_reduction_min=0.45,
        hba1c_reduction_max=1.0,
        annual_cost_mean=1200,       # Based on $39,563/QALY and effectiveness
        annual_cost_std=300,
        implementation_cost=2500,
        sustainability_factor=0.85   # 85% effect retention over time
    ),
    
    'CGM': InterventionParameters(
        name='Continuous Glucose Monitoring',
        hba1c_reduction_mean=0.585,  # Mean of 0.2-0.97% range
        hba1c_reduction_std=0.256,   # SD to capture range
        hba1c_reduction_min=0.2,
        hba1c_reduction_max=0.97,
        annual_cost_mean=2800,       # Based on device and sensor costs
        annual_cost_std=700,
        implementation_cost=1500,
        sustainability_factor=0.90   # High retention with technology
    ),
    
    'SMBG': InterventionParameters(
        name='Self-Monitoring of Blood Glucose',
        hba1c_reduction_mean=0.25,   # Baseline comparator, modest independent effect
        hba1c_reduction_std=0.10,
        hba1c_reduction_min=0.1,
        hba1c_reduction_max=0.4,
        annual_cost_mean=400,        # Test strips and meter costs
        annual_cost_std=100,
        implementation_cost=200,
        sustainability_factor=0.75   # Requires ongoing supply
    ),
    
    'Task_Shifting': InterventionParameters(
        name='Task-Shifting to Non-Physician Providers',
        hba1c_reduction_mean=0.985,  # Mean of 0.77-1.2% range
        hba1c_reduction_std=0.143,   # SD to capture range
        hba1c_reduction_min=0.77,
        hba1c_reduction_max=1.2,
        annual_cost_mean=800,        # Lower personnel costs
        annual_cost_std=200,
        implementation_cost=3000,    # Training and setup costs
        sustainability_factor=0.80   # Depends on ongoing supervision
    ),
    
    'mHealth': InterventionParameters(
        name='Mobile Health Technologies',
        hba1c_reduction_mean=0.36,   # Mean of 0.16-0.56% range
        hba1c_reduction_std=0.133,   # SD to capture range
        hba1c_reduction_min=0.16,
        hba1c_reduction_max=0.56,
        annual_cost_mean=150,        # Low marginal cost after development
        annual_cost_std=50,
        implementation_cost=5000,    # Platform development costs
        sustainability_factor=0.70   # Requires ongoing engagement
    )
}

# Intervention Cluster Parameters (derived from systematic review and expert validation)
INTERVENTION_CLUSTERS = {
    'Integrated_Care': ClusterParameters(
        name='Integrated Care Delivery Models',
        components=['DSMES', 'Task_Shifting'],
        hba1c_reduction_mean=1.585,  # Mean of 0.77-2.4% range
        hba1c_reduction_std=0.543,   # SD to capture range
        hba1c_reduction_min=0.77,
        hba1c_reduction_max=2.4,
        annual_cost_mean=1800,       # Combined costs with some efficiency
        annual_cost_std=450,
        implementation_cost=4500,
        synergy_factor=1.15          # 15% synergistic benefit
    ),
    
    'Public_Private': ClusterParameters(
        name='Public-Private Partnership Models',
        components=['DSMES', 'mHealth', 'Task_Shifting'],
        hba1c_reduction_mean=2.4,    # Based on India pilot study
        hba1c_reduction_std=0.4,     # Estimated uncertainty
        hba1c_reduction_min=1.8,
        hba1c_reduction_max=3.0,
        annual_cost_mean=2000,       # Leveraged resources reduce per-unit cost
        annual_cost_std=500,
        implementation_cost=8000,    # Complex coordination costs
        synergy_factor=1.25          # 25% synergistic benefit from coordination
    ),
    
    'Technology_Enhanced': ClusterParameters(
        name='Technology-Enhanced Self-Management',
        components=['CGM', 'mHealth', 'DSMES', 'SMBG'],
        hba1c_reduction_mean=1.1,    # Mean of 0.7-1.5% estimated range
        hba1c_reduction_std=0.267,   # SD to capture range
        hba1c_reduction_min=0.7,
        hba1c_reduction_max=1.5,
        annual_cost_mean=3500,       # High technology costs
        annual_cost_std=875,
        implementation_cost=6000,
        synergy_factor=1.10          # 10% synergistic benefit from integration
    )
}

# Population Parameters
POPULATION_PARAMS = {
    'age_at_onset_mean': 15.0,       # Mean age at T1D onset
    'age_at_onset_std': 8.0,         # Standard deviation
    'baseline_hba1c_mean': 9.2,      # Mean baseline HbA1c in resource-limited settings
    'baseline_hba1c_std': 1.8,       # Standard deviation
    'baseline_hba1c_min': 7.0,       # Minimum baseline HbA1c
    'baseline_hba1c_max': 14.0,      # Maximum baseline HbA1c
    'male_proportion': 0.52,         # Proportion male
    'urban_proportion': 0.45,        # Proportion in urban settings
}

# Economic Parameters
ECONOMIC_PARAMS = {
    'discount_rate': 0.03,           # 3% annual discount rate
    'time_horizon': 10,              # 10-year time horizon
    'cycle_length': 1/12,            # Monthly cycles (1/12 year)
    'currency_year': 2025,           # Cost year
    'willingness_to_pay_gdp_multiple': 1.0,  # 1x GDP per capita threshold
    'alternative_wtp_thresholds': [0.5, 1.0, 2.0, 3.0],  # Multiple thresholds
}

# Health State Utilities (from Global Burden of Disease and diabetes-specific studies)
UTILITY_PARAMS = {
    'diabetes_no_complications': 0.85,
    'diabetes_microvascular_only': 0.75,
    'diabetes_macrovascular_only': 0.70,
    'diabetes_both_complications': 0.60,
    'age_related_decline': 0.002,    # Annual utility decline per year of age
}

# Complication Risk Parameters (adapted for resource-limited settings)
COMPLICATION_PARAMS = {
    # Microvascular complications
    'retinopathy_base_risk': 0.02,   # Annual base risk
    'retinopathy_hba1c_coefficient': 0.15,  # Risk increase per 1% HbA1c
    'nephropathy_base_risk': 0.015,
    'nephropathy_hba1c_coefficient': 0.12,
    'neuropathy_base_risk': 0.025,
    'neuropathy_hba1c_coefficient': 0.18,
    
    # Macrovascular complications
    'chd_base_risk': 0.008,          # Coronary heart disease
    'chd_hba1c_coefficient': 0.10,
    'stroke_base_risk': 0.005,
    'stroke_hba1c_coefficient': 0.08,
    'pvd_base_risk': 0.006,          # Peripheral vascular disease
    'pvd_hba1c_coefficient': 0.09,
    
    # Mortality
    'background_mortality_multiplier': 1.5,  # Increased mortality in resource-limited settings
    'diabetes_mortality_multiplier': 2.0,    # Additional diabetes-related mortality
}

# Healthcare Utilization Parameters
UTILIZATION_PARAMS = {
    'routine_visits_per_year': 4,    # Quarterly visits
    'specialist_visit_cost': 50,     # Cost per specialist visit (2025 Int$)
    'primary_care_visit_cost': 20,   # Cost per primary care visit
    'emergency_visit_cost': 200,     # Cost per emergency visit
    'hospitalization_cost_per_day': 100,  # Cost per hospital day
    'lab_test_cost': 15,             # Cost per HbA1c test
}

# Sensitivity Analysis Parameters
SENSITIVITY_PARAMS = {
    'n_monte_carlo_iterations': 10000,
    'confidence_level': 0.95,
    'parameter_variation_range': 0.25,  # ±25% for one-way sensitivity
    'correlation_matrix': {
        # Correlation between intervention costs and effectiveness
        'cost_effectiveness_correlation': 0.3,
    }
}

def get_intervention_parameters(intervention_name: str) -> InterventionParameters:
    """Get parameters for a specific intervention"""
    if intervention_name in INDIVIDUAL_INTERVENTIONS:
        return INDIVIDUAL_INTERVENTIONS[intervention_name]
    else:
        raise ValueError(f"Unknown intervention: {intervention_name}")

def get_cluster_parameters(cluster_name: str) -> ClusterParameters:
    """Get parameters for a specific intervention cluster"""
    if cluster_name in INTERVENTION_CLUSTERS:
        return INTERVENTION_CLUSTERS[cluster_name]
    else:
        raise ValueError(f"Unknown cluster: {cluster_name}")

def get_all_interventions() -> List[str]:
    """Get list of all individual intervention names"""
    return list(INDIVIDUAL_INTERVENTIONS.keys())

def get_all_clusters() -> List[str]:
    """Get list of all intervention cluster names"""
    return list(INTERVENTION_CLUSTERS.keys())

