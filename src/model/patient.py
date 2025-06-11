"""
Patient Class for Type 1 Diabetes Microsimulation Model

This module defines the Patient class that represents individual patients
in the microsimulation model with all relevant characteristics and methods
for disease progression and intervention response.

Author: Sanjay Basu, MD, PhD
Institution: University of California San Francisco / Waymark Care
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class HealthState(Enum):
    """Health states for diabetes progression"""
    NO_COMPLICATIONS = "no_complications"
    MICROVASCULAR_ONLY = "microvascular_only"
    MACROVASCULAR_ONLY = "macrovascular_only"
    BOTH_COMPLICATIONS = "both_complications"
    DEAD = "dead"

class ComplicationType(Enum):
    """Types of diabetes complications"""
    RETINOPATHY = "retinopathy"
    NEPHROPATHY = "nephropathy"
    NEUROPATHY = "neuropathy"
    CHD = "coronary_heart_disease"
    STROKE = "stroke"
    PVD = "peripheral_vascular_disease"

@dataclass
class PatientCharacteristics:
    """Patient baseline characteristics"""
    age_at_onset: float
    baseline_hba1c: float
    sex: str  # 'male' or 'female'
    setting: str  # 'urban' or 'rural'
    bmi: float
    systolic_bp: float
    smoking: bool
    socioeconomic_status: str  # 'low', 'medium', 'high'

class Patient:
    """
    Individual patient in the microsimulation model
    
    Tracks patient characteristics, health state, complications,
    intervention history, and outcomes over time.
    """
    
    def __init__(self, patient_id: int, characteristics: PatientCharacteristics):
        """Initialize patient with baseline characteristics"""
        self.patient_id = patient_id
        self.characteristics = characteristics
        
        # Current state
        self.current_age = characteristics.age_at_onset
        self.current_hba1c = characteristics.baseline_hba1c
        self.diabetes_duration = 0.0
        self.health_state = HealthState.NO_COMPLICATIONS
        
        # Complications tracking
        self.complications = {comp: False for comp in ComplicationType}
        self.complication_onset_times = {comp: None for comp in ComplicationType}
        
        # Intervention tracking
        self.current_interventions = []
        self.intervention_history = []
        self.intervention_start_times = {}
        self.intervention_effects = {}
        
        # Outcomes tracking
        self.alive = True
        self.death_time = None
        self.death_cause = None
        self.total_costs = 0.0
        self.total_qalys = 0.0
        self.total_dalys = 0.0
        
        # Utilization tracking
        self.healthcare_utilization = {
            'routine_visits': 0,
            'specialist_visits': 0,
            'emergency_visits': 0,
            'hospitalizations': 0,
            'hospital_days': 0
        }
        
        # Time-varying tracking
        self.hba1c_history = [characteristics.baseline_hba1c]
        self.utility_history = []
        self.cost_history = []
        
    def get_current_utility(self) -> float:
        """Calculate current health utility based on health state and age"""
        from .parameters import UTILITY_PARAMS
        
        base_utility = UTILITY_PARAMS[f'diabetes_{self.health_state.value}']
        
        # Age-related decline
        age_adjustment = (self.current_age - self.characteristics.age_at_onset) * \
                        UTILITY_PARAMS['age_related_decline']
        
        # Complication-specific adjustments
        complication_adjustment = 0.0
        if self.complications[ComplicationType.RETINOPATHY]:
            complication_adjustment -= 0.05
        if self.complications[ComplicationType.NEPHROPATHY]:
            complication_adjustment -= 0.08
        if self.complications[ComplicationType.NEUROPATHY]:
            complication_adjustment -= 0.06
        if self.complications[ComplicationType.CHD]:
            complication_adjustment -= 0.10
        if self.complications[ComplicationType.STROKE]:
            complication_adjustment -= 0.15
        if self.complications[ComplicationType.PVD]:
            complication_adjustment -= 0.07
            
        utility = max(0.0, base_utility - age_adjustment + complication_adjustment)
        return utility
    
    def apply_intervention(self, intervention_name: str, current_time: float):
        """Apply intervention to patient"""
        from .parameters import get_intervention_parameters
        
        if intervention_name not in self.current_interventions:
            self.current_interventions.append(intervention_name)
            self.intervention_start_times[intervention_name] = current_time
            
            # Calculate intervention effect on HbA1c
            params = get_intervention_parameters(intervention_name)
            
            # Sample effect size from distribution
            effect_size = np.random.normal(
                params.hba1c_reduction_mean,
                params.hba1c_reduction_std
            )
            effect_size = np.clip(
                effect_size,
                params.hba1c_reduction_min,
                params.hba1c_reduction_max
            )
            
            # Adjust for baseline HbA1c (greater effect in poorly controlled patients)
            baseline_adjustment = min(1.5, max(0.5, (self.current_hba1c - 7.0) / 3.0))
            effect_size *= baseline_adjustment
            
            self.intervention_effects[intervention_name] = effect_size
            
            # Apply immediate effect
            self.current_hba1c = max(5.0, self.current_hba1c - effect_size)
            
            # Record intervention start
            self.intervention_history.append({
                'intervention': intervention_name,
                'start_time': current_time,
                'effect_size': effect_size
            })
    
    def apply_cluster(self, cluster_name: str, current_time: float):
        """Apply intervention cluster to patient"""
        from .parameters import get_cluster_parameters
        
        cluster_params = get_cluster_parameters(cluster_name)
        
        # Apply individual components
        for component in cluster_params.components:
            self.apply_intervention(component, current_time)
        
        # Apply synergistic effect
        synergy_effect = np.random.normal(
            cluster_params.hba1c_reduction_mean * (cluster_params.synergy_factor - 1.0),
            cluster_params.hba1c_reduction_std * 0.5
        )
        synergy_effect = max(0.0, synergy_effect)
        
        self.current_hba1c = max(5.0, self.current_hba1c - synergy_effect)
        
        # Record cluster application
        self.intervention_history.append({
            'intervention': f'CLUSTER_{cluster_name}',
            'start_time': current_time,
            'effect_size': synergy_effect,
            'components': cluster_params.components
        })
    
    def update_hba1c(self, time_step: float):
        """Update HbA1c considering intervention effects and natural progression"""
        # Natural progression (slight deterioration over time)
        natural_progression = 0.02 * time_step  # 0.02% per month
        
        # Intervention effect decay
        total_intervention_effect = 0.0
        for intervention, effect in self.intervention_effects.items():
            from .parameters import get_intervention_parameters
            params = get_intervention_parameters(intervention)
            
            # Calculate time since intervention start
            time_since_start = self.diabetes_duration - self.intervention_start_times[intervention]
            
            # Apply sustainability factor
            sustained_effect = effect * (params.sustainability_factor ** time_since_start)
            total_intervention_effect += sustained_effect
        
        # Update HbA1c
        self.current_hba1c += natural_progression
        self.current_hba1c = max(5.0, self.current_hba1c - total_intervention_effect)
        
        # Add some random variation
        random_variation = np.random.normal(0, 0.1)
        self.current_hba1c = max(5.0, self.current_hba1c + random_variation)
        
        self.hba1c_history.append(self.current_hba1c)
    
    def calculate_complication_risks(self) -> Dict[ComplicationType, float]:
        """Calculate annual complication risks based on current state"""
        from .parameters import COMPLICATION_PARAMS
        
        risks = {}
        
        # Microvascular complications
        for comp_type in [ComplicationType.RETINOPATHY, ComplicationType.NEPHROPATHY, ComplicationType.NEUROPATHY]:
            if not self.complications[comp_type]:
                base_risk_key = f'{comp_type.value}_base_risk'
                coeff_key = f'{comp_type.value}_hba1c_coefficient'
                
                base_risk = COMPLICATION_PARAMS[base_risk_key]
                hba1c_coeff = COMPLICATION_PARAMS[coeff_key]
                
                # Risk increases with HbA1c and diabetes duration
                hba1c_effect = hba1c_coeff * (self.current_hba1c - 7.0)
                duration_effect = 0.01 * self.diabetes_duration  # 1% increase per year
                
                annual_risk = base_risk * (1 + hba1c_effect + duration_effect)
                risks[comp_type] = max(0.0, min(0.5, annual_risk))
            else:
                risks[comp_type] = 0.0
        
        # Macrovascular complications
        for comp_type in [ComplicationType.CHD, ComplicationType.STROKE, ComplicationType.PVD]:
            if not self.complications[comp_type]:
                if comp_type == ComplicationType.CHD:
                    base_risk_key = 'chd_base_risk'
                    coeff_key = 'chd_hba1c_coefficient'
                elif comp_type == ComplicationType.STROKE:
                    base_risk_key = 'stroke_base_risk'
                    coeff_key = 'stroke_hba1c_coefficient'
                else:  # PVD
                    base_risk_key = 'pvd_base_risk'
                    coeff_key = 'pvd_hba1c_coefficient'
                
                base_risk = COMPLICATION_PARAMS[base_risk_key]
                hba1c_coeff = COMPLICATION_PARAMS[coeff_key]
                
                # Risk increases with HbA1c, age, and other factors
                hba1c_effect = hba1c_coeff * (self.current_hba1c - 7.0)
                age_effect = 0.02 * (self.current_age - 40) if self.current_age > 40 else 0
                
                # Additional risk factors
                sex_effect = 0.2 if self.characteristics.sex == 'male' else 0
                smoking_effect = 0.5 if self.characteristics.smoking else 0
                
                annual_risk = base_risk * (1 + hba1c_effect + age_effect + sex_effect + smoking_effect)
                risks[comp_type] = max(0.0, min(0.3, annual_risk))
            else:
                risks[comp_type] = 0.0
        
        return risks
    
    def update_complications(self, time_step: float):
        """Update complication status based on risks"""
        risks = self.calculate_complication_risks()
        
        for comp_type, annual_risk in risks.items():
            if not self.complications[comp_type]:
                # Convert annual risk to monthly probability
                monthly_risk = 1 - (1 - annual_risk) ** time_step
                
                if np.random.random() < monthly_risk:
                    self.complications[comp_type] = True
                    self.complication_onset_times[comp_type] = self.diabetes_duration
        
        # Update health state based on complications
        self.update_health_state()
    
    def update_health_state(self):
        """Update health state based on current complications"""
        has_microvascular = any([
            self.complications[ComplicationType.RETINOPATHY],
            self.complications[ComplicationType.NEPHROPATHY],
            self.complications[ComplicationType.NEUROPATHY]
        ])
        
        has_macrovascular = any([
            self.complications[ComplicationType.CHD],
            self.complications[ComplicationType.STROKE],
            self.complications[ComplicationType.PVD]
        ])
        
        if has_microvascular and has_macrovascular:
            self.health_state = HealthState.BOTH_COMPLICATIONS
        elif has_microvascular:
            self.health_state = HealthState.MICROVASCULAR_ONLY
        elif has_macrovascular:
            self.health_state = HealthState.MACROVASCULAR_ONLY
        else:
            self.health_state = HealthState.NO_COMPLICATIONS
    
    def calculate_mortality_risk(self) -> float:
        """Calculate annual mortality risk"""
        from .parameters import COMPLICATION_PARAMS
        
        # Base mortality risk for age and sex (simplified)
        if self.characteristics.sex == 'male':
            base_mortality = 0.001 * (self.current_age / 50) ** 3
        else:
            base_mortality = 0.0008 * (self.current_age / 50) ** 3
        
        # Diabetes-related mortality multiplier
        diabetes_multiplier = COMPLICATION_PARAMS['diabetes_mortality_multiplier']
        
        # Complication-related mortality increases
        complication_multiplier = 1.0
        if self.complications[ComplicationType.CHD]:
            complication_multiplier *= 2.0
        if self.complications[ComplicationType.STROKE]:
            complication_multiplier *= 1.8
        if self.complications[ComplicationType.NEPHROPATHY]:
            complication_multiplier *= 1.5
        
        # HbA1c effect on mortality
        hba1c_multiplier = 1 + 0.1 * (self.current_hba1c - 7.0)
        
        # Resource-limited setting multiplier
        setting_multiplier = COMPLICATION_PARAMS['background_mortality_multiplier']
        
        annual_mortality = base_mortality * diabetes_multiplier * complication_multiplier * \
                          hba1c_multiplier * setting_multiplier
        
        return max(0.0, min(0.5, annual_mortality))
    
    def update_survival(self, time_step: float, current_time: float):
        """Update survival status"""
        if self.alive:
            annual_mortality = self.calculate_mortality_risk()
            monthly_mortality = 1 - (1 - annual_mortality) ** time_step
            
            if np.random.random() < monthly_mortality:
                self.alive = False
                self.death_time = current_time
                self.health_state = HealthState.DEAD
                
                # Determine cause of death
                if any([self.complications[ComplicationType.CHD], 
                       self.complications[ComplicationType.STROKE]]):
                    self.death_cause = "cardiovascular"
                elif self.complications[ComplicationType.NEPHROPATHY]:
                    self.death_cause = "renal"
                else:
                    self.death_cause = "other_diabetes"
    
    def calculate_monthly_costs(self) -> float:
        """Calculate costs for current month"""
        from .parameters import UTILIZATION_PARAMS, get_intervention_parameters
        
        monthly_costs = 0.0
        
        if not self.alive:
            return monthly_costs
        
        # Routine care costs
        routine_visits_per_month = UTILIZATION_PARAMS['routine_visits_per_year'] / 12
        monthly_costs += routine_visits_per_month * UTILIZATION_PARAMS['primary_care_visit_cost']
        
        # Intervention costs
        for intervention in self.current_interventions:
            params = get_intervention_parameters(intervention)
            monthly_intervention_cost = params.annual_cost_mean / 12
            monthly_costs += monthly_intervention_cost
        
        # Complication-related costs
        if any(self.complications.values()):
            # Additional monitoring and treatment
            monthly_costs += 50  # Additional monthly costs for complications
            
            # Specialist visits
            if self.health_state in [HealthState.MICROVASCULAR_ONLY, HealthState.BOTH_COMPLICATIONS]:
                monthly_costs += 0.5 * UTILIZATION_PARAMS['specialist_visit_cost']  # Bi-monthly specialist
            
            if self.health_state in [HealthState.MACROVASCULAR_ONLY, HealthState.BOTH_COMPLICATIONS]:
                monthly_costs += 0.33 * UTILIZATION_PARAMS['specialist_visit_cost']  # Quarterly cardiologist
        
        # Emergency care (probabilistic)
        emergency_risk = 0.01  # 1% monthly risk of emergency visit
        if self.current_hba1c > 10.0:
            emergency_risk *= 2  # Higher risk with poor control
        
        if np.random.random() < emergency_risk:
            monthly_costs += UTILIZATION_PARAMS['emergency_visit_cost']
            self.healthcare_utilization['emergency_visits'] += 1
        
        # Hospitalization (probabilistic)
        hospitalization_risk = 0.005  # 0.5% monthly risk
        if self.health_state == HealthState.BOTH_COMPLICATIONS:
            hospitalization_risk *= 3
        
        if np.random.random() < hospitalization_risk:
            hospital_days = np.random.poisson(3) + 1  # 1-7 days typically
            hospitalization_cost = hospital_days * UTILIZATION_PARAMS['hospitalization_cost_per_day']
            monthly_costs += hospitalization_cost
            self.healthcare_utilization['hospitalizations'] += 1
            self.healthcare_utilization['hospital_days'] += hospital_days
        
        return monthly_costs
    
    def update_outcomes(self, time_step: float):
        """Update QALY and DALY outcomes"""
        if self.alive:
            # Calculate QALYs
            current_utility = self.get_current_utility()
            monthly_qalys = current_utility * time_step
            self.total_qalys += monthly_qalys
            self.utility_history.append(current_utility)
            
            # Calculate DALYs (simplified - using 1 - utility as disability weight)
            disability_weight = 1 - current_utility
            monthly_dalys = disability_weight * time_step
            self.total_dalys += monthly_dalys
        
        # Update costs
        monthly_costs = self.calculate_monthly_costs()
        self.total_costs += monthly_costs
        self.cost_history.append(monthly_costs)
    
    def simulate_month(self, current_time: float, time_step: float):
        """Simulate one month for this patient"""
        if not self.alive:
            return
        
        # Update age and diabetes duration
        self.current_age += time_step
        self.diabetes_duration += time_step
        
        # Update HbA1c
        self.update_hba1c(time_step)
        
        # Update complications
        self.update_complications(time_step)
        
        # Update survival
        self.update_survival(time_step, current_time)
        
        # Update outcomes
        self.update_outcomes(time_step)
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for this patient"""
        return {
            'patient_id': self.patient_id,
            'alive': self.alive,
            'death_time': self.death_time,
            'death_cause': self.death_cause,
            'final_age': self.current_age,
            'diabetes_duration': self.diabetes_duration,
            'final_hba1c': self.current_hba1c,
            'health_state': self.health_state.value,
            'complications': {comp.value: status for comp, status in self.complications.items()},
            'total_costs': self.total_costs,
            'total_qalys': self.total_qalys,
            'total_dalys': self.total_dalys,
            'interventions': self.current_interventions,
            'healthcare_utilization': self.healthcare_utilization
        }

