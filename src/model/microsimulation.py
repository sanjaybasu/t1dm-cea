"""
Microsimulation Model for Type 1 Diabetes Cost-Effectiveness Analysis

This module implements the main microsimulation model that simulates
individual patients over time to evaluate intervention cost-effectiveness.

Author: Sanjay Basu, MD, PhD
Institution: University of California San Francisco / Waymark Care
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from tqdm import tqdm

from .patient import Patient, PatientCharacteristics, HealthState
from .parameters import (
    POPULATION_PARAMS, ECONOMIC_PARAMS, INDIVIDUAL_INTERVENTIONS, 
    INTERVENTION_CLUSTERS, get_intervention_parameters, get_cluster_parameters
)

@dataclass
class SimulationResults:
    """Results from microsimulation"""
    intervention_name: str
    n_patients: int
    total_costs: float
    total_qalys: float
    total_dalys: float
    life_years: float
    incremental_costs: float
    incremental_qalys: float
    incremental_dalys: float
    icer_per_qaly: float
    icer_per_daly: float
    patient_summaries: List[Dict]
    
class MicrosimulationModel:
    """
    Patient-level discrete-time microsimulation model for Type 1 diabetes
    
    Simulates individual patients over specified time horizon to evaluate
    intervention cost-effectiveness in resource-limited settings.
    """
    
    def __init__(self, n_patients: int = 1000, random_seed: Optional[int] = None):
        """
        Initialize microsimulation model
        
        Args:
            n_patients: Number of patients to simulate
            random_seed: Random seed for reproducibility
        """
        self.n_patients = n_patients
        self.time_horizon = ECONOMIC_PARAMS['time_horizon']
        self.cycle_length = ECONOMIC_PARAMS['cycle_length']
        self.discount_rate = ECONOMIC_PARAMS['discount_rate']
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.results = {}
        self.baseline_results = None
        
    def generate_patient_population(self) -> List[Patient]:
        """Generate population of patients with baseline characteristics"""
        patients = []
        
        for i in range(self.n_patients):
            # Sample baseline characteristics
            characteristics = self._sample_patient_characteristics()
            patient = Patient(patient_id=i, characteristics=characteristics)
            patients.append(patient)
        
        return patients
    
    def _sample_patient_characteristics(self) -> PatientCharacteristics:
        """Sample baseline patient characteristics from population distributions"""
        # Age at onset
        age_at_onset = np.random.normal(
            POPULATION_PARAMS['age_at_onset_mean'],
            POPULATION_PARAMS['age_at_onset_std']
        )
        age_at_onset = max(5, min(65, age_at_onset))  # Constrain to reasonable range
        
        # Baseline HbA1c
        baseline_hba1c = np.random.normal(
            POPULATION_PARAMS['baseline_hba1c_mean'],
            POPULATION_PARAMS['baseline_hba1c_std']
        )
        baseline_hba1c = max(
            POPULATION_PARAMS['baseline_hba1c_min'],
            min(POPULATION_PARAMS['baseline_hba1c_max'], baseline_hba1c)
        )
        
        # Sex
        sex = 'male' if np.random.random() < POPULATION_PARAMS['male_proportion'] else 'female'
        
        # Setting
        setting = 'urban' if np.random.random() < POPULATION_PARAMS['urban_proportion'] else 'rural'
        
        # BMI (simplified)
        bmi = np.random.normal(22, 3)
        bmi = max(15, min(35, bmi))
        
        # Blood pressure (simplified)
        systolic_bp = np.random.normal(120, 15)
        systolic_bp = max(90, min(180, systolic_bp))
        
        # Smoking
        smoking = np.random.random() < 0.15  # 15% smoking rate
        
        # Socioeconomic status
        ses_rand = np.random.random()
        if ses_rand < 0.6:
            socioeconomic_status = 'low'
        elif ses_rand < 0.85:
            socioeconomic_status = 'medium'
        else:
            socioeconomic_status = 'high'
        
        return PatientCharacteristics(
            age_at_onset=age_at_onset,
            baseline_hba1c=baseline_hba1c,
            sex=sex,
            setting=setting,
            bmi=bmi,
            systolic_bp=systolic_bp,
            smoking=smoking,
            socioeconomic_status=socioeconomic_status
        )
    
    def simulate_baseline(self) -> SimulationResults:
        """Simulate baseline scenario (standard care without interventions)"""
        self.logger.info("Simulating baseline scenario...")
        
        patients = self.generate_patient_population()
        
        # Simulate each patient
        for patient in tqdm(patients, desc="Simulating baseline patients"):
            self._simulate_patient(patient, intervention=None)
        
        # Calculate results
        results = self._calculate_results(patients, "Baseline")
        self.baseline_results = results
        
        return results
    
    def simulate_intervention(self, intervention_name: str) -> SimulationResults:
        """
        Simulate intervention scenario
        
        Args:
            intervention_name: Name of intervention or cluster to simulate
        """
        self.logger.info(f"Simulating intervention: {intervention_name}")
        
        patients = self.generate_patient_population()
        
        # Simulate each patient with intervention
        for patient in tqdm(patients, desc=f"Simulating {intervention_name}"):
            self._simulate_patient(patient, intervention=intervention_name)
        
        # Calculate results
        results = self._calculate_results(patients, intervention_name)
        
        # Calculate incremental results if baseline exists
        if self.baseline_results is not None:
            results.incremental_costs = results.total_costs - self.baseline_results.total_costs
            results.incremental_qalys = results.total_qalys - self.baseline_results.total_qalys
            results.incremental_dalys = results.total_dalys - self.baseline_results.total_dalys
            
            # Calculate ICERs
            if results.incremental_qalys > 0:
                results.icer_per_qaly = results.incremental_costs / results.incremental_qalys
            else:
                results.icer_per_qaly = float('inf')
            
            if results.incremental_dalys < 0:  # DALYs averted (negative is good)
                results.icer_per_daly = results.incremental_costs / abs(results.incremental_dalys)
            else:
                results.icer_per_daly = float('inf')
        
        self.results[intervention_name] = results
        return results
    
    def _simulate_patient(self, patient: Patient, intervention: Optional[str]):
        """Simulate individual patient over time horizon"""
        n_cycles = int(self.time_horizon / self.cycle_length)
        
        # Apply intervention at start if specified
        if intervention is not None:
            if intervention in INDIVIDUAL_INTERVENTIONS:
                patient.apply_intervention(intervention, 0.0)
            elif intervention in INTERVENTION_CLUSTERS:
                patient.apply_cluster(intervention, 0.0)
            else:
                raise ValueError(f"Unknown intervention: {intervention}")
        
        # Simulate each cycle
        for cycle in range(n_cycles):
            current_time = cycle * self.cycle_length
            
            if patient.alive:
                patient.simulate_month(current_time, self.cycle_length)
            else:
                break
    
    def _calculate_results(self, patients: List[Patient], intervention_name: str) -> SimulationResults:
        """Calculate aggregate results from patient simulations"""
        # Get patient summaries
        patient_summaries = [patient.get_summary_statistics() for patient in patients]
        
        # Calculate totals
        total_costs = sum(patient.total_costs for patient in patients)
        total_qalys = sum(patient.total_qalys for patient in patients)
        total_dalys = sum(patient.total_dalys for patient in patients)
        
        # Calculate life years
        life_years = sum(
            patient.diabetes_duration if patient.alive 
            else (patient.death_time if patient.death_time else 0)
            for patient in patients
        )
        
        # Apply discounting
        total_costs = self._apply_discounting(total_costs)
        total_qalys = self._apply_discounting(total_qalys)
        total_dalys = self._apply_discounting(total_dalys)
        
        return SimulationResults(
            intervention_name=intervention_name,
            n_patients=len(patients),
            total_costs=total_costs,
            total_qalys=total_qalys,
            total_dalys=total_dalys,
            life_years=life_years,
            incremental_costs=0.0,  # Will be calculated later
            incremental_qalys=0.0,
            incremental_dalys=0.0,
            icer_per_qaly=0.0,
            icer_per_daly=0.0,
            patient_summaries=patient_summaries
        )
    
    def _apply_discounting(self, value: float) -> float:
        """Apply discounting to costs and outcomes"""
        # Simplified discounting - assumes even distribution over time horizon
        discount_factor = (1 - (1 + self.discount_rate) ** (-self.time_horizon)) / self.discount_rate
        return value * discount_factor / self.time_horizon
    
    def run_all_interventions(self) -> Dict[str, SimulationResults]:
        """Run simulation for all interventions and clusters"""
        all_results = {}
        
        # Run baseline first
        baseline = self.simulate_baseline()
        all_results['Baseline'] = baseline
        
        # Run individual interventions
        for intervention_name in INDIVIDUAL_INTERVENTIONS.keys():
            results = self.simulate_intervention(intervention_name)
            all_results[intervention_name] = results
        
        # Run intervention clusters
        for cluster_name in INTERVENTION_CLUSTERS.keys():
            results = self.simulate_intervention(cluster_name)
            all_results[cluster_name] = results
        
        return all_results
    
    def get_cost_effectiveness_summary(self) -> pd.DataFrame:
        """Get summary table of cost-effectiveness results"""
        if not self.results or self.baseline_results is None:
            raise ValueError("Must run simulations before generating summary")
        
        summary_data = []
        
        # Add baseline
        summary_data.append({
            'Intervention': 'Baseline',
            'Total_Costs': self.baseline_results.total_costs,
            'Total_QALYs': self.baseline_results.total_qalys,
            'Total_DALYs': self.baseline_results.total_dalys,
            'Incremental_Costs': 0,
            'Incremental_QALYs': 0,
            'Incremental_DALYs': 0,
            'ICER_per_QALY': 0,
            'ICER_per_DALY': 0
        })
        
        # Add interventions
        for intervention_name, results in self.results.items():
            summary_data.append({
                'Intervention': intervention_name,
                'Total_Costs': results.total_costs,
                'Total_QALYs': results.total_qalys,
                'Total_DALYs': results.total_dalys,
                'Incremental_Costs': results.incremental_costs,
                'Incremental_QALYs': results.incremental_qalys,
                'Incremental_DALYs': results.incremental_dalys,
                'ICER_per_QALY': results.icer_per_qaly,
                'ICER_per_DALY': results.icer_per_daly
            })
        
        return pd.DataFrame(summary_data)
    
    def export_detailed_results(self, filename: str):
        """Export detailed patient-level results to CSV"""
        all_patient_data = []
        
        # Baseline patients
        if self.baseline_results:
            for patient_summary in self.baseline_results.patient_summaries:
                patient_summary['intervention'] = 'Baseline'
                all_patient_data.append(patient_summary)
        
        # Intervention patients
        for intervention_name, results in self.results.items():
            for patient_summary in results.patient_summaries:
                patient_summary['intervention'] = intervention_name
                all_patient_data.append(patient_summary)
        
        df = pd.DataFrame(all_patient_data)
        df.to_csv(filename, index=False)
        self.logger.info(f"Detailed results exported to {filename}")
    
    def calculate_net_monetary_benefit(self, willingness_to_pay: float) -> Dict[str, float]:
        """Calculate net monetary benefit for each intervention"""
        nmb_results = {}
        
        for intervention_name, results in self.results.items():
            nmb = (results.incremental_qalys * willingness_to_pay) - results.incremental_costs
            nmb_results[intervention_name] = nmb
        
        return nmb_results
    
    def get_cost_effectiveness_acceptability(self, wtp_range: List[float]) -> pd.DataFrame:
        """Calculate cost-effectiveness acceptability across WTP range"""
        # This would require probabilistic sensitivity analysis
        # For now, return deterministic results
        acceptability_data = []
        
        for wtp in wtp_range:
            nmb_results = self.calculate_net_monetary_benefit(wtp)
            
            for intervention, nmb in nmb_results.items():
                acceptability_data.append({
                    'WTP_Threshold': wtp,
                    'Intervention': intervention,
                    'Net_Monetary_Benefit': nmb,
                    'Cost_Effective': nmb > 0
                })
        
        return pd.DataFrame(acceptability_data)

