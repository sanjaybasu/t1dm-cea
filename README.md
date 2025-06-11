# Cost-Effectiveness Analysis of Type 1 Diabetes Interventions in Resource-Limited Settings

## Overview

This repository contains the complete microsimulation model and analysis code for evaluating the cost-effectiveness of Type 1 diabetes interventions in resource-limited settings. The analysis follows WHO CHOICE methodology and CHEERS 2022 guidelines.

## Study Design

**Objective:** To evaluate the cost-effectiveness of individual Type 1 diabetes interventions and intervention clusters in resource-limited settings through microsimulation modeling with expert validation.

**Methods:** Patient-level discrete-time microsimulation model with 10-year time horizon, health system perspective, and probabilistic sensitivity analysis.

**Interventions Evaluated:**

### Individual Interventions:
1. **DSMES** - Diabetes Self-Management Education & Support
2. **CGM** - Continuous Glucose Monitoring  
3. **SMBG** - Self-Monitoring of Blood Glucose
4. **Task-Shifting** - Training non-physician healthcare providers
5. **mHealth** - Mobile Health Technologies

### Intervention Clusters:
1. **Integrated Care Delivery Models** - DSMES + Task-Shifting
2. **Public-Private Partnership Models** - DSMES + mHealth + Task-Shifting  
3. **Technology-Enhanced Self-Management** - CGM + mHealth + DSMES + SMBG

## Repository Structure

```
├── src/                          # Source code
│   ├── model/                    # Microsimulation model
│   ├── analysis/                 # Analysis scripts
│   ├── utils/                    # Utility functions
│   └── visualization/            # Plotting and visualization
├── data/                         # Input data and parameters
├── results/                      # Model outputs and results
├── docs/                         # Documentation
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Features

- **Patient-level microsimulation** with individual heterogeneity
- **WHO CHOICE cost methodology** with international dollar standardization
- **Probabilistic sensitivity analysis** with 10,000 Monte Carlo iterations
- **Expert-validated parameters** through two-round Delphi process
- **Comprehensive intervention clustering** based on systematic review evidence
- **Resource-limited setting focus** with appropriate adaptations

## Model Specifications

- **Population:** Type 1 diabetes patients aged 10-65 years in LMICs
- **Time horizon:** 10 years with monthly cycles
- **Perspective:** Health system (WHO CHOICE methodology)
- **Discount rate:** 3% annually for costs and outcomes
- **Outcomes:** DALYs averted, QALYs gained, cost per outcome
- **Currency:** 2025 International Dollars and % GDP per capita

## Installation and Usage

```bash
# Clone repository
git clone https://github.com/sanjaybasu/t1dm-cea.git
cd t1dm-cea

# Install dependencies
pip install -r requirements.txt

# Run base case analysis
python src/analysis/base_case_analysis.py

# Run sensitivity analysis
python src/analysis/sensitivity_analysis.py

# Generate results summary
python src/analysis/results_summary.py
```

## Authors

**Sanjay Basu, MD, PhD**  
University of California San Francisco / San Francisco General Hospital  
Waymark Care  

**Jessica Hanae Zafra-Tanaka, MSc, PhD**  
University of Geneva and Geneva University Hospitals  

**David Beran, PhD**  
University of Geneva and Geneva University Hospitals  

## Citation

If you use this code or methodology, please cite:

Basu S, Zafra-Tanaka JH, Beran D. Cost-Effectiveness of Type 1 Diabetes Interventions in Resource-Limited Settings: A Systematic Review and Economic Evaluation with Delphi Expert Validation. *Lancet Diabetes Endocrinol*. 2025.

## License

MIT License - see LICENSE file for details.

## Funding

This work was supported by Breakthrough T1D. The funder had no role in study design, data collection, analysis, interpretation, or manuscript preparation.

## Contact

For questions about the model or analysis:
- Email: sanjay.basu@ucsf.edu
- GitHub: https://github.com/sanjaybasu/t1dm-cea

