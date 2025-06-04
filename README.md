# Cost-Effectiveness of Type 1 Diabetes Interventions in Resource-Limited Settings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete analysis code, data, and manuscript for a comprehensive cost-effectiveness evaluation of Type 1 diabetes interventions in resource-limited settings. The study uses patient-level discrete-time microsimulation to compare 19 individual interventions organized into 4 clusters, providing evidence-based guidance for health system decision makers.

## Repository Contents

### Core Files
- `analysis.R` - Complete R microsimulation code with full risk equations

### Documentation
- `CHEERS_checklist.md` - CHEERS 2022 compliance checklist
- `LICENSE` - MIT License for open source sharing

## Methodology

### Microsimulation Model
- **Patient-level discrete-time simulation** 
- **Validated risk equations** from DCCT/EDIC, Finnish Diabetes Register, Swedish Inpatient Register
- **Complete complication modeling** (CVD, ESRD, PDR, LEA, mortality)
- **Realistic correlation structure** between costs and effectiveness

### Expert Validation
- **Two-round Delphi process** with 15 international experts
- **Consensus achieved** on all intervention clusters and effectiveness estimates

### Economic Evaluation
- **Health system perspective** with 3% annual discount rate
- **Costs as % GDP per capita** following WHO guidelines
- **CHEERS 2022 compliant** reporting
- **Monte Carlo uncertainty analysis** with 95% confidence intervals

## Installation and Usage

### Requirements
```r
# R version 4.0 or higher
install.packages(c("dplyr", "ggplot2", "survival", "MASS", "readr"))
```

### Running the Analysis
```r
# Load and run the complete microsimulation
source("analysis.R")

# Results will be saved to:
# - individual_results.csv
# - cluster_results.csv
# - Generated figures (PNG format)
```

### Customization Options
The analysis can be customized by modifying parameters in `analysis.R`:
- **Population size:** Change `n_patients` (default: 10,000)
- **Policy horizon:** Modify `time_horizon` (default: 10 years)
- **Discount rate:** Adjust `discount_rate` (default: 3%)
- **Intervention effects:** Update HbA1c reduction values


## Citation

If you use this code or data in your research, please cite:

```
Basu S, Zafra J, Beran D. Cost-Effectiveness of Type 1 Diabetes Interventions 
in Resource-Limited Settings: A Microsimulation Analysis. Github. 2025.
```

## Authors

- **Sanjay Basu, MD, PhD** (Corresponding author)
  - University of California / San Francisco General Hospital
  - Waymark Care, San Francisco, CA
  - Email: sanjay.basu@ucsf.edu

- **Jessica Zafra, MSc**
  - Division of Tropical and Humanitarian Medicine, University of Geneva

- **David Beran, PhD**
  - Division of Tropical and Humanitarian Medicine, University of Geneva

## Funding

This work was supported by Breakthrough T1DM.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve the analysis or extend it to other settings. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Contact

For questions about the methodology or implementation, please contact:
- Sanjay Basu: sanjay.basu@ucsf.edu
- Repository issues: Use GitHub Issues for technical questions

## Acknowledgments

- International experts who participated in the Delphi validation process
- DCCT/EDIC, Finnish Diabetes Register, and Swedish Inpatient Register for risk equation data
- Breakthrough T1DM for funding support

