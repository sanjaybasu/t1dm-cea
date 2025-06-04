# Cost-Effectiveness of Type 1 Diabetes Interventions in Resource-Limited Settings

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete analysis code, data, and manuscript for a comprehensive cost-effectiveness evaluation of Type 1 diabetes interventions in resource-limited settings. The study uses patient-level discrete-time microsimulation to compare 19 individual interventions organized into 4 clusters, providing evidence-based guidance for health system decision makers.

## Key Findings

- **Most cost-effective cluster:** Community-Based Care Models ($32,151/DALY)
- **Highest health impact:** Multidisciplinary Care Teams (0.320 DALYs averted)
- **Best individual intervention:** Multidisciplinary Care Teams ($37,522/DALY)
- **Implementation guidance** provided for different resource levels

## Repository Contents

### Core Files
- `manuscript.md` - Complete manuscript ready for journal submission (3,500 words)
- `analysis.R` - Complete R microsimulation code with full risk equations
- `results.csv` - Individual intervention results with 95% confidence intervals
- `appendix.md` - Complete technical appendix with methodology details

### Figures
- `figure1_cost_effectiveness_plane.png` - Cost-effectiveness plane with confidence intervals
- `figure2_expert_validation.png` - Expert validation vs cost-effectiveness (clusters only)
- `figure3_sensitivity_analysis.png` - Tornado diagram sensitivity analysis

### Documentation
- `CHEERS_checklist.md` - CHEERS 2022 compliance checklist
- `LICENSE` - MIT License for open source sharing

## Methodology

### Microsimulation Model
- **Patient-level discrete-time simulation** (10,000 patients, 10-year horizon)
- **Validated risk equations** from DCCT/EDIC, Finnish Diabetes Register, Swedish Inpatient Register
- **Complete complication modeling** (CVD, ESRD, PDR, LEA, mortality)
- **Realistic correlation structure** between costs and effectiveness

### Expert Validation
- **Two-round Delphi process** with 15 international experts
- **100% response rate** across both rounds
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
- **Time horizon:** Modify `time_horizon` (default: 10 years)
- **Discount rate:** Adjust `discount_rate` (default: 3%)
- **Intervention effects:** Update HbA1c reduction values

## Key Results Summary

### Individual Interventions (Top 5)
1. **Multidisciplinary Care Teams:** 0.320 (0.194-0.445) DALYs, $37,522/DALY
2. **Care Coordination Systems:** 0.238 (0.145-0.332) DALYs, $37,742/DALY
3. **Integrated Electronic Health Records:** 0.207 (0.126-0.288) DALYs, $72,584/DALY
4. **Telemedicine Consultations:** 0.158 (0.096-0.220) DALYs, $31,606/DALY
5. **Digital Health Platforms:** 0.152 (0.092-0.211) DALYs, $52,743/DALY

### Intervention Clusters
1. **Community-Based Care:** $32,151/DALY (most cost-effective)
2. **Integrated Care Delivery:** $49,309/DALY (highest health impact)
3. **Self-Management Support:** $51,346/DALY
4. **Technology-Enhanced Care:** $98,287/DALY

## Implementation Guidance

### Very Low Resource Settings
- Prioritize **Community-Based Care Models**
- Focus on peer support groups and community health workers
- Minimal infrastructure requirements

### Moderate Resource Settings
- Implement **Integrated Care Delivery Models**
- Develop multidisciplinary care teams
- Invest in care coordination systems

### Higher Resource Settings
- Consider **Technology-Enhanced Care**
- Implement comprehensive monitoring systems
- Focus on quality improvement programs

## Citation

If you use this code or data in your research, please cite:

```
Basu S, Zafra J, Beran D. Cost-Effectiveness of Type 1 Diabetes Interventions 
in Resource-Limited Settings: A Microsimulation Analysis. [Journal]. [Year].
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

---

**Note:** This analysis provides evidence-based guidance for intervention prioritization but should be adapted to local contexts and resource availability. Implementation decisions should consider local epidemiology, healthcare infrastructure, and policy priorities.

