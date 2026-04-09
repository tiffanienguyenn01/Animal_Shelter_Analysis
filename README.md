# Long Beach Animal Shelter Operations Analysis and Adoption Probability Prediction

---

## Executive Summary

This project explores animal shelter operations in Long Beach and surrounding areas using SQL, Python, and Tableau to assess adoption rates and uncover key factors driving animal outcomes. The study reveals significant variation in adoption rates and length of stay across species and intake conditions, identifying operational bottlenecks and areas for improvement.

A logistic regression model was built to predict adoption likelihood based on animal species, sex, and intake condition. Results reveal stark disparities: neutered or spayed dogs classified as under age or weight show predicted adoption rates above 99%, while cats with unknown sex or severe medical/behavioral issues fall below 1%. Intake condition and animal type emerged as the strongest predictors of adoption outcomes.

---

## Business Problem

Animal shelters face limited capacity and resource constraints, yet adoption likelihood varies significantly across animal types and intake conditions. Without predictive insights, shelters cannot efficiently prioritize high-risk animals or optimize adoption strategies. This project addresses that gap by building a data-driven system that surfaces adoption risk early and enables targeted operational interventions.

---

## Methodology

Built an automated data engineering workflow using SQL to extract and transform over 50,000 shelter records into structured analytical datasets, engineering key metrics such as Adoption Rate and Live Release Rate, and performing grouped aggregations to evaluate operational performance by animal type, sex, and intake condition.

Created a logistic regression model in Python to estimate adoption probability based on animal-level features, evaluated outcome disparities across demographic groups, and identified high-risk populations with significantly lower anticipated adoption likelihood.

Visualized key insights in Tableau through interactive dashboards tracking intake trends, outcome distributions, length of stay, and forecasted adoption risk, enabling effective stakeholder communication and data-driven decision-making.

---

## Skills

- **SQL:** Data cleaning, aggregation, CASE statements, window functions, KPI development
- **Python:** pandas, NumPy, scikit-learn, statistical testing, data preprocessing, feature engineering, logistic regression modeling
- **Machine Learning:** Adoption probability prediction using logistic regression
- **Tableau:** Interactive dashboards, calculated fields, trend visualization

---

## Analysis Architecture

```
Raw Shelter Data → SQL Cleaning & KPI Engineering → Python Modeling
               → Logistic Regression Scores → Tableau Dashboard (Interactive)
```

---

## Data Sources

| Source | Data | Volume |
|---|---|---|
| Long Beach Animal Care Services | Shelter intake, outcome, and length-of-stay records | 50,000+ records |
| Shelter Operations Data | Animal type, sex, intake condition, age, weight | Weekly updated |

---

## Tools & Technologies

| Layer | Tool | Purpose |
|---|---|---|
| Data Engineering | SQL | Cleaning, aggregation, CASE statements, window functions |
| Predictive Modeling | Python — scikit-learn | Logistic regression, feature engineering, statistical testing |
| Data Manipulation | Python — pandas, NumPy | Preprocessing, group analysis, model input preparation |
| Dashboard | Tableau | Interactive dashboards, calculated fields, trend visualization |
| Machine Learning | Logistic Regression | Adoption probability prediction by species, sex, and intake condition |

---
## Dashboard
[View Dashboard](https://public.tableau.com/app/profile/thu.nguyen6411/viz/LongBeachAnimalShelterAnalysisDashboard/Overview)

---
## Results and Business Recommendations

**Results:** Adoption outcomes vary dramatically across animal groups. Neutered or spayed dogs classified as under age or weight show predicted adoption probabilities near 99%, while cats with unknown sex or severe medical/behavioral conditions fall below 1% — a 98+ percentage point gap. Length-of-stay analysis identified operational bottlenecks concentrated among low-probability groups, resulting in increased resource consumption and reduced shelter capacity efficiency. The logistic regression model confirmed that animal type and intake condition are the strongest predictors of adoption outcomes, providing a quantitative framework for prioritizing medical treatment, behavioral intervention, and targeted marketing.

**Business Recommendations:** Shelters should flag high-risk animals at intake for immediate behavioral assessment and targeted outreach rather than waiting for the standard adoption queue. Animals with low predicted adoption scores should receive proactive marketing through social media spotlights, rescue partner outreach, and foster placement programs to reduce length of stay and free capacity. Future iterations should incorporate additional features such as breed, color, time of year, and intake source to improve model accuracy and enable real-time risk scoring at the point of intake.

---

## Impact

- Created a data-driven system to identify high-risk animal groups with low expected adoption rates, enabling focused medical, behavioral, and marketing interventions to improve outcomes.
- Improved operational visibility by quantifying adoption inequalities and length-of-stay bottlenecks, allowing for more effective resource allocation and shelter capacity planning.
- Delivered executive-level dashboards converting predictive model outputs into actionable KPIs, supporting strategic decision-making and long-term adoption performance monitoring.
