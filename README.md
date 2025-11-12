# Health Insurance Cost Prediction 🏥

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/stable/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A machine learning project to predict health insurance costs based on various personal and demographic factors. Using Random Forest Regressor, the model achieves 86% accuracy in predicting insurance charges.

## Dataset Overview

The dataset contains information about health insurance beneficiaries, including their:
- Age
- Sex
- BMI (Body Mass Index)
- Number of Children
- Smoking Status
- Region
- Insurance Charges

Dataset Source: [Kaggle - Healthcare Insurance Dataset](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance)

## Key Insights

### Distribution of Insurance Charges
- Insurance charges show a right-skewed distribution
- Majority of charges fall between 0-20,000 USD
- Some extreme cases exceed 60,000 USD

### Impact of Smoking
- Smokers consistently face higher insurance charges
- The median charge for smokers is approximately 3-4 times higher than non-smokers
- Smoking status is one of the most significant factors affecting insurance costs

### BMI and Charges Correlation
- Higher BMI generally correlates with increased charges
- This correlation is particularly strong for smokers
- Non-smokers show a more moderate increase in charges with BMI

### Age and Cost Relationship
- Insurance charges tend to increase with age
- The relationship appears to be more pronounced for smokers
- Age-related increase is more gradual for non-smokers

## Technical Implementation

### Data Preprocessing
- Label encoding for categorical variables (sex, smoker)
- One-hot encoding for regional data
- No missing values or anomalies were detected in the dataset

### Technologies Used
- Python 3.x
- Libraries:
  - pandas - Data manipulation and analysis
  - numpy - Numerical operations
  - scikit-learn - Machine learning implementation
  - matplotlib - Data visualization
  - seaborn - Statistical data visualization

### Model Development
- Algorithm: Random Forest Regressor
- Performance: 86% accuracy
- Features: All preprocessed variables used in prediction

## Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/toshimajaiswal/Health-Insurance-Prediction.git
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook
```

## Results and Performance

The Random Forest Regressor model achieved:
- Accuracy Score: 86%
- Capable of predicting insurance charges based on personal factors
- Effectively captures non-linear relationships in the data

## Author

**Toshima Jaiswal**
- GitHub: [toshimajaiswal](https://github.com/toshimajaiswal)
- LinkedIn: [Toshima Jaiswal](https://www.linkedin.com/in/toshimajaiswal/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by Kaggle
- Inspiration from real-world insurance pricing challenges
- Open-source community for various tools and libraries used

