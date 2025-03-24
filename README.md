# Graduation Project 2025

Main repository contains codework for Graduation Project 2025 led by Recep Furkan Akın and Ziya Kadir Tokluoğlu and GTU directed by Kursat Ince and Salih Sarp.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Advanced EDA](#advanced-eda)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project aims to analyze battery performance data to understand the degradation patterns and estimate the state of health (SOH) of batteries over time. The analysis includes both basic and advanced exploratory data analysis (EDA) techniques.

## Project Structure
```
.
├── basic_EDA.ipynb
├── advanced_EDA.ipynb
├── data
│   └── battery_alt_dataset
│       └── regular_alt_batteries
│           └── battery00.csv
├── README.md
└── requirements.txt
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:        
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To run the notebooks, you can use Jupyter Notebook or Jupyter Lab. Start Jupyter Notebook with the following command:
```sh
jupyter notebook
```
Then open `basic_EDA.ipynb` or `advanced_EDA.ipynb` to explore the data and analysis.

## Data
The dataset used in this project is located in the `data/battery_alt_dataset/regular_alt_batteries/` directory. The main file used for analysis is `battery00.csv`.

## Exploratory Data Analysis (EDA)
The `basic_EDA.ipynb` notebook includes:
- Dataset information and summary statistics
- Missing values analysis
- Duplicate rows handling
- Histograms and boxplots for numerical features

## Advanced EDA
The `advanced_EDA.ipynb` notebook includes:
- Handling missing values
- Outlier detection using the IQR method
- Scatter plots for visualizing outliers
- Cycle analysis based on mode changes
- Calculation of cycle metrics such as duration, discharge capacity, and energy used
- State of Health (SOH) estimation and trend analysis

## Results
The results of the analysis are documented within the notebooks. Key findings include:
- Identification of outliers in temperature and voltage data
- Estimation of battery SOH over cycles
- Trend analysis of capacity loss per cycle

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements
We would like to thank our project advisors Kursat Ince and Salih Sarp for their guidance and support.