# AI-based Trading Strategies

This repository presents a comprehensive framework for developing AI-driven trading strategies focusing on systematic risk premia. It integrates state-of-the-art machine learning techniques with quantitative finance methodologies to create, validate, and backtest trading signals.

## Description:  

This project deploys a systematic, AI-driven framework to develop and evaluate quantitative trading strategies. Our modular pipeline integrates robust API-based data acquisition with advanced feature engineering—including regime detection via Hidden Markov Models—to produce high-quality inputs for our models. We implement a suite of deep learning architectures, including a GRU, a standalone LSTM model, and an LSTM-CNN hybrid, all validated through rolling cross-validation and realistic backtesting that incorporates transaction cost adjustments. Performance is benchmarked against a buy-and-hold baseline, and the framework supports rapid experimentation with alternative architectures such as LSTM with attention mechanisms and transformers. 


## Overview

The project explores the intersection of artificial intelligence and financial markets by:
- **Identifying risk premia:** Leveraging historical market data to capture systematic risk factors.
- **Building predictive models:** Utilizing machine learning algorithms (e.g., neural networks) to forecast asset returns.
- **Constructing trading strategies:** Translating predictive insights into actionable trading signals.
- **Backtesting performance:** Evaluating the robustness of strategies through rigorous historical simulations that account for transaction costs and risk metrics.

## Motivation

The financial landscape is increasingly driven by data and algorithmic decision-making. By combining AI with quantitative finance, this project aims to:
- Enhance signal detection in noisy market environments.
- Optimize asset allocation and risk management.
- Provide a reproducible framework for testing and refining trading strategies.
- Contribute to the evolving research in systematic risk premia and machine learning-based investment strategies.

## Methodology
1. Data Collection & Preprocessing:

- Aggregation of historical market data (prices, volumes, macroeconomic indicators).
- Cleaning and normalization of raw data.
- Feature engineering to extract relevant predictors for risk premia analysis.

2. Model Development:

- Implementation of various machine learning models to forecast returns.
- Use of hyperparameter tuning and cross-validation to optimize model performance.
- Comparative analysis of models based on predictive accuracy and risk-adjusted performance.

3. Strategy Construction & Backtesting:

- Generation of trading signals from model outputs.
- Development of systematic trading strategies incorporating risk management (e.g., stop-loss, position sizing).
- Backtesting the strategies over historical periods to assess performance metrics such as Sharpe ratio, maximum drawdown, and cumulative returns.

## Installation & Requirements
The project is built using Python 3.8+ and relies on several key libraries. To set up the environment:

1. Clone the repository:

git clone https://github.com/FranQuant/AI-based-Trading-Strategies.git

cd AI-based-Trading-Strategies

2. Set up a virtual environment (optional but recommended):

python -m venv venv

source venv/bin/activate  

3. Install dependencies:

pip install -r requirements.txt

## Contributing
Contributions are welcome! If you have ideas for improvements, please feel free to open an issue or submit a pull request. When contributing, please adhere to the repository’s coding style and document any changes for clarity.

## License
This project is licensed under the MIT License.








