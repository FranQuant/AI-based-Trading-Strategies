# AI-based Trading Strategies: Comparing Deep Learning Architectures

This repository presents a comprehensive framework for developing AI-driven trading strategies focusing on systematic risk premia. It integrates state-of-the-art machine learning techniques with quantitative finance methodologies to create, validate, and backtest trading signals.

## Data Source

This project utilizes historical market data obtained through the **[EODHD APIs](https://eodhistoricaldata.com/financial-apis/)**. Their high-quality financial data has been instrumental in developing and testing AI-based trading strategies.

Special thanks to **[EODHD](https://eodhistoricaldata.com/financial-apis/)** for supporting this research.

## Description:  

This project deploys a systematic, AI-driven framework to develop and evaluate quantitative trading strategies. Our modular pipeline integrates robust API-based data acquisition with advanced feature engineering—including regime detection via Hidden Markov Models—to produce high-quality inputs for our models. We implement a suite of deep learning architectures, including a GRU, a standalone LSTM model, and an LSTM-CNN hybrid, all validated through rolling cross-validation and realistic backtesting that incorporates transaction cost adjustments. Performance is benchmarked against a buy-and-hold baseline, and the framework supports rapid experimentation with alternative architectures such as LSTM with attention mechanisms and transformers. 


## Project Structure
```
AI-based-Trading-Strategies/
│
├── envs/
│   ├── env_models_simple.yml      # LSTM, GRU, CNN-LSTM
│   ├── env_models_tf.yml          # Attention LSTM, Transformer
│   ├── env_comparison.yml         # vectorbt and reporting
│
├── notebooks/
│   ├── LSTM_model.ipynb
│   ├── GRU_model.ipynb
│   ├── CNN_LSTM_model.ipynb
│   ├── ATT_LSTM_model.ipynb
│   ├── Transformer_model.ipynb
│   ├── compare_models_vectorbt.ipynb  
│
├── data/
│   ├── df_lstm.csv
│   ├── df_gru.csv
│   ├── df_cnn.csv
│   ├── df_att.csv
│   ├── df_trans.csv
│   ├── GSPC_fixed.csv              
│
├── README.md
├── requirements.txt
└── .gitignore
```


## Neural Networks for Trading Strategies

Neural networks (NNs) have been widely adopted in trading strategies due to their ability to model nonlinear relationships, detect patterns, and adapt to changing market conditions. The evolution of deep learning architectures, including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformer-based models, has further enhanced their predictive power. NNs have revolutionized trading by enhancing prediction accuracy, optimizing execution, and identifying alpha signals in ways that traditional methods cannot. However, challenges such as interpretability, overfitting, and computational cost must be addressed for robust deployment.

Long Short-Term Memory (LSTM) networks are a specialized type of RNNs designed to handle sequential data and long-term dependencies—making them particularly well-suited for financial time-series forecasting and trading strategies. LSTMs are a powerful tool for time-series forecasting and trading, but they require careful tuning and robust validation. With recent advancements in deep learning, transformer architectures have proven highly effective in time-series forecasting, outperforming LSTMs and traditional statistical models.

#### LSTM with Attention-Based Trading Strategy

Our approach leverages a state-of-the-art deep learning architecture that combines the sequential modeling strengths of LSTM networks with an attention mechanism designed to isolate and weigh the most predictive segments of historical market data. This attention-based framework also opens promising avenues for systematic exploration in asset allocation strategies, enabling dynamic weighting of assets informed by adaptive market regime detection.

## Overview

The project explores the intersection of artificial intelligence and financial markets by:
- **Identifying risk premia:** Leveraging historical market data to capture systematic risk factors.
- **Building predictive models:** Utilizing machine learning algorithms (e.g., neural networks) to forecast asset returns.
- **Constructing trading strategies:** Translating predictive insights into actionable trading signals.
- **Backtesting performance:** Evaluating the robustness of strategies through rigorous historical simulations that account for transaction costs and risk metrics.

## Motivation

Data and algorithmic decision-making increasingly drive the financial landscape. By combining AI with quantitative finance, this project aims to:
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

  
4. Evaluation & Performance Review
- Visualization of Results. 
- Use QuantStats to generate: Cumulative returns plots, Drawdown analysis, Rolling Sharpe ratio plots, etc...
- Compare the strategy performance against benchmarks to determine if the model adds value. 

## Installation & Requirements

The project is built using Python 3.8+ and relies on several key libraries. To set up the environment:

1. Clone the repository:
```
git clone https://github.com/FranQuant/AI-based-Trading-Strategies.git
```

2. Set up a virtual environment (optional but recommended):
```
python -m venv venv
 ```

3. Install dependencies:
```
pip install -r requirements.txt
```
## Contributing

Contributions are welcome! If you have ideas for improvements, please feel free to open an issue or submit a pull request. When contributing, please adhere to the repository’s coding style and document any changes for clarity.

## License

This project is licensed under the MIT License.








