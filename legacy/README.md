# Legacy DAX Trading System (2021-2023)

An ensemble of 10 Deep Q-Network reinforcement learning agents deployed across 24 AWS Lambda modules, orchestrated by Step Functions, trading DAX knock-out options on IG.com via headless Selenium.

---

## Architecture

The system ran on a daily cycle: download market data, run the ensemble, score confidence, decide whether to trade, execute via Selenium, and log performance. Day and night sessions had separate model ensembles, each containing 10 DQN agents that voted on trade direction.

**Orchestration:** AWS Step Functions coordinated the Lambda invocations in sequence, with branching for day vs. night paths and error handling via SNS notifications.

**State:** DynamoDB tables stored runtime state (current direction, turnaround signals, performance metrics, RSI values, alpha parameters). S3 stored model weights, historical data, deviation logs, and confidence files.

---

## Module Reference

### Data Pipeline

| Module | Description |
|--------|-------------|
| `downloaddax` | Downloads daily DAX OHLC data and holiday calendar from S3, appends new trading days |
| `NDXdownload` | Downloads daily NASDAQ-100 data from S3, parallel data source for cross-index analysis |
| `newdailydata` | Daily data enrichment pipeline with technical indicators (ta library), timezone-aware scheduling |
| `prepdaydata` | Fetches 5-minute intraday DAX data from TwelveData API, computes technical indicators via ta library, runs Bayesian hyperparameter optimization for decision models |

### Ensemble Models

| Module | Description |
|--------|-------------|
| `daymodel` | Day session ensemble: 10 DQN agents with LSTM layers, trained on DAX data in a custom Gym environment, each voting on direction |
| `nightmodel` | Night session ensemble: identical architecture to daymodel but trained on overnight price movements |
| `ndxagent` | NASDAQ-100 reinforcement learning agent: 10 DQN models with Boltzmann Q-Policy in a custom Gym stock environment |

### Decision Layer

| Module | Description |
|--------|-------------|
| `decidemodel` | Combines ensemble votes with deviation data (predicted vs. actual OHLC), computes sigma-normalized features, generates trade signals |
| `decidemodelrange` | Range-based refinement of decision signals using deviation statistics and turnaround detection |
| `decidemodelupdate` | Updates decision model parameters based on recent performance, adjusts turnaround thresholds |
| `decidermodelnight` | Night session decision model: loads pre-trained weights from S3, evaluates overnight signals |
| `decidermodelstats` | Statistical analysis of decision quality using Bayesian optimization to tune decision boundaries |

### Confidence Scoring

| Module | Description |
|--------|-------------|
| `dayconfidence` | Confidence scoring model: 3-layer dense network (LeakyReLU, Dropout, BatchNorm) evaluating ensemble agreement and market features |
| `dayconfidenceml` | ML-based confidence gate: reads performance and RSI state from DynamoDB, decides whether the system should trade today |
| `collectdayactions` | Aggregates daily confidence actions from S3 into a summary CSV for performance tracking |

### Execution

| Module | Description |
|--------|-------------|
| `autotraderig` | Selenium-based trade execution on IG.com: headless Chrome in Lambda, logs into IG, navigates to Turbo24 account, places knock-out option orders |
| `selldecision` | Evaluates exit conditions using deviation-normalized metrics, computes sell confidence, triggers trade closure or holds |
| `endtrade` | Closes open positions: loads model weights, evaluates current market state, executes sell via the autotrader |
| `sendMail` | Trade notification dispatcher: sends signals via SNS, invokes the autotrader Lambda for execution |

### Analytics & Optimization

| Module | Description |
|--------|-------------|
| `collectperformance` | Records strategy performance metrics to S3 for historical analysis |
| `collectdeviation` | Computes mean and sigma of prediction deviations across the 10-agent ensemble, appends to deviation history |
| `oppositestrategy` | Contrarian analysis: loads model weights and evaluates what would happen if the system traded in the opposite direction |
| `autohyperparametertuning` | Bayesian hyperparameter optimization: tunes layer sizes, dropout rates, LeakyReLU alphas, epochs, and batch sizes using `bayes_opt` |
| `fargatehyperparametertuning` | Fargate-based training: runs longer optimization cycles on ECS Fargate when Lambda's 15-minute timeout is insufficient |

---

## DQN Agent Details

Each of the 10 agents:
- Uses a Sequential model with Dense + LSTM + LeakyReLU + Dropout + BatchNormalization layers
- Operates in a custom `stock` Gym environment that simulates knock-out option P&L
- Has a discrete action space of 19 actions (varying position sizes and directions)
- Uses `LinearAnnealedPolicy` wrapping `BoltzmannQPolicy` for exploration
- Is trained with `SequentialMemory` for experience replay

The ensemble votes are aggregated -- if a majority of agents agree on a direction, the system generates a signal. Confidence scoring then decides whether the signal is strong enough to trade.

---

## Confidence Scoring

The confidence pipeline works in two stages:

1. **dayconfidence**: A neural network evaluates the ensemble's prediction spread (up/down point estimates) alongside date features (weekday, month-day one-hot encoded) and deviation history. Outputs a probability that the signal will be profitable.

2. **dayconfidenceml**: A DynamoDB-backed gate that checks running performance means, opposite-strategy performance, RSI indicators, and historical alpha to decide whether the system should be allowed to trade. This prevents trading during drawdown periods.

---

## Notes

- All legacy code is shown in full, including strategy parameters, since this strategy is no longer active
- Model weight files (`.h5`, `.h5f`) are excluded from the repository
- S3 bucket names have been replaced with `REDACTED_BUCKET`
- The IG.com Selenium automation was engineered to run inside AWS Lambda with a Chrome binary layer -- a notably constrained environment for browser automation
