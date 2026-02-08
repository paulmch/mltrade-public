# Redaction Notice

This repository is a sanitized version of a private multi-year ML trading project. It is shared as an engineering portfolio showcase, not as a turnkey trading system. The redaction process is automated by a build script (not included in this repository) and follows three tiers.

---

## Tier 1: Secrets

All credentials and secret values have been replaced with placeholder tokens.

| Original | Replacement |
|----------|-------------|
| AWS account IDs | `REDACTED_AWS_ACCOUNT_ID` |
| API keys (TwelveData, xAI, etc.) | `REDACTED_API_KEY` |
| Broker credential secret names | `BROKER_CREDENTIALS` |

These values were used for authentication with AWS services, market data APIs, and broker platforms. None of the original values appear anywhere in this repository.

---

## Tier 2: Infrastructure

AWS resource names and identifiers have been genericized to prevent mapping to live infrastructure.

| Original | Replacement |
|----------|-------------|
| S3 bucket names | `REDACTED_BUCKET` |
| S3 URLs (`s3://bucketname/`) | `s3://REDACTED_BUCKET/` |
| SNS topic ARNs | `arn:aws:sns:REGION:ACCOUNT_ID:TOPIC` |
| IAM role ARNs | `arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME` |
| Step Functions ARNs | `arn:aws:states:REGION:ACCOUNT_ID:stateMachine:STATE_MACHINE` |

The code references to these resources remain intact so the architecture is readable, but the actual resource names are hidden.

---

## Tier 3: Strategy Parameters

V2 trading parameters, thresholds, and knockout factors are hidden. These include:

- Knockout barrier percentages and adjustment factors
- Stop-loss thresholds
- Confidence thresholds for trade entry/exit
- Model architecture hyperparameters in saved config files (excluded from the repository)
- Trained model weights (`.h5`, `.h5f`, `.pkl` files are excluded)
- Training data files (`.csv`, `.npz` files are excluded)

The V2 code structure and logic are visible, but the specific numerical parameters that defined the trading strategy are not included. Note: the V2 strategy is also no longer actively traded.

---

## Legacy Code

All legacy code (the `legacy/` directory) is shown in full, including strategy parameters, model architectures, and decision logic. This strategy operated on the DAX index from 2021 to 2023 and is no longer active. It is preserved as a complete engineering reference.

---

## Notebooks

Jupyter notebooks have had their outputs stripped (cell outputs cleared, execution counts reset) to remove any rendered charts, model metrics, or data samples that could reveal strategy performance characteristics.

---

## Verification

The sanitization build includes a verification step that scans every file in the output for known secret patterns. The build fails if any secrets are detected in the output.

---

## Purpose

This repository exists to demonstrate:

- System architecture and evolution across multiple years
- ML engineering practices (reinforcement learning, NCP/LTC, Bayesian optimization, LLM integration)
- Infrastructure design on AWS (Lambda, Step Functions, DynamoDB, S3)
- Creative automation (Selenium-based broker interaction, binary search through infinite scroll)

It is not intended as a reproducible trading system and should not be used as one.
