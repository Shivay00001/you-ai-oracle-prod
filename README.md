# YOU.AI Oracle Service (Production)

A production-ready, rule-based AI Guardian for DAO governance. It evaluates proposals based on a defined founder personality and core values, executing decisions on-chain without LLM dependencies.

## Features

- **Rule-Based Decision Engine**: deterministic evaluation logic.
- **Founder Personality**: configurable vision, risk tolerance, and red flags.
- **Smart Contract Integration**: Interacts with YOUDAO and YOUAIGuardian contracts.
- **Redis & SQLite Support**: For caching and state management.
- **Metrics & Monitoring**: Tracks approvals, rejections, and health status.

## Configuration

Requires a `config.yaml` file with Ethereum RPC, contract addresses, and Oracle private key.

## Usage

```bash
python you_ai_oracle_prod.py
```

## Dependencies

- `web3`
- `redis`
- `numpy`
- `scikit-learn`
- `pyyaml`
