# LightCred

Official implementation of LightCred, a consortium blockchain-based personal credit management framework integrating lightweight nodes, Merkle proof-based verification, smart contract-driven access and incentive mechanisms, and privacy-preserving credit services.

This repository provides the Python experimental framework used to generate the main evaluation results reported in the manuscript:

**“A Personal Credit Management Scheme for Consortium Blockchains Integrating Smart Contracts and Light Node Mechanisms”**

## Overview

LightCred is designed to support scalable, auditable, and privacy-aware personal credit management in consortium blockchain environments. The framework combines:

- lightweight node simulation;
- Merkle tree construction and proof verification;
- blockchain-style transaction and storage simulation;
- smart contract-inspired access control and reputation-based incentive mechanisms;
- privacy-preserving credit service simulation;
- system-level benchmarking under synthetic and public credit-risk datasets.

The code in this repository focuses on the experimental evaluation pipeline used in the manuscript, including throughput, latency, storage overhead, privacy leakage, audit verification time, F1-score, and Brier score.

## Repository Structure

```text
LightCred/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
└── lightcred_experiment_framework.py
```

The main script is:

```text
lightcred_experiment_framework.py
```

It provides the complete experimental framework for running synthetic benchmarks, public dataset benchmarks, ablation studies, result aggregation, and figure generation.

## Requirements

The code was implemented in Python. The main dependencies are:

```text
numpy
pandas
matplotlib
scikit-learn
```

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` file can be written as:

```text
numpy
pandas
matplotlib
scikit-learn
```

## Usage

### 1. Run the default synthetic benchmark

To run the default synthetic benchmark:

```bash
python lightcred_experiment_framework.py
```

By default, the script runs the synthetic benchmark and saves outputs to:

```text
./lightcred_outputs/
```

### 2. Specify the number of trials and transactions

```bash
python lightcred_experiment_framework.py --trials 5 --transactions 200000
```

Main options include:

```text
--output-dir       Directory to save CSV files and figures
--trials           Number of independent trials
--transactions     Synthetic transaction count
--batch-size       Simulated block/commit batch size
--seed             Random seed
```

### 3. Run only the synthetic benchmark

```bash
python lightcred_experiment_framework.py --skip-public
```

### 4. Run public credit-risk dataset experiments

The framework supports public credit-risk datasets such as:

- Default of Credit Card Clients;
- Statlog German Credit Data.

You can run the public dataset experiments by providing local CSV files:

```bash
python lightcred_experiment_framework.py \
  --statlog-csv ./data/statlog.csv \
  --statlog-target default \
  --defaultcc-csv ./data/default_credit_card.csv \
  --defaultcc-target default.payment.next.month
```

If the target column name differs from the examples above, please specify the corresponding column using:

```text
--statlog-target
--defaultcc-target
```

### 5. Run ablation studies

To run ablation studies on the public datasets:

```bash
python lightcred_experiment_framework.py \
  --statlog-csv ./data/statlog.csv \
  --statlog-target default \
  --defaultcc-csv ./data/default_credit_card.csv \
  --defaultcc-target default.payment.next.month \
  --run-ablation
```

The ablation settings include variants such as:

```text
w/o LightweightNodes
w/o MerkleProof
w/o ReputationSC
w/o ZKP_HE
```

## Outputs

After execution, the script generates the following outputs:

```text
lightcred_outputs/
├── raw_trial_results.csv
├── summary_results.csv
├── experiment_config.json
└── figures/
    ├── synthetic_1M_style_tps.png
    ├── synthetic_1M_style_latency.png
    ├── synthetic_1M_style_privacy.png
    ├── synthetic_1M_style_audit.png
    └── ...
```

The main output files are:

- `raw_trial_results.csv`: raw results for each independent trial;
- `summary_results.csv`: aggregated results with mean, standard deviation, and 95% confidence intervals;
- `experiment_config.json`: hardware, workload, and method configurations used in the experiment;
- `figures/`: generated figures for throughput, latency, storage, privacy leakage, audit verification time, F1-score, and Brier score.

## Compared Methods

The framework evaluates LightCred against representative baseline methods, including:

```text
CCB
PB-OC
CB-Full
CB-DP
OC-PPC
RA-AC
BT-SCS
LightCred
```

These methods represent different credit data management, blockchain storage, privacy-preserving, and access-control settings.

## Evaluation Metrics

The following metrics are reported:

- throughput;
- latency;
- storage overhead;
- privacy leakage;
- audit verification time;
- F1-score;
- Brier score.

For each method and workload setting, the results are summarized across independent trials. The output summary includes mean values, standard deviations, and 95% confidence intervals where applicable.

## Reproducibility Notes

To improve reproducibility, the script records the experimental configuration in:

```text
experiment_config.json
```

The script also supports user-defined random seeds through:

```bash
--seed
```

For example:

```bash
python lightcred_experiment_framework.py --seed 42 --trials 5
```

## Access and Restrictions

The code is publicly available for academic and research purposes. No special permission is required to access this repository.

This repository does not include private credit records, personally identifiable information, confidential institutional data, or any proprietary deployment configuration. Public datasets should be obtained from their original sources.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

The formal citation information will be updated after publication.

## Contact

For questions regarding the code or experimental reproduction, please contact the corresponding author listed in the manuscript.
