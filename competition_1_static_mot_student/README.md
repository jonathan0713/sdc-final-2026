# Competition 1: Static-Ego 4D Radar Multi-Object Tracking

## Folder Structure

- `public_dev/seq_1/`
  - Development sequence.
  - Includes radar data, cluster masks, images, and ground truth.
  - Use this sequence for local testing.

- `test_input/seq_2/`
  - Kaggle test sequence.
  - Includes radar data, cluster masks, and images.
  - Ground truth is hidden. Generate `result.csv` and submit it to Kaggle.

- `starter_code/`
  - Starter tracking code with TODO sections.
  - Main entry point: `main.py`

- `sample_submission_seq2.csv`
  - Kaggle submission format reference.

## Quick Start

```bash
cd starter_code
pip install -r requirements.txt
```

## Run on development sequence

```bash
python main.py \
  --data-root ../public_dev \
  --seq seq_1 \
  --output-root ../outputs
```

## Evaluate on development sequence:

```bash
python evaluate_tracking.py \
  --gt ../public_dev/seq_1/gt_answer_seq1.csv \
  --pred ../outputs/seq_1_*/result.csv \
  --out ../outputs/seq_1_eval.csv
```

## Run on Kaggle test sequence:

```bash
python main.py \
  --data-root ../test_input \
  --seq seq_2 \
  --output-root ../outputs
```

## Submit the generated file:
../outputs/seq_2_*/result.csv