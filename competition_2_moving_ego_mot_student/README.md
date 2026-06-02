# Competition 2: Moving-Ego 4D Radar Multi-Object Tracking

## Folder Structure

- `public_dev/seq_3/`
  - Development sequence.
  - Includes radar data, cluster masks, images, ego pose, and ground truth.
  - Use this sequence for local testing.

- `test_input/seq_4/`
  - Kaggle test sequence.
  - Includes radar data, cluster masks, images, and ego pose.
  - Ground truth is hidden. Generate `result.csv` and submit it to Kaggle.

- `starter_code/`
  - Starter tracking code with TODO sections.
  - Main entry point: `main.py`

- `sample_submission_seq4.csv`
  - Kaggle submission format reference.

## Ego-Motion Information

Competition 2 provides: ego_global_pos.txt
Use this file to design ego-motion compensation or ego-aware association costs.

## Quick Start

```bash
cd starter_code
pip install -r requirements.txt
```

## Run on development sequence:

```bash
python main.py \
  --data-root ../public_dev \
  --seq seq_3 \
  --output-root ../outputs
```

## Evaluate on development sequence:

```bash
python evaluate_tracking.py \
  --gt ../public_dev/seq_3/gt_answer_seq3.csv \
  --pred ../outputs/seq_3_*/result.csv \
  --out ../outputs/seq_3_eval.csv
```

## Run on Kaggle test sequence:

```bash
python main.py \
  --data-root ../test_input \
  --seq seq_4 \
  --output-root ../outputs
```

## Submit the generated file:
../outputs/seq_4_*/result.csv