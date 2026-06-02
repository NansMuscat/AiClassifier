# IFLS Food Classifier

Hierarchical food product classifier using DistilBERT + LoRA.
Predicts the IFLS segment (group > family > subfamily > segment) from a product name.

## Setup

```bash
# Create (once)
python -m venv .venv

# Activate (every session)
source .venv/bin/activate          # Linux / Mac
.venv\Scripts\activate   

# Install
pip install -e ".[dev]"
```

## Data format

`data/products.csv` must contain these columns:

| column | type | description |
|---|---|---|
| `product_name` | string | Raw product name |
| `group_id` | long | DB primary key for the IFLS group |
| `family_id` | long | DB primary key for the IFLS family |
| `subfamily_id` | long | DB primary key for the IFLS subfamily |
| `segment_id` | long | DB primary key for the IFLS segment |

Here is the equivalent query from database:
``` sql
SELECT p.nom_public as product_name, 
	ncap.id_groupe as group_id, 
	ncap.ID_FAMILLE as family_id,
	ncap.ID_SOUS_FAMILLE as subfamily_id,
	ncap.ID_SEGMENT as segment_id
FROM produit p 
join AIDOMENUOBJET_E_CLASSIFICATION aec on (aec.id_objet = p.id_produit)
join NIVEAU_CLASSIFICATION_A_PLAT ncap on (ncap.id_niveau = aec.id_niveau)
```

## Train

```bash
python scripts/train.py --config config/base.yaml
```

Outputs:
- `models/best/` — best checkpoint
- `models/best/temperature.pt` — calibration scalar
- `models/best/results.json` — accuracy, loss, temperature
- `data/hierarchy.json` — ID↔index mappings + tree structure

## Predict

```bash
python scripts/predict_batch.py \
    --input data/new_products.csv \
    --output data/predictions.csv
```

Output adds columns: `segment_id`, `subfamily_id`, `family_id`, `group_id`,
`confidence_segment`, `confidence_subfamily`, `confidence_family`, `confidence_group`, `uncertain`.

## Test

```bash
pytest
```

Tests are CPU-only and take ~30 seconds. No GPU required.

## Config

All hyperparameters live in `config/base.yaml`. Create overrides:

```bash
cp config/base.yaml config/quick_run.yaml
# edit epochs: 1, batch_size: 32
python scripts/train.py --config config/quick_run.yaml
```
