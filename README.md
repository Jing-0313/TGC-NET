# TGC-Net

<p align="center">
  <b>TGC-Net: Residue-Level Fusion of Multi-Scale Sequence Motifs and Structural Microenvironments for Enzyme Optimal Temperature Prediction</b>
</p>

Enzyme optimal temperature (**Topt**) is the assay temperature at which catalytic activity is maximal. Measuring it requires protein expression, purification, and activity assays across multiple temperatures, so computational prediction is useful for prioritizing candidates before laboratory testing. TGC-Net is an **OGT-independent, residue-aligned sequence–geometry model** developed for this purpose.

The model addresses a specific representation problem: short-range sequence patterns and folding-induced spatial neighborhoods provide complementary context for the same residues, yet many predictors rely mainly on sequence or OGT, and structure-aware approaches often encode the two contexts separately or combine already extracted features. TGC-Net keeps the pretrained sequence pathway as the reference representation and uses continuous geometry to condition how structural context is formed, selected, and incorporated at aligned residue positions.

The repository also contains workflows for:

* fixed held-out Topt evaluation and comparison with published baselines;
* structural-quality analysis based on Avg_pLDDT;
* residue-level interpretability and perturbation analysis;
* a 32-β-agarase case study;
* temporal external validation on newly reported wild-type enzymes;
* WT–mutant response analysis;
* extension of the same core architecture to protein melting temperature (**Tm**) and enzyme optimal pH prediction.

TGC-Net does **not** require source-organism optimal growth temperature (OGT) as a model input. It is intended for approximate Topt estimation and candidate prioritization, not as a replacement for biochemical measurement.

---

## 1. Research question and model logic

TGC-Net is organized around one question: **how should local sequence patterns and folded spatial environments jointly refine the representation of the same residue?** The four functional levels answer different parts of that question while retaining a shared residue index.

| Functional level | Information problem | TGC-Net design and purpose |
| --- | --- | --- |
| Sequence | A pretrained language model captures broad context but may not emphasize short motifs at several local scales. | LoRA adapts ESM-2 to the Topt task, while parallel convolutions with kernel sizes 3, 5, and 7 provide short-range patterns. A residue- and channel-wise gate adds them as controlled corrections, preserving the pretrained representation as the main pathway. |
| Structure | Sequence-near residues and spatially near residues are not equivalent after folding; binary graph connectivity also discards within-neighborhood distance variation. | The corrected sequence state initializes a two-layer Geometric GNN. Continuous Cα distances are encoded within 15 Å neighborhoods so that folded-neighborhood membership and distance variation can both condition the aligned residue state. |
| Fusion | Retrieving relevant structural context and deciding how strongly it should alter a sequence state are distinct operations. | Four-head cross-attention uses head-specific continuous-distance biases to select geometry-conditioned context. A feature-wise gate derived from squared sequence–structure discrepancy calibrates residual context incorporation. The gate is learned and should not be interpreted as a fixed monotonic confidence rule. |
| Output | Uniform protein-level averaging can dilute signals concentrated at particular residues or feature channels. | Lightweight Residue-aware Attention Pooling (L-RAP) combines masked residue attention with feature-channel gating before three residual dense blocks perform continuous regression. |

The sequence and structural pathways are therefore not independent feature extractors. The structural branch starts from the corrected sequence states, geometry conditions their folded microenvironments, and the fusion stage separately controls context selection and update strength before selective protein-level aggregation.

The implemented forward path is:

```text
LoRA-adapted ESM-2 + Multi-Scale Motif CNN
            ↓
   controlled local correction
            ↓
 continuous-distance geometric conditioning
            ↓
 geometry-guided context selection and calibration
            ↓
 selective residue/channel aggregation
            ↓
        3 × RDBlock
            ↓
       predicted Topt
```

### Main findings reported in the manuscript

* On the fixed 291-protein held-out test set, TGC-Net achieved **R² = 0.624**, **RMSE = 11.42 °C**, and **MAE = 8.23 °C**.
* Relative to Seq2Topt on the same test set, R² was 10.1% higher, RMSE 6.9% lower, and MAE 7.4% lower.
* Across 100 matched subsamples drawn from the fixed test set, TGC-Net retained a higher R² than each of the four compared baselines.
* In the 32-protein β-agarase study, 7 of the experimental top 10 candidates were recovered; among nine newly reported wild-type enzymes, TGC-Net had the lowest absolute error among the compared predictors for seven samples.

---

## 2. Repository structure

The following layout reflects the released scripts and the data organization used in this project.

```text
TGC-Net/
├── README.md
│
├── model.py
├── functions.py
│
├── csv-fastas.py
├── generate_pdbs.py
├── check-pLDDT.py
├── pdb_pt.py
│
├── topt-train.py
├── topt-test.py
├── tm-train.py
├── tm-test.py
├── ph-train.py
├── ph-test.py
│
├── interpretability.py
├── wild-type.py
├── WT-mutant.py
├── β-agarase.py
│
└── data/
    ├── compare/
    │   ├── result_deepet.csv
    │   ├── result_preoptem.csv
    │   ├── result_Seq2Topt.csv
    │   ├── result_TGC.csv
    │   ├── result_tomer.csv
    │   └── test.csv
    │
    ├── explanation/
    │   ├── residue_importance_long.csv
    │   ├── sample_summary.csv
    │   ├── perturbation_results.csv
    │   ├── perturbation_summary.csv
    │   ├── O30012_top10_residues.csv
    │   ├── O30012.fasta
    │   └── O30012.pdb
    │
    ├── topt/
    │   ├── topt_train.csv
    │   ├── topt_test.csv
    │   ├── topt_train_quality.csv
    │   ├── topt_test_quality.csv
    │   ├── topt_train_structures.pt
    │   └── topt_test_structures.pt
    │
    ├── tm/
    │   ├── tm_train.csv
    │   ├── tm_test.csv
    │   ├── tm_quality.csv
    │   └── tm_structures.pt
    │
    ├── ph/
    │   ├── ph_train.csv
    │   ├── ph_test.csv
    │   ├── ph_quality.csv
    │   └── ph_structures.pt
    │
    ├── wild-type enzymes/
    │   ├── wild-type_test.csv
    │   ├── wild-type_quality.csv
    │   ├── wild-type_structures.pt
    │   └── additional summary/result CSV files
    │
    ├── WT-mutant/
    │   ├── WT-mutant_test.csv
    │   ├── WT-mutant_quality.csv
    │   ├── WT-mutant_structures.pt
    │   ├── WT-mutant_result-TGC.csv
    │   └── annotated result table
    │
    └── β-agarase/
        ├── beta_agarase.csv
        ├── beta_agarase_quality.csv
        ├── beta_agarase_structures.pt
        └── beta_agarase_result.csv
```

The `.pt` structure files are Python dictionaries keyed by protein identifier, with each value containing an aligned Cα-coordinate tensor.

---

## 3. Development datasets

TGC-Net was developed on the public Topt benchmark released with Seq2Topt. The benchmark can be traced to BRENDA-derived records used by TOME/TOMER and contains 2,917 enzyme sequences. The same architecture was subsequently retrained independently on Tm and optimal-pH datasets to examine architectural extensibility; these are separate prediction tasks and do not represent parameter transfer from the Topt model.

| Task | Source benchmark | Original total | Original training | Fixed test | Final training | Final validation | Final test |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Topt | Seq2Topt benchmark | 2,917 | 2,626 | 291 | 2,135 | 238 | 291 |
| Tm | DeepTM | 7,790 | 6,240 | 1,550 | 4,761 | 529 | 1,477 |
| Optimal pH | EpHod | 9,855 | 7,884 | 1,971 | 6,552 | 728 | 1,885 |

For the primary Topt task, the original 291-sample test split is retained as the fixed held-out test set. It is excluded from training, Avg_pLDDT-threshold selection, hyperparameter tuning, and checkpoint selection. Candidate Avg_pLDDT thresholds are compared only on the development data, and the final checkpoint is selected by validation RMSE.

After checking structure availability and applying Avg_pLDDT ≥ 0.7 only to the development pool, the retained proteins are shuffled with `random_state=42` and split 9:1 into training and validation sets. The fixed test set is checked for structure availability but is not filtered by Avg_pLDDT.

The manuscript also notes a data-level limitation: a recent literature reassessment questioned the support for several low-temperature labels in the public Seq2Topt benchmark. The original test split is retained to preserve direct comparability with prior work, so reported benchmark results remain subject to possible label noise in the public data.

---

## 4. Input data format

The training and testing scripts expect CSV files containing at least:

```text
uniprot_id
sequence
<task target>
```

The task target column is:

```text
topt   : enzyme optimal temperature
 tm    : protein melting temperature
pHopt  : enzyme optimal pH
```

Example:

```csv
uniprot_id,sequence,topt
P00001,MKT...,37.0
P00002,MVL...,65.0
```

Structural-quality CSV files are expected to contain:

```text
Filename
Avg_pLDDT
```

The training scripts derive `uniprot_id` by removing the `.pdb` suffix from `Filename`.

---

## 5. Sequence branch

Sequence tokens and Cα coordinates are aligned before either pathway is used. Because ESM-2 inserts boundary tokens, structural coordinates begin at token position 1 and align with the residue tokens between the boundaries. A shared valid-residue mask excludes boundary tokens, padding, and positions without aligned coordinates from structural computation and protein-level pooling.

### 5.1 ESM-2

TGC-Net uses:

```text
facebook/esm2_t33_650M_UR50D
```

The 650M ESM-2 model provides 1,280-dimensional contextual residue representations. These states serve as the reference pathway that the local and geometric components refine.

Input sequences are tokenized with dynamic padding and truncated to:

```text
maximum token length = 800
```

### 5.2 LoRA adaptation

The ESM-2 backbone is adapted using LoRA with:

```text
rank (r)        = 16
alpha           = 32
dropout         = 0.1
target modules  = query, key, value, dense, intermediate.dense
```

LoRA provides task-specific adaptation without full fine-tuning of the 650M-parameter backbone. The LoRA parameters and the remaining TGC-Net modules are optimized jointly.

### 5.3 Multi-Scale Motif CNN

The local sequence branch starts from a learnable token embedding and applies three parallel one-dimensional convolutions:

```text
kernel size 3 → 426 channels
kernel size 5 → 426 channels
kernel size 7 → 428 channels
```

For a 1,280-dimensional ESM-2 backbone, the three branches concatenate back to 1,280 dimensions and are followed by LayerNorm. This branch supplies explicit short-range patterns at several sequence scales; it is not used as a replacement for the pretrained representation.

### 5.4 Adaptive Gated Fusion

The local motif representation is injected into the ESM-2 representation through a residue- and channel-wise learned gate:

```text
H_seq = H_ESM + Gate([H_ESM || H_CNN]) ⊙ H_CNN
```

This residue- and channel-wise update keeps the ESM-2 representation as the primary sequence pathway while allowing local patterns to act as learned corrections. The gate controls correction magnitude independently at each feature position; it is not an external confidence score.

---

## 6. Structural prediction workflow

TGC-Net uses ESMFold-predicted structures to derive residue-level geometry.

The structure-generation pipeline is:

```text
CSV
 ↓
csv-fastas.py
 ↓
per-protein FASTA files
 ↓
generate_pdbs.py
 ↓
ESMFold PDB structures
 ├── check-pLDDT.py → Avg_pLDDT quality CSV
 └── pdb_pt.py      → packed Cα-coordinate dictionary (.pt)
```

### 6.1 Convert CSV sequences to FASTA files

`csv-fastas.py` expects input CSV files containing:

```text
uniprot_id
sequence
```

The current helper script defines its input files and output directory at the bottom of the file. Edit these paths before execution, then run:

```bash
python csv-fastas.py
```

One FASTA file is generated for each protein identifier.

### 6.2 Generate ESMFold structures

`generate_pdbs.py` uses:

```text
facebook/esmfold_v1
```

Basic usage:

```bash
python generate_pdbs.py \
  --input_dir <fasta_directory> \
  --output_dir <pdb_directory>
```

The current implementation:

```text
ESMFold trunk sequence length = 64
non-standard residue symbols = replaced by A
sequences > 2000 residues     = skipped
existing PDB files            = skipped
```

A CUDA device is used when available.

### 6.3 Calculate Avg_pLDDT

`check-pLDDT.py` reads Cα B-factor values from each PDB file and reports their mean as `Avg_pLDDT`.

The current script defines:

```text
pdb_folder
output_csv
```

at the top of the file. Edit these paths before running:

```bash
python check-pLDDT.py
```

The generated CSV contains:

```text
Filename
Avg_pLDDT
Rating
```

### 6.4 Pack PDB coordinates into a `.pt` file

`pdb_pt.py` extracts Cα coordinates and stores them in a dictionary keyed by protein identifier.

```bash
python pdb_pt.py \
  --pdb_dir <pdb_directory> \
  --save_path <structures.pt>
```

The PDB parser is implemented in `functions.py`.

---

## 7. Geometric GNN

The structural pathway begins from the gated sequence states rather than from an independently encoded structural vector. This shared initialization preserves residue alignment while allowing folding geometry to condition the same representations. For aligned Cα coordinates, TGC-Net computes the normalized pairwise distance matrix:

```text
D = cdist(Cα, Cα) / 10
```

A normalized threshold of `1.5` therefore corresponds to a physical cutoff of **15 Å**.

The Geometric GNN uses:

```text
number of layers       = 2
edge hidden dimension  = 64
local cutoff           = 15 Å
```

Continuous distance values are mapped by:

```text
Linear(1, 64) → SiLU → Linear(64, 64)
```

and summed over residue pairs inside the 15 Å local neighborhood. The resulting distance-derived geometric summary is concatenated with the current residue representation and transformed through a residual node update. In the released implementation, messages are derived from continuous edge distances; semantic neighbor states are not propagated as edge messages.

The cutoff determines which residue pairs belong to a folded local neighborhood, while the continuous distances preserve variation among pairs within that neighborhood. The 15 Å cutoff is used specifically for **local Geometric GNN computation**.

---

## 8. Geometry-guided context selection and calibration

TGC-Fusion performs four-head asymmetric cross-attention to retrieve structural context for each sequence position.

```text
Query : sequence representation
Key   : geometry-enhanced representation
Value : geometry-enhanced representation
heads : 4
```

Continuous pairwise distances are mapped to head-specific additive attention biases using:

```text
Linear(1, 16) → SiLU → Linear(16, 4)
```

The attention score is therefore modulated by both representation compatibility and continuous geometry. This addresses **which** geometry-conditioned context should be selected for a sequence residue.

Importantly, the 15 Å hard cutoff used by the Geometric GNN is **not reapplied as a hard attention mask in TGC-Fusion**. TGC-Fusion uses the continuous pairwise distance matrix to generate additive spatial biases, while invalid Key/Value positions are excluded with the residue-validity mask.

After structural-context aggregation, TGC-Fusion computes a feature-wise calibration gate from:

```text
(H_seq - H_struct)^2
```

and incorporates the aggregated structural context through a gated residual update. This separately addresses **how strongly** the selected context should update the sequence state. The learned gate does not impose a predefined rule that larger discrepancy must produce a smaller value; its behavior is determined during training.

---

## 9. L-RAP and regression head

### 9.1 Lightweight Residue-aware Attention Pooling

TGC-Fusion returns a variable-length set of residue representations. L-RAP is used to prevent localized information from being uniformly diluted when these representations are reduced to one protein-level vector. It contains two complementary pathways.

Residue-level attention:

```text
Linear(1280, 128) → Tanh → learnable query → masked Softmax
```

Feature-channel gating:

```text
Linear(1280, 1280) → Sigmoid
```

Special-token, padding, and structurally unavailable positions are masked before Softmax. Residue attention determines which positions contribute, while the sigmoid gate recalibrates feature channels within each residue. The resulting weights describe model-internal attribution and are not causal biological evidence.

### 9.2 RDBlock

The pooled vector can still contain nonlinear relationships among sequence, motif, and geometric features. It is therefore refined by three residual blocks before scalar regression:

```text
LayerNorm
  ↓
Linear
  ↓
LeakyReLU(0.1)
  ↓
Dropout(0.15)
  ↓
Residual addition
```

Final configuration:

```text
number of RDBlocks = 3
output dimension   = 1
```

---

## 10. Training configuration

The main training configuration is:

| Setting | Value |
| --- | --- |
| Sequence backbone | `facebook/esm2_t33_650M_UR50D` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| LoRA learning rate | `1e-4` |
| TGC-Net module learning rate | `2e-4` |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Batch size | 4 |
| Gradient accumulation | 1 |
| Maximum epochs | 60 |
| Maximum token length | 800 |
| Loss | MSE |
| Gradient clipping | max norm 1.0 |
| Mixed precision | enabled on CUDA |
| Coordinate perturbation | Gaussian, σ = 0.05 Å on valid coordinates during training |
| Scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 4 |
| Minimum learning rate | `1e-6` |
| Checkpoint selection | validation RMSE only |

The retained training data are shuffled with `random_state=42` and split 9:1 into training and validation subsets.

---

## 11. Structural-quality filtering

The primary Topt workflow uses:

```text
Avg_pLDDT threshold = 0.7
```

for training/validation structural-quality filtering.

The fixed held-out test set is filtered only for structural availability; it is not filtered by the Avg_pLDDT threshold.

Avg_pLDDT is used for:

```text
training-data quality filtering
structural-confidence analysis
```

and is **not** supplied to TGC-Net as:

```text
an input feature
an attention bias
a gating coefficient
```

---

## 12. Topt resampling and label scaling

The primary Topt task uses weighted random resampling to increase exposure to underrepresented temperature ranges.

| Topt range | Sampling weight |
| --- | ---: |
| `< 20 °C` | 5.0 |
| `20–<60 °C` | 1.0 |
| `60–<80 °C` | 1.5 |
| `≥ 80 °C` | 3.0 |

Sampling is performed with replacement using `WeightedRandomSampler`.

Target ranges are:

```text
Topt       : 0–120 °C
Tm         : 0–100 °C
optimal pH : 0–14
```

Targets are normalized to `[0,1]` with these task-specific bounds for optimization and converted back to their original units for reporting.

Tm and optimal-pH training use equal sample weights rather than the Topt temperature-stratified weighting scheme.

---

## 13. Train Topt

Example:

```bash
python topt-train.py \
  --train_path data/topt/topt_train.csv \
  --test_path data/topt/topt_test.csv \
  --structure_path data/topt/topt_train_structures.pt \
  --test_structure_path data/topt/topt_test_structures.pt \
  --quality_path data/topt/topt_train_quality.csv \
  --plddt_threshold 0.7 \
  --task topt \
  --model_name facebook/esm2_t33_650M_UR50D \
  --lr 1e-4 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --epochs 60 \
  --max_length 800 \
  --save_path checkpoints/topt_best_model.pth \
  --ckpt_path checkpoints/topt_resume.pth \
  --final_pred_path results/topt_final_predictions.csv \
  --history_fig_path results/topt_training.png \
  --final_fig_path results/topt_test_prediction.png
```

The training script:

```text
1. filters training samples by structure availability and Avg_pLDDT;
2. preserves the fixed test set except for structure availability;
3. creates a 90/10 training/validation split;
4. applies Topt weighted resampling;
5. trains LoRA-adapted ESM-2 and TGC-Net jointly;
6. selects the best checkpoint by validation RMSE;
7. reloads the selected checkpoint;
8. evaluates the fixed held-out test set once;
9. saves predictions and a test scatter plot.
```

The saved best-model checkpoint contains both the merged ESM-2 state dictionary and the TGC-Net state dictionary, together with validation metrics and the selected epoch.

---

## 14. Test Topt

After obtaining a trained checkpoint:

```bash
python topt-test.py \
  --input data/topt/topt_test.csv \
  --output results/topt_test_results.csv \
  --model_path checkpoints/topt_best_model.pth \
  --structure_path data/topt/topt_test_structures.pt \
  --quality_path data/topt/topt_test_quality.csv \
  --esm_name facebook/esm2_t33_650M_UR50D \
  --batch_size 4
```

The script writes a CSV containing `pred_topt` and generates a scatter plot with R², RMSE, and MAE.

---

## 15. Tm extension

The same TGC-Net architecture is independently retrained for protein melting-temperature prediction.

Example training command:

```bash
python tm-train.py \
  --train_path data/tm/tm_train.csv \
  --test_path data/tm/tm_test.csv \
  --structure_path data/tm/tm_structures.pt \
  --test_structure_path data/tm/tm_structures.pt \
  --quality_path data/tm/tm_quality.csv \
  --plddt_threshold 0.7 \
  --task tm \
  --target_min 0 \
  --target_max 100 \
  --lr 1e-4 \
  --batch_size 4 \
  --epochs 60
```

Example testing command:

```bash
python tm-test.py \
  --input data/tm/tm_test.csv \
  --output results/tm_test_results.csv \
  --model_path checkpoints/tm_best_model.pth \
  --structure_path data/tm/tm_structures.pt \
  --quality_path data/tm/tm_quality.csv \
  --batch_size 4
```

Reported independent-test performance:

```text
R²   = 0.827
RMSE = 5.27 °C
MAE  = 3.87 °C
```

---

## 16. Optimal-pH extension

The same TGC-Net architecture is independently retrained for optimal-pH prediction.

Example training command:

```bash
python ph-train.py \
  --train_path data/ph/ph_train.csv \
  --test_path data/ph/ph_test.csv \
  --structure_path data/ph/ph_structures.pt \
  --test_structure_path data/ph/ph_structures.pt \
  --quality_path data/ph/ph_quality.csv \
  --plddt_threshold 0.7 \
  --task pHopt \
  --target_min 0 \
  --target_max 14 \
  --lr 1e-4 \
  --batch_size 4 \
  --epochs 60
```

Example testing command:

```bash
python ph-test.py \
  --input data/ph/ph_test.csv \
  --output results/ph_test_results.csv \
  --model_path checkpoints/ph_best_model.pth \
  --structure_path data/ph/ph_structures.pt \
  --quality_path data/ph/ph_quality.csv \
  --batch_size 4
```

Reported independent-test performance:

```text
R²   = 0.530
RMSE = 0.79
MAE  = 0.55
```

Because Topt, Tm, and optimal pH use different datasets, target ranges, and biochemical endpoints, their metric values should not be interpreted as direct comparisons of task difficulty.

---

## 17. Main Topt benchmark results

### 17.1 Fixed held-out test set

On the fixed 291-sample held-out Topt test set, the reported performance is:

| Model | Input information | R² ↑ | RMSE (°C) ↓ | MAE (°C) ↓ |
| --- | --- | ---: | ---: | ---: |
| **TGC-Net** | Sequence + predicted structure | **0.624** | **11.42** | **8.23** |
| Seq2Topt | Sequence | 0.567 | 12.26 | 8.89 |
| TOMER | Sequence + OGT | 0.552 | 12.46 | 8.74 |
| DeepET | Sequence | 0.492 | 13.28 | 9.92 |
| Preoptem | Sequence | 0.332 | 15.22 | 11.15 |

Approximately 70% of the held-out samples had a TGC-Net absolute error within 10 °C. PatchET and ETopt are not included in this table because their reported results use separately curated and partitioned benchmarks and are therefore not directly comparable with the fixed Seq2Topt test split.

The `data/compare/` directory contains the prediction/result CSV files used for the fixed-test comparison.

### 17.2 Stability under test-subset composition

The manuscript additionally reports 100 matched resampling analyses. Each iteration sampled approximately 80% of the 291 held-out proteins without replacement and evaluated all models on the same subset.

| Model | R² | RMSE (°C) | MAE (°C) |
| --- | ---: | ---: | ---: |
| **TGC-Net** | **0.630 ± 0.024** | **11.33 ± 0.33** | **8.18 ± 0.24** |
| Seq2Topt | 0.569 ± 0.025 | 12.22 ± 0.34 | 8.87 ± 0.24 |
| TOMER | 0.558 ± 0.037 | 12.37 ± 0.44 | 8.68 ± 0.25 |
| DeepET | 0.496 ± 0.037 | 13.22 ± 0.41 | 9.89 ± 0.26 |
| Preoptem | 0.336 ± 0.024 | 15.18 ± 0.37 | 11.12 ± 0.29 |

TGC-Net had a higher R² than each baseline in all 100 matched subsets. Mean ΔR² values were 0.060 against Seq2Topt, 0.072 against TOMER, 0.134 against DeepET, and 0.294 against Preoptem. Paired Wilcoxon signed-rank tests with Holm–Bonferroni correction gave an adjusted `p = 7.79 × 10⁻18` for each comparison. These analyses assess stability of the relative ranking under changes in test-subset composition; they do not constitute evaluation on 100 independent external datasets.

### 17.3 Ablation and replacement analyses

The four model levels were tested by removing a component or replacing it with a simpler alternative while retaining the same evaluation setting.

| Variant | Functional level | R² ↑ | RMSE (°C) ↓ | MAE (°C) ↓ | ΔR² |
| --- | --- | ---: | ---: | ---: | ---: |
| **Full TGC-Net** | — | **0.624** | **11.42** | **8.23** | — |
| w/o Multi-Scale Motif CNN | Sequence enhancement | 0.610 | 11.63 | 8.72 | −0.014 |
| Adaptive gate → concatenation | Sequence fusion | 0.594 | 11.87 | 8.68 | −0.030 |
| Geometric GNN → GCN | Structure encoder | 0.588 | 11.96 | 9.03 | −0.036 |
| Geometric GNN → GAT | Structure encoder | 0.600 | 11.78 | 8.49 | −0.024 |
| Geometric GNN → 1D CNN | Structure encoder | 0.597 | 11.83 | 8.61 | −0.027 |
| TGC-Fusion → cross-attention | Cross-modal fusion | 0.584 | 12.01 | 8.73 | −0.040 |
| TGC-Fusion → add-based MHA | Cross-modal fusion | 0.591 | 11.91 | 8.58 | −0.033 |
| TGC-Fusion → add + MLP | Cross-modal fusion | 0.589 | 11.94 | 8.65 | −0.035 |
| L-RAP → mean pooling | Output pooling | 0.583 | 12.03 | 8.88 | −0.041 |
| w/o RDBlock | Regression head | 0.607 | 11.67 | 8.45 | −0.017 |

All tested variants underperformed the complete model on the three reported metrics. The largest R² decreases occurred when TGC-Fusion was replaced by standard cross-attention and when L-RAP was replaced by mean pooling. These results support the contribution of coordinated context selection/incorporation and selective protein-level aggregation within the complete architecture; they do not isolate the distance bias and discrepancy gate as independent causal effects.

---

## 18. Residue-level interpretability

`interpretability.py` evaluates residue-level attribution and prediction sensitivity.

For each valid residue, the script combines:

```text
L-RAP attention weight
        ×
L2 norm of the gated fused feature
```

and normalizes the resulting importance values within each protein.

The default perturbation analysis masks:

```text
top-importance residues : 10%
low-importance residues : 10%
random residues         : 10%
random repeats          : 5
```

Basic usage:

```bash
python interpretability.py \
  --input data/topt/topt_test.csv \
  --model_path checkpoints/topt_best_model.pth \
  --structure_path data/topt/topt_test_structures.pt \
  --quality_path data/topt/topt_test_quality.csv \
  --outdir data/explanation \
  --esm_name facebook/esm2_t33_650M_UR50D \
  --task topt \
  --batch_size 4 \
  --top_ratio 0.10 \
  --mask_ratio 0.10 \
  --random_repeats 5 \
  --fill_value zero \
  --seed 42
```

Core tabular outputs include:

```text
residue_importance_long.csv
sample_summary.csv
perturbation_results.csv
perturbation_summary.csv
```

The script also generates residue-importance profiles, a structure-coordinate projection, perturbation plots, and a combined interpretability figure in PNG/PDF/SVG formats.

In the manuscript analysis, masking the 10% least-important residues produced mean and median absolute prediction changes of 0.00 °C. Randomly masking 10% of residues produced mean and median changes of 0.33 and 0.01 °C, whereas masking the 10% highest-importance residues increased them to 4.32 and 3.91 °C. The high-importance perturbation exceeded the corresponding mean random perturbation effect in 275 of 291 held-out proteins (94.5%).

The O30012 case study maps the top 10% of model-ranked residues onto the predicted structure. The selected positions include both spatially clustered and dispersed residues, illustrating how the analysis can generate hypotheses for follow-up annotation or experiments without assigning a biological mechanism from model weights alone.

The attribution analysis characterizes **model sensitivity and internal attribution**; it should not be interpreted as direct evidence of biological causality for individual residues.

---

## 19. Temporal external validation on wild-type enzymes

`wild-type.py` applies a trained Topt checkpoint to an external set of wild-type enzymes with precomputed structural coordinates.

Example:

```bash
python wild-type.py \
  --input "data/wild-type enzymes/wild-type_test.csv" \
  --output "results/wild-type_predictions.csv" \
  --model_path checkpoints/topt_best_model.pth \
  --structure_path "data/wild-type enzymes/wild-type_structures.pt" \
  --esm_name facebook/esm2_t33_650M_UR50D \
  --batch_size 4
```

For the nine temporally external wild-type enzymes reported in 2025–2026, TGC-Net achieved:

```text
mean absolute error   = 7.02 °C
median absolute error = 4.55 °C
lowest error among the compared models = 7 / 9 samples
```

Five samples had errors below 5 °C and six had errors no greater than 10 °C. Clear underestimation remained for two high-temperature enzymes, so these results are intended as limited external evidence for approximate temperature-range estimation and candidate prioritization rather than as a substitute for biochemical measurement.

---

## 20. WT–mutant analysis

`WT-mutant.py` applies the trained Topt model to WT and mutant proteins using the same sequence–structure inference pipeline.

Example:

```bash
python WT-mutant.py \
  --input data/WT-mutant/WT-mutant_test.csv \
  --output results/WT-mutant_predictions.csv \
  --model_path checkpoints/topt_best_model.pth \
  --structure_path data/WT-mutant/WT-mutant_structures.pt \
  --esm_name facebook/esm2_t33_650M_UR50D \
  --batch_size 4
```

The case-study analysis shows that the current model can provide useful absolute Topt estimates for some WT enzymes but strongly compresses mutation-induced ΔTopt values toward 0 °C. It should therefore not be treated as a dedicated mutation-effect predictor.

---

## 21. β-Agarase case study

The β-agarase directory contains the 32-enzyme case-study data, structural features, quality report, and TGC-Net predictions.

The current `β-agarase.py` script is a visualization/evaluation utility for a CSV that already contains:

```text
uniprot_id
topt
pred_topt
```

Run:

```bash
python β-agarase.py \
  --input data/β-agarase/beta_agarase_result.csv \
  --output results/beta_agarase.png \
  --id_col uniprot_id \
  --true_col topt \
  --pred_col pred_topt
```

The script saves both PNG and SVG figures and reports RMSE and MAE.

Reported case-study results:

```text
n              = 32 β-agarases
RMSE           = 7.19 °C
MAE            = 5.00 °C
Top-10 overlap = 70%
Jaccard index  = 0.538
```

The overlap corresponds to 7 shared enzymes between the experimental and predicted top-10 lists. Seq2Topt previously reported a 50% top-10 overlap in a comparable β-agarase screening analysis. The present case study supports ranking-based prescreening, while the remaining continuous-value errors still require experimental follow-up.

---

## 22. Baseline-comparison files

The `data/compare/` directory contains prediction outputs for the fixed held-out Topt test set:

```text
result_TGC.csv
result_Seq2Topt.csv
result_tomer.csv
result_deepet.csv
result_preoptem.csv
test.csv
```

These files support the model-level comparisons reported for R², RMSE, MAE, and absolute-error distributions.

---

## 23. Software environment

The scripts rely primarily on the following packages:

```text
torch
transformers
peft
numpy
pandas
scikit-learn
biopython
matplotlib
seaborn
tqdm
requests
```

The supplied scripts do not pin exact package versions. A CUDA-enabled PyTorch environment is strongly recommended for ESM-2 fine-tuning and ESMFold structure generation.

Example environment setup:

```bash
conda create -n tgcnet python=3.10
conda activate tgcnet

pip install \
  torch \
  transformers \
  peft \
  numpy \
  pandas \
  scikit-learn \
  biopython \
  matplotlib \
  seaborn \
  tqdm \
  requests
```

Install the PyTorch build appropriate for the local CUDA version when GPU acceleration is required.

The experiments reported in the manuscript were performed on a single NVIDIA RTX 4080 GPU.

---

## 24. Reproducibility notes

The following rules are important when reproducing the released workflow:

```text
The fixed held-out test set is not used for checkpoint selection.

The fixed held-out test set is not used for hyperparameter tuning.

The main Avg_pLDDT threshold is selected using training/validation data.

Training/validation samples are filtered by structural availability and the selected Avg_pLDDT threshold.

Fixed held-out test samples are filtered only by structural availability.

Avg_pLDDT is not an input feature of TGC-Net.

The 15 Å hard cutoff is used by the local Geometric GNN, not as a hard mask in TGC-Fusion.

TGC-Fusion uses the continuous pairwise distance matrix as an additive attention bias.

Coordinate perturbation (σ = 0.05 Å) is applied only to valid coordinates during training.

The best checkpoint is selected exclusively by validation RMSE.

The fixed held-out test set is evaluated after model selection.

Topt uses weighted random resampling; Tm and optimal pH use equal sampling weights.
```

The 90/10 training/validation split is produced after shuffling the retained development samples with `random_state=42`.

---

## 25. Scope and limitations

TGC-Net relies on predicted three-dimensional structures and therefore remains sensitive to structural-model quality.

Avg_pLDDT is used as a protein-level quality-control statistic during training-data selection, but the current model does not use pLDDT directly inside the Geometric GNN or TGC-Fusion.

In the fixed Topt test set, Avg_pLDDT was not significantly correlated with absolute prediction error, indicating that mean structural confidence should not be interpreted as a per-sample prediction-confidence score.

The manuscript reports Spearman ρ = −0.022 (`p = 0.714`) and Pearson r ≈ −0.023 (`p = 0.698`) across the 291 held-out samples. Avg_pLDDT is therefore used as a development-data quality-control criterion, not as a claimed estimator of prediction reliability.

Predictions for structurally uncertain proteins should therefore be interpreted together with sequence evidence and, where possible, validated experimentally.

Additional boundaries described in the manuscript are:

* Reported Topt values depend on substrate, pH, buffer, assay duration, enzyme concentration, and other measurement conditions that are not provided to the model.
* The current structural representation is based mainly on Cα coordinates and inter-residue distances; it does not explicitly model side-chain orientation, dihedral angles, hydrogen bonds, salt bridges, solvent exposure, or conformational dynamics.
* The nine-enzyme temporal validation set is small, and broader generalization across enzyme families and temperature ranges remains to be assessed.
* Training for absolute Topt prediction does not make the model a dedicated predictor of mutation-induced ΔTopt.
* Residue-level attention and perturbation results indicate model sensitivity, not experimentally established residue function.

---

## 26. Model checkpoints

Training checkpoints contain the sequence-backbone state and TGC-Net state in the same file:

```text
plm_model_state_dict
model_state_dict
best_dev_rmse
best_dev_r2
best_dev_mae
epoch
selection_metric
```

The training scripts additionally support resume checkpoints containing optimizer, scheduler, gradient-scaler, and training-history states.

If pretrained checkpoints are distributed separately, place them in a local directory such as:

```text
checkpoints/
├── topt_best_model.pth
├── tm_best_model.pth
└── ph_best_model.pth
```

and pass the corresponding path through `--model_path` during testing or case-study inference.

---

## 27. Data and output availability

Source code and data supporting the manuscript are available at [https://github.com/Jing-0313/TGC-NET](https://github.com/Jing-0313/TGC-NET).

The project files cover:

```text
source code
Topt training and testing data
Tm extension data
optimal-pH extension data
predicted structural coordinates
Avg_pLDDT quality reports
baseline-comparison outputs
residue-level interpretability outputs
wild-type temporal external-validation data
WT–mutant case-study data
β-agarase case-study data
```

Large `.pt` structural files and model checkpoints may be distributed separately from the normal Git history if required by repository size limits.

---

## 28. Citation

If you use TGC-Net, the released datasets, or the associated evaluation workflow, please cite:

```text
Xu J, Liu Q, Zhu L, Wang Y, Chang S, Yang S.
TGC-Net: Residue-Level Fusion of Multi-Scale Sequence Motifs and Structural Microenvironments for Enzyme Optimal Temperature Prediction.
```

The final journal citation and DOI can be added after publication.

---

## 29. Contact

For questions regarding TGC-Net, its datasets, or reproducibility, please open an issue in this repository.

Corresponding authors:

```text
Lun Zhu
zl@cczu.edu.cn

Yan Wang
wy6868@jlu.edu.cn

Shan Chang
schang@jsut.edu.cn

Sen Yang
ys@cczu.edu.cn
```
