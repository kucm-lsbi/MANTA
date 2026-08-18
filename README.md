# MANTA

<img src="asset/manta.png" width="150" align="right" alt="manta logo">

**MANTA** is a sequence-to-ensemble framework for generating Cα-level conformational ensembles of intrinsically disordered proteins (IDPs).
Given an amino-acid sequence, MANTA extracts ESM-2 representations, predicts sequence-derived geometric priors, and realizes multiple conformations using a confidence-weighted graph-based decoder with SMACOF optimization.


---

## Installation

Clone the repository:

```bash
git clone https://github.com/kucm-lsbi/MANTA
cd MANTA
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The minimal repository structure is:

```text
MANTA/
├── MANTA_generation.py
├── requirements.txt
└── weight/
    └── MANTA.pth
```

The trained encoder checkpoint must be placed at:

```text
weight/MANTA.pth
```

The pretrained ESM-2 model (`facebook/esm2_t33_650M_UR50D`) is loaded through Hugging Face Transformers.

The current implementation uses CUDA by default.

---

## Usage

### Default generation

```bash
python MANTA_generation.py \
    "AMINO_ACID_SEQUENCE" \
    output.pdb
```

By default, MANTA generates **300 conformers** using the sequence-predicted Rg and the Rg-distribution width used in the manuscript.

### Controlling the number of conformers

```bash
python MANTA_generation.py \
    "AMINO_ACID_SEQUENCE" \
    output.pdb \
    --num-frames 1000
```

### External Rg conditioning

A target mean Rg can be supplied in Å:

```bash
python MANTA_generation.py \
    "AMINO_ACID_SEQUENCE" \
    output.pdb \
    --target-rg 35.0
```

When `--target-rg` is provided, the external value replaces the sequence-predicted mean Rg while the sequence-derived pairwise geometric priors are retained.

Because extreme Rg values may force the decoder toward distorted or nonphysical conformations, MANTA displays a warning and requires confirmation before externally conditioned generation begins.

### Controlling Rg-distribution width

The width of the frame-specific Rg distribution can be changed using:

```bash
python MANTA_generation.py \
    "AMINO_ACID_SEQUENCE" \
    output.pdb \
    --rg-std-scale 150
```

The value is expressed relative to the manuscript default:

| `--rg-std-scale` | Rg-distribution width |
| ---: | :--- |
| `50` | 50% of the default width |
| `100` | Manuscript/default width |
| `150` | 150% of the default width |
| `200` | 200% of the default width |

A larger value increases the spread of compact and extended conformations around the target mean Rg.

---

## Command-line options

| Option | Description | Default |
| :--- | :--- | :--- |
| `--num-frames` | Number of conformers to generate | `300` |
| `--target-rg` | External target mean Rg in Å | Sequence-predicted |
| `--rg-std-scale` | Relative Rg-distribution width | `100` |

Additional command-line information is available with:

```bash
python MANTA_generation.py --help
```

---

## Output

MANTA writes a multi-model PDB file containing one Cα atom per residue for each generated conformer.

After generation, the script reports information including:

- output PDB path
- sequence length
- number of conformers
- target Rg source
- target mean Rg
- Rg-distribution width
- ESM-2 runtime
- encoder runtime
- decoder runtime
- end-to-end runtime

The generated structures are coarse-grained Cα-level ensemble realizations and should not be interpreted as directly sampled all-atom equilibrium conformations.

---

## Notes

- Only the 20 canonical amino acids are accepted.
- The maximum sequence length is 1,022 residues.
- External Rg conditioning should be used with physically reasonable values.
- Increasing the Rg-distribution width can produce more extreme compact or extended conformations.
- The default inference settings should be used when reproducing the standard MANTA results reported in the manuscript.
