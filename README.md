# Data and scripts for COVID-19

This GitHub repo contains data and scripts relevant to COVID-19, which is the disease caused by the virus SARS-CoV-2. For a full descriptions of our efforts, please see https://www.aicures.mit.edu/.

Note that since relatively little data for SARS-CoV-2 is available, most of the data in this repo is for SARS-CoV (responsible for the 2002/3 [SARS](https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome) outbreak) and other related coronaviruses. The hope is that models trained on this data will be able to retain their predictive ability on SARS-CoV-2.

Although the data contained in this repo can be used by any model, we have primarily been working with the message passing neural network model [chemprop](https://github.com/chemprop/chemprop). Our trained models are available on http://chemprop.csail.mit.edu/predict.

## [data/](https://github.com/yangkevin2/coronavirus_data/tree/master/data)

SARS-CoV data
- [AID1706_binarized_sars.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/AID1706_binarized_sars.csv) - (N = 290,726; hits = 405) In-vitro assay that detects inhibition of SARS-CoV 3CL protease via fluorescence from [PubChem AID1706](https://pubchem.ncbi.nlm.nih.gov/bioassay/1706).
- [evaluation_set_v2.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/evaluation_set_v2.csv) - (N = 5,671; hits = 41) An evaluation set for SARS-CoV 3CL protease containing 41 experimentally validated hits along with 5630 molecules from the [Broad Repurposing Hub](https://www.broadinstitute.org/drug-repurposing-hub) which are treated as non-hits. There is no overlap with AID1706_binarized_sars.csv.
- [AID1706_binarized_sars_full_eval_actives.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/AID1706_binarized_sars_full_eval_actives.csv) - (N = 290,767; hits = 446) is AID1706_binarized_sars.csv combined with the 41 validated hits from evaluation_set_v2.csv.
- [PLpro.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/PLpro.csv) - (N = 233,891; hits = 697) Bioassay that detects activity against SARS-CoV in yeast models via PL protease inhibition. Combines PubChem data from [AID652038](https://pubchem.ncbi.nlm.nih.gov/bioassay/652038) and [AID485353](https://pubchem.ncbi.nlm.nih.gov/bioassay/485353).

SARS-CoV-2 data
- [​mpro_xchem.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/mpro_xchem.csv) - (N = 880; hits = 78) Fragments screened for 3CL protease binding using crystallography techniques. Data is sourced from the [Diamond Light Source](https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem.html) group.

​Data extracted from literature
- [corona_literature_idex.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/corona_literature_idex.csv) - (N = 101) FDA-approved drugs that are mentioned in generic coronavirus literature. Drug to SMILES mapping is generated through the PubChem idex service and may contain multiple SMILES for generic drug names. These are not guaranteed to be effective against any targets; they simply appear in the literature.

​Catalogues of drugs that can be screened for repurposing
- [broad_repurposing_library.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/broad_repurposing_library.csv) - (N = 6,111) Compounds from the [Broad Repurposing Hub](https://www.broadinstitute.org/drug-repurposing-hub), many of which are FDA-approved.
- [external_library.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/external_library.csv) - (N = 861) A set of FDA-approved drugs.
- [expanded_external_library.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/expanded_external_library.csv) - (N = 2,661) A larger set of FDA-approved drugs, but not a strict superset of external_library.csv.

Other property prediction data
- [ecoli.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/ecoli.csv) - (N = 2,335; hits = 120) Compounds which have been screened for inhibitory activity against *E. coli*, from the paper [A Deep Learning Approach to Antibiotic Discovery](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1).

## [splits.zip](https://github.com/yangkevin2/coronavirus_data/tree/master/splits.zip)
Contains train/dev/test splits (using a scaffold split) of some of the above datasets for benchmarking purposes.

## [raw_data/](https://github.com/yangkevin2/coronavirus_data/tree/master/raw_data)
Original raw data files and format conversions. 

## [predictions/](https://github.com/yangkevin2/coronavirus_data/tree/master/predictions)
Predictions made by trained models on some of the repurposing datasets. See the README inside the predictions/ directory for details.

## [plots/](https://github.com/yangkevin2/coronavirus_data/tree/master/plots)
t-SNE plots comparing the datasets. Note that in the plots, "sars_pos" and "sars_neg" refer to any hits or non-hits, respectively, across both [AID1706_binarized_sars.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/AID1706_binarized_sars.csv) and [PLpro.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/PLpro.csv).

## [conversions/](https://github.com/yangkevin2/coronavirus_data/tree/master/conversions)
Files for converting between smiles/cid/name. Obtained from https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi

## [similarity_computations/](https://github.com/yangkevin2/coronavirus_data/tree/master/similarity_computations)
The nearest neighbor computations from each test set to the training set. 

## [scripts/](https://github.com/yangkevin2/coronavirus_data/tree/master/scripts)
Various data processing scripts for reuse/reproducibility.

## [statistics/](https://github.com/yangkevin2/coronavirus_data/tree/master/statistics)
Statistics about overlap between the SMILES strings of various datasets.

## [interpretation/](https://github.com/yangkevin2/coronavirus_data/tree/master/interpretation)

t-SNE plots of chemical rationales extracted (using this [code](https://github.com/chemprop/chemprop/blob/master/interpret.py)) from a model trained on the combined AID1706 and PLpro datasets.

## [old/](https://github.com/yangkevin2/coronavirus_data/tree/master/old)
Older versions of files from when we combined AID1706 data with other data that was unhelpful. 

## Experiment Commands

These commands are for running experiments using [chemprop](https://github.com/chemprop/chemprop) and should be run from the main directory in the chemprop repo. You may need to modify some paths depending on your directory structure. The commands below assume you are using [AID1706_binarized_sars.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/AID1706_binarized_sars.csv) but can be modified to work with any of the datasets.

### Generating RDKit features

To speed up experiments, you can pre-generate RDKit features using the script `save_features.py` in `chemprop/scripts`. You should run this command:

```
python save_features.py
    --data_path ../../coronavirus_data/data/AID1706_binarized_sars.csv \
    --save_path ../../coronavirus_data/features/AID1706_binarized_sars.npz \
    --features_generator rdkit_2d_normalized
```

By default this will run feature generation using parallel processing. On occasion the parallel processing gets stuck near the end of feature generation, so if this happens, just kill the process and restart with the `--sequential` flag. This will pick up where the parallel version stopped and will finish correctly.


### Training and testing

```
python train.py \
    --data_path ../coronavirus_data/data/AID1706_binarized_sars.csv \
    --dataset_type classification \
    --save_dir ../coronavirus_data/ckpt/AID1706_binarized_sars \
    --features_path ../coronavirus_data/features/AID1706_binarized_sars.npz \
    --no_features_scaling \
    --split_type scaffold_balanced \
    --quiet
```

The data splitting mechanism in [chemprop](https://github.com/chemprop/chemprop) is seeded so that this will reproduce the same train/dev/test split as in [splits.zip](https://github.com/yangkevin2/coronavirus_data/blob/master/splits.zip).

### Class balance

To run experiments with class balance, switch to the `class_weights` branch of chemprop (`git checkout class_weights`) and add the `--class_balance` flag. This will train with an equal number of positives and negatives in each batch.

### Multi-task training for SARS-CoV-2 3CLpro and SARS-CoV 3CLpro

Experiment combining data on the 3CLpro target for SARS-CoV-2 [​mpro_xchem.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/mpro_xchem.csv) and SARS-CoV [AID1706_binarized_sars.csv](https://github.com/yangkevin2/coronavirus_data/blob/master/data/AID1706_binarized_sars.csv).

5-fold cross validation performance is 0.850 +/- 0.022.

```
python multitask.py \
    --data_path data/mpro_xchem.csv \
    --source_data_path data/AID1706_binarized_sars.csv \
    --dataset_type classification \
    --save_dir ckpt/
```
