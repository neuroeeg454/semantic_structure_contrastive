# semantic_structure_contrastive
In this work, we explore the alignment of EEG and text representations in a shared embedding space using a dual-tower contrastive learning framework. We generate word-aligned EEG and text embeddings for two Chinese EEG datasets and perform cross-modal retrieval, where EEG segments are mapped to their corresponding semantic representations (words or sentences) within a continuous neural embedding space. Under the semantic-aware evaluation framework, the model achieves more semantically appropriate neural-to-text retrieval.This outcome not only clarifies the role of evaluation protocols in assessing EEG-text alignment but also lays the foundation for future EEG-text semantic alignment in Chinese.
![figure]()  
## Environment
```
conda create --name semantic_contrastive python=3.10.18
```
[Requirements](https://github.com/neuroeeg454/semantic_structure_contrastive/blob/main/requirements.txt) installation:
```
pip install -r requirements.txt
```
## Dataset layout
Place your aligned CSVs and EEG HDF5 files like:
```
data/
  aligned/
    sub-01/ses-01/task-reading/run-01/*.csv
    sub-01/ses-01/task-reading/run-02/*.csv
  eeg/
    sub-01/ses-01/eeg/sub-01_ses-01_task-reading_run-01_eeg.h5
    ...
```
Each CSV should contain aligned info used by `AlignedEEGTextDataset` (see `chineseeeg2.py`): `segment`, `timestart`, `duration`, `word_alignment` or `sentence_chars` fields as applicable.

## Run training
From repo root:
```
python train_alignment_system.py --config config/config_example.yaml
```
