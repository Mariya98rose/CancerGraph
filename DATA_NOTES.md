# Data notes

- Source: UCSC Xena, cohort TCGA Breast Cancer (BRCA), TCGA Hub (tcga.xenahubs.net).
- Expression: dataset TCGA.BRCA.sampleMap/HiSeqV2 (version 2017-10-13),
  gene-level RNA-seq, unit log2(norm_count+1), 20,530 genes x 1,218 samples.
- Clinical: dataset TCGA.BRCA.sampleMap/BRCA_clinicalMatrix, 1,247 samples.
- Label column: PAM50Call_RNAseq (the PAM50 call provided in the Xena clinical matrix;
  the Xena page does not document exactly how it was computed).
- Cleaning: samples present in both files, primary tumours only (barcode code 01),
  labelled samples only, one sample per patient (first 12 barcode characters).
- Result: 844 samples. LumA 421, LumB 192, Basal 141, Her2 67, Normal 23.
- Split: stratified 70/15/15, seed 42 (results/split_seed42.csv). Same split for all models.
- Caveat: the Normal class is very small (23 primary tumours), so its metrics are unreliable.
  Raw data are not stored in this repo; download them from Xena.
