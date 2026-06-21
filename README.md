# Machine learning-based prediction of human structural variation and characterization of associated sequence determinants
doi: https://doi.org/10.64898/2025.12.09.693295

## Abstract
Structural variants (SVs) represent a major source of genetic diversity and play key roles in human disease and evolution. Yet, the extent to which local sequence context shapes the likelihood of structural variant formation remains poorly quantified. Here, we develop machine learning models to predict the occurrence of SVs across the human genome and characterize genomic determinants associated with their formation. We developed both a sequence only-based convolutional neural network (CNN) model as well as a random forest approach integrating diverse genomic annotations. Both models achieve high predictive performance individually (>90% AUROC) which can be further improved in an ensemble. The predictive ability of these models demonstrates that SV-prone regions can be accurately inferred from sequence context. Model interpretability techniques reveal key genomic contributors to SVs, including effects of sequence motifs such as microhomology and non-canonical DNA structures, as well as the presence of SV hotspots. We find that different classes of SVs exhibit distinct sequence determinants, with transposable elements and inversions displaying particularly unique signatures. Moreover, predicted SV probability correlates with allele frequency and gene functional constraint, highlighting the utility of the model for variant effect prediction. These findings demonstrate that machine learning models trained on local sequence features can identify unstable genomic regions and provide a framework for quantifying SV susceptibility and SV variant effects in personalized genomics.

## Repository structure
`data-processing.py` extracts the positive and negative sets for training, and computes the number of variants within each context window.     
`feature-annotation.py` annotates each sample with various curated genomic features, such as conservation scores, repeat content, and gene features.     
`model-training-predictions.ipynb` trains and tests each model (CNN, random forest, logistic regression, and ensemble), and runs further analysis to compute feature importances of genomic annotations on model predictions and representational analyses on the CNN.       
`motif-analysis.ipynb` performs analysis on the effects of various motifs on CNN predictions, such as non-canonical DNA structures, kmers of various composition, and homology.    
`in-silico-mutagenesis.ipynb` performs in-silico mutagenesis on Alu-associated SVs and identifies important motifs involved, and compares findings with empirical observations.   
`constraint-analysis.ipynb` explores relationships between model predictions and functional constraints, such as allele frequency, CADD-SV scores, pLI, and LOEUF scores.   
`hprc_benchmark.ipynb` performs a benchmark of model performance on variants called on individual genomes from the Human Pangenome Reference Consortium (HPRC).   

## Dependencies
Dependencies can be found in `environment.yml`.

## Data availability
Structural variant calls were obtained from Phase 3 of the Human Genome Structural Variation Consortium (HGSVC3), using variants reported against the GRCh38 reference genome. Variant callsets included large insertions and deletions, large inversions, small indels, and SNPs. 

HGSVC callset (for GRCh38): https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC3/release/Variant_Calls/1.0/GRCh38/

Further benchmarks for the model were made using variants reported against the CHM13 reference which can also be found in HGSVC3, and variants in individual genomes from the Human Pangenome Reference Consortium (HPRC).

HGSVC callset (for CHM13): https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC3/release/Variant_Calls/1.0/T2T-CHM13/

HPRC callsets:
https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/759B21AD-0ED8-4640-A433-7C92A57EA3D3--UW_EEE_SV_Calls/

Genomic annotations were retrieved from the UCSC Genome Browser, including phyloP conservation scores, recombination rate estimates, RepeatMasker repeat annotations, candidate cis-regulatory elements (cCREs), DNase accessibility, and transcription factor ChIP–seq peak clusters. Genome annotations were obtained for the GRCh38 reference.  

## Acknowledgements
This work was supported by NIH National Institute of General Medicine award R35GM142916 to Peter H Sudmant. This work was also supported by the UC Berkeley Summer Undergraduate Research Fellowship (SURF) to Daven Lim.
