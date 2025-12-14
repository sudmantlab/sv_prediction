# Machine learning-based prediction of human structural variation and characterization of associated sequence determinants
doi: https://doi.org/10.64898/2025.12.09.693295

## Abstract
Structural variants (SVs) represent a major source of genetic diversity and play key roles in human disease and evolution. Yet, the extent to which local sequence context shapes the likelihood of structural variant formation remains poorly quantified. Here, we develop machine learning models to predict the occurrence of SVs across the human genome and characterize genomic determinants associated with their formation. We developed both a sequence only-based convolutional neural network (CNN) model as well as a random forest approach integrating diverse genomic annotations. Both models achieve high predictive performance individually (>90% AUROC) which can be further improved in an ensemble. The predictive ability of these models demonstrates that SV-prone regions can be accurately inferred from sequence context. Model interpretability techniques reveal key genomic contributors to SVs, including effects of sequence motifs such as microhomology and non-canonical DNA structures, as well as the presence of SV hotspots. We find that different classes of SVs exhibit distinct sequence determinants, with inversions displaying particularly unique signatures. Moreover, predicted SV probability correlates with allele frequency and gene functional constraint, highlighting the utility of the model for variant effect prediction. These findings demonstrate that machine learning models trained on local sequence features can identify unstable genomic regions and provide a framework for quantifying SV susceptibility and SV variant effects in personalized genomics.

## Repository structure
`data-processing-1.py` extracts the positive and negative sets for training.
`annotation-features-2.py` annotates each sample with genomic features.
`predictions-3.ipynb` trains and tests each model (CNN, random forest, logistic regression, and ensemble), and runs further analysis to compute importances of genomic annotations and sequence features on model predictions.
`constraints-4.ipynb` runs further analysis to identify relationships between model predictions and genomic functional constraints.

## Dependencies
You can find the dependencies from the `environment.yml` file.

## Data availability
Structural variant calls were obtained from Phase 3 of the Human Genome Structural Variation Consortium (HGSVC3), using variants reported against the GRCh38 reference genome. Variant callsets included large insertions and deletions, large inversions, small indels, and SNPs. Genomic annotations were retrieved from the UCSC Genome Browser, including phyloP conservation scores, recombination rate estimates, RepeatMasker repeat annotations, candidate cis-regulatory elements (cCREs), DNase accessibility, and transcription factor ChIP–seq peak clusters. Reference genome sequences and gene annotations were obtained for GRCh38.