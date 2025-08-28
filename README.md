# GeneAgent: Self-verification Language Agent for Gene Set Analysis

<p align="center" width="50%">
  <img width="80%" src="https://github.com/ncbi-nlp/GeneAgent/blob/main/workflow.geneagent.svg">
</p>

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Uoghluvm/GeneAgent-openai-1.x.git
cd GeneAgent
conda env create -f environment.yml -y
conda activate geneagent

# Configure API (edit .env file with your credentials)
cp ".env copy" .env

# Run
python main_cascade.py
```

## Workflows

GeneAgent provides three workflows for different analysis needs:

- **[English Tutorial](GeneAgent_Workflow_Tutorial.md)** | **[中文教程](GeneAgent工作流教程.md)**
- **Cascade** (`main_cascade.py`) - High-accuracy with self-verification
- **Chain-of-Thought** (`main_CoT.py`) - Fast structured reasoning
- **Summary** (`main_summary.py`) - Enrichment analysis

### Example Output

```
Process: MAPK Signaling Pathway
The proteins ERBB2, ERBB4, FGFR2, FGFR4, HRAS, and KRAS are integral components of the MAPK signaling pathway, crucial for cell growth, differentiation, and survival...
```

> **Evaluation:** Use `evaluate.ipynb` for result analysis

## Demo Website

**Live Demo:** https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/GeneAgent/

<p align="center" width="50%">
  <img width="80%" src="https://github.com/ncbi-nlp/GeneAgent/blob/main/homepage.geneagent.jpg">
</p>

## Datasets

- **Gene Ontology**: 1000 gene sets from GO:BP branch
- **MsigDB**: 56 hallmark gene sets
- **NeST**: 50 gene sets from human cancer proteomics

*Source data: [LLM Evaluation](https://github.com/idekerlab/llm_evaluation_for_gene_set_interpretation/blob/main/data/) | [Talisman](https://github.com/monarch-initiative/talisman-paper/tree/main/genesets/human)*

---

**Citation:** [DOI: 10.5281/zenodo.15008591](https://zenodo.org/records/15008591)

**Funding:** NIH National Library of Medicine Intramural Research Programs

**Disclaimer:** Research tool - not for clinical diagnosis. Consult healthcare professionals for medical decisions.
