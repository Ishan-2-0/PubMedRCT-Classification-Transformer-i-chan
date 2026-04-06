PubMed RCT Sentence Classifier: Baseline Transformer vs Optimized Transformer
This project implements two Transformer-based models for medical abstract sentence classification using the PubMed 200k RCT dataset. The goal is to classify sentences from medical research abstracts into their structural roles, compare model performance, and visualize self-attention patterns learned by the encoder.
DatasetSource: PubMed 200k RCT (Dernoncourt & Lee, 2017)
Classes: BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS
Subset used:
Training sentences: 50,000
Validation sentences: 10,000
Test sentences: Full test split

Preprocessing: Whitespace tokenization, word-to-index vocabulary mapping, sequences padded/truncated to max length 50
Approach
Baseline Transformer
Custom Transformer Encoder built in PyTorch (nn.TransformerEncoder)
Learned word embeddings + learned positional embeddings
Classification via [CLS]-style first token output → Linear layer
2 encoder layers, 2 attention heads, d_model=32
Trained with CrossEntropyLoss and Adam optimizer
Mixed precision training via torch.amp.autocast

Optimized Transformer
Same architecture as baseline
Added LayerNorm on input embeddings (stabilizes training)
Added Dropout (0.1) on embeddings and encoder layers
Reduces overfitting and improves generalization on medical text

Training
GPU acceleration: torch.device("cuda" if torch.cuda.is_available() else "cpu")
Batch size: 16
Epochs: 10
Optimizer: Adam (lr=1e-3)
Mixed precision: GradScaler + autocast for faster GPU training

Metrics
Validation accuracy per epoch
Training loss curves
Confusion matrix on validation set
Self-attention heatmap (last encoder layer, averaged over batch)

Key Insights
Transformer encoder learns sentence structure patterns in medical text using self-attention
Adding LayerNorm and Dropout improves generalization over the baseline
Most common misclassification: OBJECTIVE ↔ BACKGROUND (genuinely similar sentence patterns)
The [CLS]-style classification approach (using first token output) mirrors the BERT classification strategy
PubMed RCT is a real benchmark dataset used in medical NLP research — making this project directly relevant to clinical AI applications

5 th cell: Predictions
The model classifies real PubMed sentences into their structural roles:
Prediction is true: METHODS         Pred: METHODS
   "Patients were randomly assigned to receive either drug A or placebo..."

Prediction is true: RESULTS         Pred: RESULTS
   "The results showed a significant reduction in symptoms after 12 weeks..."
Prediction is true: BACKGROUND      Pred: BACKGROUND
   "Hypertension is a major risk factor for cardiovascular disease..."

Prediction is not True: OBJECTIVE       Pred: BACKGROUND
   "The aim of this study was to evaluate the efficacy of treatment..."

Results
All plots are saved in the Results/ folder:
FileDescriptionbase_model_train_loss.pngTraining loss curve — Baseline Transformeroptimized_model_train_loss.pngTraining loss curve — Optimized Transformerloss_curves.pngCombined loss curve comparisonconfusion_matrix.pngConfusion matrix — Optimized Transformerheatmap.pngSelf-attention heatmap (last encoder layer)
Folder Structure
PubMed-RCT-Classification-Transformer/
├── Data/
│   └── PubMed_200k_RCT/
│       ├── train.txt
│       ├── dev.txt
│       └── test.txt
├── Notebooks/
│   └── pubmed-transformer.ipynb
├── Results/
│   ├── base_model_train_loss.png
│   ├── optimized_model_train_loss.png
│   ├── loss_curves.png
│   ├── confusion_matrix.png
│   └── heatmap.png
└── README.md

Why This Project
Medical abstract sentences follow a structured narrative — Background → Objective → Methods → Results → Conclusions. Automating this classification helps researchers quickly navigate literature, supports systematic review tools, and demonstrates transformer-based NLP applied directly to clinical research text.