# Voice Recognition — Speaker Identification from MFCC Features

> Academic project — **INSEA** · *Statistiques Multivariées* · 2024

A multivariate statistical study on speaker identification from audio recordings, combining hypothesis testing, dimensionality reduction, and supervised classification on MFCC (Mel Frequency Cepstral Coefficients) features.

---

## Project overview

Working from audio recordings of 10 American speakers (5 men, 5 women), this project explores:

1. **MFCC feature extraction** from raw audio (via `librosa`)
2. **PCA & t-SNE** for visualization
3. **Hotelling's T² test** to evaluate group differences (gender, sub-groups)
4. **Custom Mahalanobis-based classifier** built with NumPy
5. **Supervised classification** of speakers (GNB, LDA, QDA, GMM, MLP, SVM, KNN, Random Forest)
6. **Likelihood-Ratio Test** for GMM model comparison
7. **Bonus** — adding personal voice recordings to test generalization to a new speaker

---

## Key results

### Part I — Statistical analysis (gender)

**Hotelling's T² test — Men vs. Women (10 MFCC features):**

| Statistic | Value |
|---|---:|
| T² | 2 535.42 |
| F-statistic | 248.96 |
| p-value | ~1.1×10⁻¹⁶ |

**Highly significant difference** between male and female recordings.

**Custom Mahalanobis-distance classifier (NumPy only) for gender prediction:**
- **Accuracy: 100 %** 

**Hotelling's T² test — Speaker 4 (even vs. odd recordings):**

| Statistic | Value |
|---|---:|
| T² | 5.03 |
| p-value | 0.93 |

**No significant difference** — the suspected sub-group hypothesis is rejected.

### Part II — Speaker identification

#### Baseline models

| Model | Train Accuracy | Test Accuracy |
|---|---:|---:|
| Gaussian Naive Bayes | 84.00 % | 72.80 % |
| LDA | 91.73 % | 85.60 % |
| **QDA** | **98.40 %** | **88.80 %** |

QDA outperforms LDA and GNB, suggesting non-linear class boundaries with class-specific covariance.

#### Gaussian Mixture Models (one GMM per speaker)

| Model | Train Accuracy | Test Accuracy |
|---|---:|---:|
| GMM (per-speaker, `n_components=1`) | 98.40 % | 88.00 % |

GMM combinations of `n_components ∈ {1..5}` × `covariance_type ∈ {full, tied, diag, spherical}` were compared using a **Likelihood-Ratio Test** — the simplest plausible model retained is `(n_components=1, covariance='tied')`.

#### Extended model comparison

| Model | Test 25 % | Test 50 % | Test 70 % |
|---|---:|---:|---:|
| QDA | 88.80 % | 80.00 % | 67.43 % |
| LDA | 85.60 % | 86.00 % | 82.57 % |
| **MLP Classifier** | 88.80 % | 82.00 % | **83.14 %** |
| SVM | 84.80 % | 81.60 % | 81.43 % |
| Naive Bayes | 72.80 % | 76.80 % | 75.43 % |
| KNN | 76.80 % | 77.60 % | 76.29 % |
| Random Forest | 76.00 % | 76.00 % | 68.29 % |

 While QDA wins on small test sets, **MLP and LDA generalize better** as the test set grows.

#### Cross-validation (final model selection)

| Model | Mean Score | Std | Stability Score |
|---|---:|---:|---:|
| **MLP Classifier** | **87.80 %** | **0.0104** | **86.90 %** |
| QDA | 87.80 % | 0.0285 | 85.37 % |
| LDA | 87.00 % | 0.0118 | 85.98 % |
| SVM | 86.20 % | 0.0207 | 84.45 % |

**Final model: MLP Classifier** — best balance between accuracy and stability across folds.

### Bonus — Adding personal voice recordings

After enriching the dataset with personal recordings (new `SpeakerID = 10`), the MLP classifier was retrained and reached **90.58 %** accuracy, demonstrating **robustness to a new speaker class**.

---

## Key takeaways

- MFCC features cleanly separate **gender** (T² test highly significant, 100 % classification with Mahalanobis distance)
- Among classical models, **QDA captures class structure best** on smaller test sets
- **MLP shows the best generalization** and cross-validation stability
- The framework **generalizes to new speakers** without retraining from scratch

---

## Author

**Houda MOURADI** 

*Project completed in the context of the Statistiques Multivariées course.*
