# Data Labeling Lab: Weak Supervision for Clickbait Detection with Snorkel

Hand-labeling data is slow and expensive, and it's often the bottleneck in an ML project. [Snorkel](https://snorkel.org) replaces most of that work with **programmatic supervision**: you write small heuristic functions, and Snorkel learns how far to trust each one. This lab uses Snorkel's three core abstractions on a **clickbait headline detection** task:

| Notebook | Snorkel abstraction | Question it answers |
|---|---|---|
| [`01_clickbait_labeling.ipynb`](01_clickbait_labeling.ipynb) | **Labeling functions** + `LabelModel` | Can we train a good classifier with **zero** hand-labeled training examples? |
| [`02_clickbait_augmentation.ipynb`](02_clickbait_augmentation.ipynb) | **Transformation functions** + augmentation policies | Does data augmentation help when we can only afford a few hundred labels? |
| [`03_clickbait_slicing.ipynb`](03_clickbait_slicing.ipynb) | **Slicing functions** + `SliceAwareClassifier` | Where does the weakly supervised model fail, and can we fix it? |

Run them in order: Part 1 writes `data/clickbait/weak_labels_train.csv`, which Part 3 reads.

## Dataset

**Stop Clickbait corpus** ([bhargaviparanjape/clickbait](https://github.com/bhargaviparanjape/clickbait)): 16,000 clickbait headlines (BuzzFeed, Upworthy, ViralNova, Scoopwhoop, …) and 16,000 news headlines (NYT, The Guardian, The Hindu, WikiNews), stored in `data/clickbait/`.

> Chakraborty, A., Paranjape, B., Kakarla, S., & Ganguly, N. (2016). *Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media.* ASONAM 2016.

`utils.load_clickbait_dataset()` creates fixed-seed stratified splits: **train** 29,000 (labels hidden), **dev** 200 (for developing LFs), **valid** 1,000 and **test** 2,000.

| | Reference (YouTube spam) | This lab (clickbait) |
|---|---|---|
| **Dataset** | 1,956 YouTube comments | 32,000 news headlines |
| **Labeling functions** | `check out`, `subscribe`, `my channel`, … | 12 new LFs: listicle numbers, 2nd-person address, curiosity-gap regex, news verbs, headline style, spaCy entities, TextBlob subjectivity. One weak LF is shown being rejected using the dev set. |
| **End model (Part 1)** | CountVectorizer + LogisticRegression | TF-IDF (1–2 grams) + LinearSVC, plus a supervised upper bound for comparison |
| **Augmentation TFs** | name/adjective swaps, synonyms | change listicle numbers, swap places (GPE), case-preserving synonyms, random word deletion |
| **Augmentation model** | Keras LSTM | Keras 1D-CNN (the reference LSTM stays at 50% on this data) |
| **Augmentation evaluation** | one run, same epochs | 5 seeds, **compute-matched** comparison, learning curve over 50–1,000 labels |
| **Slicing** | 5 comment slices, model trained on gold labels | 7 headline slices; **weak-label vs gold-label** model per slice; root cause traced to label quality |
| **Slice-aware model** | compared with LogisticRegression | compared with an **identical plain MLP**; hyperparameters chosen on validation, 3 seeds |

## Key results (test set, 2,000 headlines)

**Part 1: labeling**

| Approach | Accuracy |
|---|---|
| Majority vote over 12 LFs | 91.6% |
| Snorkel `LabelModel` | 93.7% |
| TF-IDF + LinearSVC on weak labels | **94.2%** |
| Same model on 29,000 gold labels (upper bound) | 97.5% |

That's about 97% of fully supervised performance without hand-labeling any training headline.

**Part 2: augmentation.** With equal epochs, augmentation *looks* helpful (87.0% → 88.3% at 250 labels). With an **equal number of gradient updates**, the gain disappears at every label budget. Augmentation's real benefit here is **lower variance across seeds** (about 30–60% smaller standard deviation from 250 labels up).

**Part 3: slicing.** Weak labels cost 3.3 points overall, but **10.2 points on headlines with quotes** and 5–6 points on Title-Case, short and person-mentioning headlines, because the weak labels themselves are noisier there. `SliceAwareClassifier` improves those slices by +1.3 to +3.4 points over an identical MLP, but doesn't reach gold-label performance. The better fix is new LFs targeted at those slices.

## Setup

Snorkel 0.9 and TensorFlow 2.15 need **Python 3.10** (they don't support 3.12+).

Then open the notebooks with the **Python 3.10 (clickbait-snorkel)** kernel. NLTK's WordNet is downloaded automatically by notebook 02. Approximate run times on an M-series Mac: Part 1 ~2 min, Part 2 ~1 min, Part 3 ~7 min.
