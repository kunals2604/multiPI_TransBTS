# multiPI-TransBTS Reproduction and ET-Loss Extension

## Overview

This repository contains our **Google Colab implementation and
reproduction study of multiPI-TransBTS**, a deep-learning model for
brain tumour segmentation from multi-modal MRI.

The work is based on the paper:

> **multiPI-TransBTS: A Multi-Path Learning Framework for Brain Tumor
> Image Segmentation Based on Multi-Physical Information**\
> arXiv:2409.12167

The official implementation is available at:

-   https://github.com/JoetheReindeer/multiPI_TransBTS

Our implementation is designed to reproduce the released model as
closely as practical on a **Google Colab T4 GPU**, audit important parts
of the released implementation, evaluate the model under both the
official slice-based protocol and an unseen-patient setting, and test a
targeted modification to the **Enhancing Tumour (ET) loss**.

------------------------------------------------------------------------

## What this project does

The project has five main goals:

1.  **Audit the released implementation**
    -   Check the official train/validation/test slice lists.
    -   Check patient overlap between the official splits.
    -   Check the behaviour of the released Hausdorff-95 metric.
    -   Verify the model parameter count and model behaviour.
2.  **Reproduce the BraTS2020 data pipeline**
    -   Download BraTS2020.
    -   Use the preprocessing logic from the released implementation.
    -   Verify that the produced slices match the official slice lists.
3.  **Train the released multiPI-TransBTS model**
    -   Use the released architecture and training hyperparameters as
        the baseline.
    -   Use the pretrained PVTv2-B2 weights supplied by the upstream
        implementation.
4.  **Test a proposed ET-loss modification**
    -   Keep the architecture and training setup unchanged.
    -   Change only the ET-loss parameter:
        -   Baseline: `beta = 0.5`
        -   Proposed variant: `beta = 0.7`
    -   The purpose is to penalise missed ET pixels more strongly.
5.  **Evaluate generalisation**
    -   Evaluate on the official test slices for comparison with the
        paper.
    -   Evaluate the same trained weights on patients that were
        completely held out from training.

------------------------------------------------------------------------

# 1. Base paper and model

## multiPI-TransBTS

multiPI-TransBTS is a multi-path Transformer-based architecture for
**brain tumour segmentation** using multiple MRI modalities.

The four MRI modalities used by the model are:

-   **T1**
-   **T1ce / T1Gd**
-   **T2**
-   **FLAIR**

The model predicts tumour regions including:

-   **WT --- Whole Tumour**
-   **TC --- Tumour Core**
-   **ET --- Enhancing Tumour**

The model uses multiple PVTv2-B2 backbone branches to process the input
modalities and combines the resulting information through the multi-path
architecture.

------------------------------------------------------------------------

# 2. Why we needed a scaled Colab implementation

The original training configuration is computationally expensive.

The paper/released setup uses a large training schedule, including:

-   369 patients
-   100 epochs
-   Batch size: 12
-   Learning rate: `0.01`
-   SGD optimiser
-   Momentum: `0.9`
-   Weight decay: `1e-4`
-   Poly learning-rate schedule

The released training setup is therefore difficult to reproduce on a
free Google Colab T4 session.

### Our solution

We introduced a **scaled reproduction preset** for Colab.

The scaled experiment uses:

-   **80 patients** in the main experiment subset
-   **30 additional patients** completely held out for unseen-patient
    evaluation
-   A reduced number of epochs, automatically selected to fit the
    available runtime
-   Batch size of **12 when GPU memory permits**
-   **Mixed-precision training (AMP)** to reduce GPU memory usage and
    improve training speed
-   The same random seed for the baseline and proposed variant

The important principle is that we **did not redesign the architecture**
to make the experiment easier. The main model architecture and released
training settings were retained, while the amount of training was scaled
to fit the available hardware.

------------------------------------------------------------------------

# 3. Repository setup

The notebook clones the official repository and checks out the upstream
commit used for the reproduction.

It then applies a small reproduction patch and installs the additional
Colab dependencies.

The main dependencies include:

-   PyTorch
-   timm
-   einops
-   NumPy
-   SciPy
-   Matplotlib
-   `hausdorff`
-   `tensorboardX`
-   `SimpleITK`
-   `kagglehub`

A pretrained **PVTv2-B2** checkpoint is downloaded for the backbone.

------------------------------------------------------------------------

# 4. Reproduction patches

The notebook does not blindly modify the upstream model. It applies a
small patch to make the released implementation runnable and
reproducible in the Colab environment.

### 4.1 Data import correction

The upstream training/testing code refers to a `data` package that is
not available in the released repository in the expected form.

We change the import to use:

``` python
data2020
```

This allows the BraTS2020 dataset loader to be used correctly.

### 4.2 Model constructor correction

The released testing code passes a `num_classes` argument that does not
match the current `multiPI_TransBTS` constructor.

We use the constructor expected by the released model:

``` python
multiPI_TransBTS([0.5, 0.5])
```

### 4.3 Model output correction

The released model returns four tensors:

``` text
main output
WT output
TC output
ET output
```

The testing code was expecting fewer outputs, so the reproduction code
correctly unpacks all four.

### 4.4 Checkpoint loading

The reproduction implementation makes pretrained-weight loading explicit
and uses CPU mapping when loading the checkpoint:

``` python
torch.load(pretrained_path, map_location="cpu")
```

This makes the loading behaviour more robust across environments.

### 4.5 Device-safe fixed kernel

The curvature module contains a fixed convolution kernel.

Instead of forcing the kernel onto CUDA during construction, we register
it as a PyTorch buffer:

``` python
self.register_buffer(...)
```

This allows the kernel to follow the model's device automatically.

### 4.6 Optimiser resume behaviour

The upstream trainer references an undefined `args.ft` condition when
loading a checkpoint.

The reproduction patch removes that invalid dependency and loads the
optimiser state directly.

------------------------------------------------------------------------

# 5. Dataset and preprocessing

The implementation uses **BraTS2020** from the public Kaggle dataset
mirror:

``` text
awsaf49/brats20-dataset-training-validation
```

The notebook downloads the dataset using `kagglehub`.

The preprocessing follows the released implementation rather than
introducing a new preprocessing pipeline.

The notebook also performs an important verification step:

> For every processed patient, the set of slices produced by
> preprocessing is compared with the slices listed in the official
> train/validation/test lists.

This is important because reproducing the model is not enough; the data
supplied to the model must also correspond to the original evaluation
protocol.

------------------------------------------------------------------------

# 6. Patient selection and evaluation splits

The scaled experiment uses two groups of patients:

### Training subset

**80 patients** are used for the main experiment.

### Unseen-patient set

**30 additional patients** are kept completely outside training.

No slices from these 30 patients are used to train the model.

This gives us two evaluation perspectives:

### A. Official test protocol

The model is evaluated on the official test slices so that the results
can be compared with the original paper/released implementation.

### B. Unseen-patient evaluation

The exact same trained weights are evaluated on the 30 patients that
were never used during training.

This tests whether the model generalises to genuinely new patients.

------------------------------------------------------------------------

# 7. Important issue: slice-level versus patient-level evaluation

MRI volumes contain many 2D slices belonging to the same patient.

The official split is based on **slices**, which means different slices
from the same patient can occur in different splits.

This creates a potential generalisation issue:

``` text
Patient A
 â”œâ”€â”€ Slice 1 â†’ training
 â”œâ”€â”€ Slice 2 â†’ training
 â”œâ”€â”€ Slice 3 â†’ test
 â””â”€â”€ Slice 4 â†’ test
```

The test slice is therefore not completely independent of the training
patient.

To investigate this, our implementation creates a separate
unseen-patient evaluation:

``` text
Training patients
 â”œâ”€â”€ Patient 1
 â”œâ”€â”€ Patient 2
 â”œâ”€â”€ ...
 â””â”€â”€ Patient 80

Unseen patients
 â”œâ”€â”€ Patient 81
 â”œâ”€â”€ ...
 â””â”€â”€ Patient 110
```

This provides a more realistic test of generalisation.

------------------------------------------------------------------------

# 8. Training configuration

The baseline uses the released model settings as the reference
configuration.

  Setting                                          Value
  --------------------------- --------------------------
  Dataset                                      BraTS2020
  Input modalities                   T1, T1ce, T2, FLAIR
  Model                                 multiPI-TransBTS
  Backbone                                      PVTv2-B2
  Batch size                           12 when supported
  Learning rate                                     0.01
  Optimiser                                          SGD
  Momentum                                           0.9
  Weight decay                                      1e-4
  LR schedule                                       Poly
  Feature-enhancement ratio                          0.5
  Random seed                                          0
  Precision                     Mixed precision in Colab
  Baseline ET beta                                   0.5
  Variant ET beta                                    0.7

The scaled run reduces the number of epochs to fit the available Colab
runtime.

------------------------------------------------------------------------

# 9. Baseline experiment

The baseline is the closest practical reproduction of the released
configuration.

The important command is conceptually:

``` bash
python -m repro.train \
    --prep_dir /content/prep \
    --out results/baseline \
    --et_beta 0.5
```

The baseline uses:

``` text
ET beta = 0.5
```

This reproduces the standard Dice-style weighting used by the released
implementation.

------------------------------------------------------------------------

# 10. Proposed modification: ET loss beta

The main proposed change in this project is a modification to the loss
used for the **Enhancing Tumour (ET)** prediction.

The baseline uses:

``` text
beta = 0.5
```

Our proposed variant uses:

``` text
beta = 0.7
```

The important point is that **this is the only intended experimental
change** between the baseline and the proposed variant.

Everything else remains the same:

-   Same architecture
-   Same dataset
-   Same patient split
-   Same seed
-   Same learning rate
-   Same optimiser
-   Same batch size
-   Same training schedule
-   Same preprocessing

This makes the experiment a controlled comparison.

------------------------------------------------------------------------

# 11. What does beta do?

The ET overlap term is written as:

``` text
ET overlap =
TP / (TP + (1 - beta) FP + beta FN)
```

where:

-   `TP` = True Positives
-   `FP` = False Positives
-   `FN` = False Negatives

A **false negative** occurs when the actual tumour is present but the
model fails to detect it.

For the baseline:

``` text
beta = 0.5
```

FP and FN receive equal weighting.

For our proposed variant:

``` text
beta = 0.7
```

the weighting becomes:

``` text
FP weight = 0.3
FN weight = 0.7
```

Therefore, the model is penalised more strongly for missing an actual ET
pixel.

### Expected behaviour

Increasing beta is expected to:

``` text
ET sensitivity â†‘
```

because the model is encouraged to miss fewer tumour pixels.

However, it can also cause:

``` text
ET precision â†“
```

because the model may predict additional pixels as tumour.

The important question is therefore whether the overall ET Dice score
improves.

------------------------------------------------------------------------

# 12. Training the proposed variant

The proposed variant is trained using the same pipeline as the baseline:

``` bash
python -m repro.train \
    --prep_dir /content/prep \
    --out results/variant_et_beta0.7 \
    --et_beta 0.7
```

The only experimental change is:

``` text
0.5 â†’ 0.7
```

for the ET loss parameter.

------------------------------------------------------------------------

# 13. Evaluation metrics

The implementation reports segmentation performance using several
metrics.

### Dice score

Dice measures the overlap between the predicted tumour and the
ground-truth tumour.

Higher is better.

``` text
1.0 = perfect overlap
0.0 = no overlap
```

### Sensitivity

Sensitivity measures how much of the actual tumour was successfully
detected.

Higher is better.

A low sensitivity means the model is missing more tumour pixels.

### Precision

Precision measures how many of the pixels predicted as tumour are
actually tumour.

Higher is better.

### HD95

HD95 measures the distance between the predicted and reference tumour
boundaries.

Lower is better.

The reproduction also checks the behaviour of the released Hausdorff
implementation against a spatial HD95 calculation.

------------------------------------------------------------------------

# 14. Reported results from the scaled experiment

In the scaled experiment described in the project report, increasing the
ET beta from `0.5` to `0.7` produced the following results on the unseen
patients:

  ET metric       Baseline Î²=0.5   Variant Î²=0.7
  ------------- ---------------- ---------------
  Sensitivity              0.637           0.779
  Precision                0.903           0.827
  Dice                     0.747       **0.802**

This shows the expected trade-off:

``` text
Sensitivity: â†‘
Precision:   â†“
Dice:        â†‘
```

The ET Dice score increased from **0.747 to 0.802** in this scaled
experiment.

The official test slices showed the same direction, with ET Dice
increasing from approximately **0.786 to 0.825**.

However, these results should not be interpreted as definitive proof
that beta `0.7` is universally optimal. The experiment is a short scaled
run using one seed, so a full training run and multiple random seeds
would be required for a stronger conclusion.

------------------------------------------------------------------------

# 15. 3D localisation / robot-target extension

In addition to segmentation, the project includes a proposed extension
that converts the tumour segmentation into a **3D target point**.

The goal is:

``` text
MRI
 â†“
Tumour segmentation
 â†“
3D tumour volume
 â†“
Tumour centroid
 â†“
Scanner coordinates
 â†“
Robot coordinates
 â†“
Robot target
```

### Step 1 --- Segment

The predicted 2D WT/TC/ET masks from the MRI slices are stacked back
into a 3D volume.

### Step 2 --- Post-process

The predicted tumour volume is cleaned by:

-   Removing small disconnected regions
-   Keeping the largest connected component
-   Filling holes

### Step 3 --- Localise

The centroid of the selected tumour region is calculated using the 3D
voxel indices:

``` text
(i, j, k)
```

The NIfTI image affine then converts these voxel coordinates into
physical scanner coordinates measured in millimetres.

### Step 4 --- Transform

A rigid transformation converts the scanner coordinates into the
simulated robot coordinate frame:

``` text
p_robot =
T_robotâ†scan Ã— A_affine Ã— [i, j, k, 1]^T
```

where:

-   `A_affine` converts voxel coordinates into scanner coordinates.
-   `T_robotâ†scan` converts scanner coordinates into the robot frame.

The robot can then use the resulting 3D point as its target.

### Important scope note

The 3D localisation/robot-target pipeline is an **extension of the
segmentation output**. It should not be confused with the core
multiPI-TransBTS architecture.

The reported robot-frame experiment uses a simulated robot coordinate
system and a known transformation, so its error primarily measures the
effect of segmentation/localisation rather than real-world robot
registration or mechanical error.

------------------------------------------------------------------------

# 16. Project limitations

This reproduction has several important limitations.

### Limited training scale

The Colab experiment uses a reduced number of patients and epochs
compared with the original paper.

Therefore, the reproduced scores should not be expected to exactly match
the paper.

### Single seed

The beta experiment uses one random seed.

A stronger study should repeat the experiment using multiple seeds.

### Short training

The scaled experiment is not equivalent to the full 100-epoch training
schedule.

The observed improvement may therefore change after full training.

### Slice-based evaluation

The released evaluation pipeline focuses on tumour-containing 2D slices.

Whole-volume 3D evaluation would provide a more complete assessment.

### Mixed precision

Mixed precision was used as a practical adaptation for the Colab T4
environment and is therefore a deviation from the original experimental
environment.

------------------------------------------------------------------------

# 17. Recommended future work

The next experiments should include:

1.  Run the complete **100-epoch training schedule** on a sufficiently
    powerful GPU.
2.  Use a strict **patient-level train/validation/test split**.
3.  Repeat the beta experiment across several random seeds.
4.  Sweep multiple beta values, for example:

``` text
Î² = 0.5
Î² = 0.6
Î² = 0.7
Î² = 0.8
```

5.  Report per-patient 3D Dice and spatial HD95.
6.  Evaluate complete MRI volumes rather than only tumour-containing
    slices.
7.  Validate the 3D localisation pipeline on real registration data
    before making claims about physical robotic targeting.

------------------------------------------------------------------------

# 18. Running the notebook

Open:

``` text
multiPI_TransBTS_colab_standalone.ipynb
```

In Google Colab:

``` text
Runtime
â†’ Change runtime type
â†’ T4 GPU
```

Then run the notebook from top to bottom.

The notebook performs:

``` text
1. Install/setup
       â†“
2. Audit released implementation
       â†“
3. Download BraTS2020
       â†“
4. Preprocess and verify slices
       â†“
5. GPU speed/memory benchmark
       â†“
6. Train baseline (Î² = 0.5)
       â†“
7. Train variant (Î² = 0.7)
       â†“
8. Evaluate official test slices
       â†“
9. Evaluate unseen patients
       â†“
10. Generate results and figures
```

------------------------------------------------------------------------

# 19. Output files

The experiment produces result folders containing information such as:

``` text
results/
â”œâ”€â”€ baseline/
â”‚   â”œâ”€â”€ results.json
â”‚   â”œâ”€â”€ checkpoint.pt
â”‚   â””â”€â”€ examples_*.npz
â”‚
â”œâ”€â”€ variant_et_beta0.7/
â”‚   â”œâ”€â”€ results.json
â”‚   â”œâ”€â”€ checkpoint.pt
â”‚   â””â”€â”€ examples_*.npz
â”‚
â”œâ”€â”€ audit.json
â”œâ”€â”€ summary.md
â””â”€â”€ figures/
    â”œâ”€â”€ dice_protocols.png
    â”œâ”€â”€ et_variant.png
    â”œâ”€â”€ training_loss.png
    â””â”€â”€ examples.png
```

The `results.json` files contain machine-readable configuration,
training history, device information, split information and evaluation
metrics.

------------------------------------------------------------------------

# 20. Reproducibility summary

The key experimental comparison is:

``` text
                    BASELINE              PROPOSED
                    --------              --------
Architecture        multiPI-TransBTS     multiPI-TransBTS
ET beta             0.5                  0.7
Dataset             BraTS2020            BraTS2020
Training patients   80                   80
Held-out patients   30                   30
Seed                0                    0
Learning rate       0.01                 0.01
Optimizer           SGD                  SGD
Batch size          12*                  12*
AMP                 Yes                  Yes
Training schedule   Same                 Same
```

`*` Batch size is reduced automatically if the available GPU cannot fit
batch 12; the actual value is recorded in the results.

The central experimental question is:

> **Does increasing the ET false-negative weight from 0.5 to 0.7 improve
> enhancing-tumour segmentation, especially on patients that were never
> seen during training?**

------------------------------------------------------------------------

## Citation

If you use the original multiPI-TransBTS model or its code, cite the
base paper and the official implementation:

**Base paper:**\
*multiPI-TransBTS: A Multi-Path Learning Framework for Brain Tumor Image
Segmentation Based on Multi-Physical Information.*\
arXiv:2409.12167

**Official implementation:**\
https://github.com/JoetheReindeer/multiPI_TransBTS

This project should be described as a **reproduction/audit with a
controlled ET-loss extension**, rather than as a replacement for the
original multiPI-TransBTS paper.
