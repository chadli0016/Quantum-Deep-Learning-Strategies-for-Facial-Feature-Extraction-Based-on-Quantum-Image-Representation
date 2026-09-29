# Quantum Deep Learning Strategies for Facial Feature Extraction Based on Quantum Image Representation

Implementation associated with the paper by **Rayane Chadli and Suzan J. Obaiys**. The proposed methods are **Quantum Image Interpolation Processing (QIIP)**, the **Quantum Fourier Spectral Descriptor (QFSD)**, and **quantum-inspired local feature extraction**. This repository evaluates their features together with AlexNet and HOG.

## Proposed methods

### Quantum Image Interpolation Processing (QIIP)

QIIP starts with a grayscale image. Quantum Image Probability Encoding (QIPE) normalizes the pixel intensities:

```math
I_{\mathrm{QIPE}}(x,y) = \sqrt{\frac{I(x,y)}{\sum_{x,y} I(x,y)}}.
```

A nonlinear phase operator is applied in the Fourier domain:

```math
I_{\mathrm{QIIP}} = \mathcal{F}^{-1}\!\left[D(u,v)\,\mathcal{F}(I_{\mathrm{QIPE}})\right],
```

```math
D(u,v) = \exp\!\left[-i\tau\sin\!\left(10(u^2+v^2)\right)\right].
```

Here, the Fourier transforms and phase modulation create a complex-valued image representation. Its magnitude and phase are passed to QFSD.

### Quantum Fourier Spectral Descriptor (QFSD)

QFSD extracts the magnitude and phase of the QIIP representation:

```math
Q(x,y)=|I_{\mathrm{QIIP}}(x,y)|,\qquad \Phi(x,y)=\arg(I_{\mathrm{QIIP}}(x,y)).
```

Pixels are grouped by normalized distance and angle around the image center:

```math
r(x,y)=\frac{\sqrt{(x-x_c)^2+(y-y_c)^2}}{r_{\max}},\qquad
\theta(x,y)=\frac{\mathrm{atan2}(y-y_c,x-x_c)+\pi}{2\pi}.
```

For every radial and angular region, QFSD computes the mean and standard deviation of both magnitude and phase. It also appends global energy and entropy:

```math
E_Q=\sum_{x,y}Q(x,y)^2,\qquad
p(x,y)=\frac{Q(x,y)}{\sum_{x,y}Q(x,y)},\qquad
H_Q=-\sum_{x,y}p(x,y)\log p(x,y).
```

With eight radial regions and eight angular regions, the implementation produces 64 regional statistics plus energy and entropy: **66 features**.

### Quantum-inspired local feature extraction

The method encodes each local four-pixel patch in a simulated four-qubit circuit. Normalized pixel values determine Y-axis rotation angles:

```math
R_y(\theta_i)=R_y(\pi x_i).
```

CNOT gates connect adjacent qubits. Each qubit contributes a probability-difference feature:

```math
|\psi(x)\rangle=U(x)|0000\rangle,\qquad f_i=P_i(0)-P_i(1).
```

Features from the patches are concatenated. Qiskit simulates the states on a classical computer; quantum hardware is not required.

## Recognition pipeline

1. Detect and crop each face.
2. Extract AlexNet and HOG features.
3. Generate QIIP and extract QFSD features.
4. Extract quantum-inspired patch features.
5. Concatenate the four descriptors and apply PCA.
6. Train and evaluate the classifiers.

The execution script evaluates one selected four-feature combination. It does not run the paper's full combination search or QML experiments.

## Results reported in the paper

| Dataset | Feature combination | Accuracy |
| --- | --- | ---: |
| Controlled | QFSD alone | 95.24% |
| Uncontrolled | AlexNet + HOG + QFSD + quantum-inspired | 81.96% |

These are published results; a different dataset may produce different accuracies.

## Files

| File | Purpose |
| --- | --- |
| `image_processing.py` | Face cropping, QIIP, QFSD, quantum-inspired features, and dataset construction |
| `machine_learning_second.py` | PCA and classifier training |
| `execution.py` | Runs the experiment and saves the results |

Keep the three files in the same directory, using these exact filenames.

## Dataset structure

```text
faces/
  person_1/
    image_1.jpg
    image_2.jpg
  person_2/
    image_1.jpg
    image_2.jpg
```

Folder names become class labels. Each person needs enough valid images for a stratified train/test split; images without a detected face are skipped.

## Run

Use Python 3.11. Install the packages imported by the scripts:

```bash
python -m pip install numpy pandas opencv-python scipy face-recognition mediapipe qiskit torch torchvision scikit-image Pillow scikit-learn xgboost matplotlib
python execution.py
```

Enter the dataset folder when prompted. The script saves `feature_combination_classifier_results.csv` and `classifier_accuracy_barplot.png` in the working directory.

## Publication

**Chadli, R., and Obaiys, S. J.** *Quantum Deep Learning Strategies for Facial Feature Extraction Based on Quantum Image Representation*.

<!-- Documentation for QIIP, QFSD, and quantum-inspired face recognition. -->
