# Quantum Deep Learning Strategies for Facial Feature Extraction Based on Quantum Image Representation

This repository implements the facial feature extraction framework presented by **Rayane Chadli and Suzan J. Obaiys**. Its main contributions are **Quantum Image Interpolation Processing (QIIP)**, the **Quantum Fourier Spectral Descriptor (QFSD)**, and a **quantum-inspired local feature extraction method**.

The execution script combines QFSD and quantum-inspired features with AlexNet and HOG for face recognition.

## Method

### 1. Quantum Image Interpolation Processing (QIIP)

QIIP produces a complex-valued representation of a grayscale face image. First, Quantum Image Probability Encoding (QIPE) normalizes each pixel intensity:

$$
I_{\mathrm{QIPE}}(x,y)
=
\sqrt{\frac{I(x,y)}{\sum_{x,y} I(x,y)}}.
$$

QIIP then applies a nonlinear phase operator in the Fourier domain:

$$
I_{\mathrm{QIIP}}
=
\mathcal{F}^{-1}
\left[
D(u,v)\,\mathcal{F}\left(I_{\mathrm{QIPE}}\right)
\right],
$$

where

$$
D(u,v)
=
\exp\left[-i\tau\sin\left(10(u^2+v^2)\right)\right].
$$

Here, $\mathcal{F}$ and $\mathcal{F}^{-1}$ are the forward and inverse Fourier transforms, and $\tau$ controls the phase modulation. The output is complex-valued, providing both magnitude and phase information for QFSD.

### 2. Quantum Fourier Spectral Descriptor (QFSD)

QFSD describes the QIIP representation through its magnitude and phase:

$$
Q(x,y)=\left|I_{\mathrm{QIIP}}(x,y)\right|,
\qquad
\Phi(x,y)=\arg\left(I_{\mathrm{QIIP}}(x,y)\right).
$$

Pixels are assigned to radial and angular regions around the image center $(x_c,y_c)$:

$$
r(x,y)
=
\frac{\sqrt{(x-x_c)^2+(y-y_c)^2}}{r_{\max}},
\qquad
\theta(x,y)
=
\frac{\operatorname{atan2}(y-y_c,x-x_c)+\pi}{2\pi}.
$$

For each region, QFSD calculates the **mean and standard deviation of both $Q$ and $\Phi$**. It then appends global magnitude energy and entropy:

$$
E_Q=\sum_{x,y}Q(x,y)^2,
$$

$$
p(x,y)=\frac{Q(x,y)}{\sum_{x,y}Q(x,y)},
\qquad
H_Q=-\sum_{x,y}p(x,y)\log p(x,y).
$$

The implementation uses eight radial regions and eight angular regions. Four statistics per region, plus energy and entropy, produce a **66-dimensional QFSD vector**.

### 3. Quantum-inspired local features

The second proposed feature extractor processes small grayscale image patches with **simulated four-qubit circuits**. Each normalized pixel intensity $x_i$ controls a rotation:

$$
R_y(\theta_i)=R_y(\pi x_i).
$$

CNOT gates connect neighboring qubits. The resulting state can be written as

$$
|\psi(x)\rangle=U(x)|0000\rangle.
$$

For each qubit, the descriptor records the difference between the probabilities of measuring 0 and 1:

$$
f_i=P_i(0)-P_i(1).
$$

The four values from each patch are concatenated across the face image. These circuits are simulated with Qiskit on a classical computer; **quantum hardware is not required**.

## Recognition pipeline

1. Detect and crop the face.
2. Extract AlexNet and HOG features from the cropped image.
3. Generate the QIIP representation and extract QFSD features.
4. Extract quantum-inspired features from local image patches.
5. Concatenate the four feature vectors.
6. Apply PCA to reduce dimensionality.
7. Train and evaluate the classifiers.

The execution script uses **AlexNet + HOG + QFSD + quantum-inspired features** as one selected feature set. It does not run every feature combination from the paper.

## Results reported in the paper

| Dataset | Feature set | Accuracy |
| --- | --- | ---: |
| Controlled | QFSD alone | 95.24% |
| Uncontrolled | AlexNet + HOG + QFSD + quantum-inspired | 81.96% |

These are the paper's reported results. Accuracy on a different dataset will depend on its images, subjects, and experimental conditions.

## Repository files

| File | Contents |
| --- | --- |
| `image_processing.py` | Face cropping, QIIP, QFSD, quantum-inspired features, and dataset construction |
| `machine_learning_second.py` | PCA and classifier training |
| `execution.py` | Runs the selected experiment and saves its results |

Keep these files in the same directory. The filename `image_processing.py` must match the import in `execution.py`.

## Dataset format

Place each person's images in a separate folder. Folder names become the identity labels:

```text
faces/
  person_1/
    image_1.jpg
    image_2.jpg
  person_2/
    image_1.jpg
    image_2.jpg
```

## Running the experiment

Use Python 3.11 and install the packages imported by the scripts:

```bash
python -m pip install numpy pandas opencv-python scipy face-recognition mediapipe qiskit torch torchvision scikit-image Pillow scikit-learn xgboost matplotlib
```

Run:

```bash
python execution.py
```

Enter the path to the `faces` folder when prompted. The script saves classifier accuracies to `feature_combination_classifier_results.csv` and a bar chart to `classifier_accuracy_barplot.png`.

## Publication

**Chadli, R., and Obaiys, S. J.** *Quantum Deep Learning Strategies for Facial Feature Extraction Based on Quantum Image Representation*.
