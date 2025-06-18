# Question 01
# 🧠 Physics-Informed Neural Network (PINN) for Solving the 2D Eikonal Equation

This repository contains a Python script that trains and compares two neural network models—one **data-only** and one **physics-informed (PINN)**—to approximate the activation time field \( T(x, y) \) in a 2D domain.

The **physics-informed model** incorporates the Eikonal equation as a soft constraint:

\[
\|\nabla T(x, y)\| \cdot V(x, y) = 1
\]

---

## 📌 Overview

- **Objective**: Approximate the scalar field \( T(x, y) \), representing activation time.
- **Models**: 
  - **Data-only neural network**: Trained using MSE loss only.
  - **Physics-Informed Neural Network (PINN)**: Trained using MSE + physics residual loss.

---

## 📋 Components

### 🔹 Data Generation

- Generates synthetic data for:
  - True activation time \( T(x, y) \)
  - Conduction velocity \( V(x, y) \)
- Uses **Latin Hypercube Sampling (LHS)** for better data coverage.

### 🔹 Neural Network

- Fully connected neural network (`EikonalNet`) with:
  - Configurable number of hidden layers and neurons
  - `Tanh` activation functions
- Computes gradients with PyTorch autograd to evaluate the Eikonal residual.

### 🔹 Training Loop

- Optimizes with **Adam optimizer**
- Loss:
  - **Data-only model**: MSE
  - **PINN**: MSE + weighted Eikonal residual

### 🔹 Evaluation & Visualization

- Generates:
  - Contour plots of ground truth and predictions
  - Error maps
  - Training loss curves
- Evaluates with **Root Mean Squared Error (RMSE)**

---

## 📊 Results

### ▶️ With 50 Training Points

| Model               | RMSE     |
|--------------------|----------|
| Data-only Model     | **0.0224** |
| Physics-Informed NN | 0.0370   |

### ▶️ With 30 Training Points

| Model               | RMSE     |
|--------------------|----------|
| Data-only Model     | **0.0540** |
| Physics-Informed NN | 0.1301   |

> 📌 **Observation**: The data-only model outperforms the PINN in this setup, especially with fewer points. This may be due to the difficulty in estimating gradients accurately with limited data.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/eikonal-pinn.git
cd eikonal-pinn
pip install -r requirements.txt

# Question 2
# Question 02  
# 🧠 Neural Ordinary Differential Equation (Neural ODE) vs Standard Neural Network for 2D Classification

This repository contains a Python script that trains and compares two neural network models—one **Standard Feedforward Neural Network** and one **Neural Ordinary Differential Equation (Neural ODE)** model—on synthetic 2D classification tasks using `scikit-learn` datasets.

---

## 📌 Overview

- **Objective**: Classify 2D data points using traditional and continuous-depth neural models.

### 🔍 Models:
- **Standard Neural Network** – A typical feedforward network with ReLU activation.
- **Neural ODE** – A continuous-depth model that learns via an ODE solver using the adjoint method for memory efficiency.

### 📊 Dataset:
- 2D synthetic data generated using `make_blobs` from `sklearn.datasets` (default).
- Options available to switch to `make_moons` or `make_circles`.

---

## 🔧 Implementation Details

- **Standardization**: Input features are standardized using `StandardScaler`.
- **Loss Function**: `CrossEntropyLoss` for multi-class classification.
- **Optimizer**: Adam with a learning rate of `0.01`.
- **Epochs**: 500 training epochs.
- **Device**: Automatically uses GPU (CUDA) if available.

---

## 🧪 Model Architectures

### 🔹 Standard Neural Network


### 🔸 Neural ODE Model


- The ODE solver learns the continuous transformation of hidden states:

  \[
  \frac{dh}{dt} = f(h(t), t)
  \]

- Implemented using `torchdiffeq.odeint_adjoint` for efficient backpropagation.

---

## 📊 Results

| Model             | Training Accuracy | Test Accuracy |
|------------------|-------------------|---------------|
| Standard NN       | 100.00%           | 100.00%       |
| Neural ODE        | 100.00%           | 100.00%       |

---

## 📈 Visualizations

- **Decision Boundaries**: Plots show how each model separates the 2D feature space.
- **Training Loss**: Loss curves illustrate convergence behavior over 500 epochs.
- **Test Accuracy Over Time**: Accuracy is tracked every 50 epochs to observe generalization.

---

## 🏁 How to Run

Install the dependencies:

```bash
pip install torch numpy matplotlib scikit-learn torchdiffeq
