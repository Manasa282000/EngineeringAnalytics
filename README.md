# 🧪 EA_Assignment3

> **To run the Python file**, first install the required dependencies from `requirements.txt`, then execute the script `EA_Assignment3.py`.  
> This script includes solutions for both **Question 1** and **Question 2**.

---

## Question 1  
### 🧠 Physics-Informed Neural Network (PINN) for Solving the 2D Eikonal Equation

This project compares two neural network models—a **data-only model** and a **Physics-Informed Neural Network (PINN)**—to approximate the activation time field \( T(x, y) \) in a 2D domain.

The **PINN** model incorporates the Eikonal equation as a soft constraint:

\[
\|\nabla T(x, y)\| \cdot V(x, y) = 1
\]

---

### 📌 Overview

- **Objective**: Approximate the scalar field \( T(x, y) \), representing activation time.
- **Models**:
  - **Data-only Neural Network**: Trained using MSE loss only.
  - **Physics-Informed Neural Network (PINN)**: Trained using MSE + physics residual loss.

---

### 📋 Components

#### 🔹 Data Generation
- Synthetic generation of:
  - True activation time \( T(x, y) \)
  - Conduction velocity \( V(x, y) \)
- Uses **Latin Hypercube Sampling (LHS)** for effective coverage.

#### 🔹 Neural Network
- Fully connected architecture (`EikonalNet`) with:
  - Configurable hidden layers and neurons
  - `Tanh` activations
- Uses PyTorch autograd for Eikonal residuals.

#### 🔹 Training Loop
- Optimizer: **Adam**
- Loss Functions:
  - **Data-only**: MSE
  - **PINN**: MSE + Weighted Eikonal Residual

#### 🔹 Evaluation & Visualization
- Contour plots: Ground truth & predictions
- Error maps and loss curves
- Metric: **Root Mean Squared Error (RMSE)**

---

### 📊 Results

#### ▶️ With 50 Training Points

| Model                | RMSE     |
|---------------------|----------|
| Data-only Model      | **0.0224** |
| Physics-Informed NN  | 0.0370   |

#### ▶️ With 30 Training Points

| Model                | RMSE     |
|---------------------|----------|
| Data-only Model      | **0.0540** |
| Physics-Informed NN  | 0.1301   |

> 📌 **Observation**: The data-only model performed better, especially with fewer points, likely due to difficulties in estimating gradients under sparse data.

---

## Question 2  
### 🧠 Neural ODE vs Standard Neural Network for 2D Classification

This experiment compares a **standard feedforward neural network** with a **Neural Ordinary Differential Equation (Neural ODE)** model on synthetic 2D classification tasks.

---

### 📌 Overview

- **Objective**: Classify 2D points using discrete and continuous-depth models.
- **Models**:
  - Standard Neural Network (ReLU activations)
  - Neural ODE (ODE solver + adjoint method)

- **Datasets**:
  - Default: `make_blobs`
  - Optional: `make_moons`, `make_circles` from `sklearn.datasets`

---

### 🔧 Implementation Details

- **Standardization**: Input normalized via `StandardScaler`
- **Loss**: `CrossEntropyLoss`
- **Optimizer**: Adam, learning rate = 0.01
- **Epochs**: 500
- **Device**: GPU (CUDA) if available

---

### 🧪 Model Architectures

#### 🔹 Standard Neural Network

\[
\text{Input} \rightarrow \text{Linear} \rightarrow \text{ReLU} \rightarrow \text{Linear} \rightarrow \text{Output}
\]

#### 🔸 Neural ODE

\[
\text{Input} \rightarrow \text{Linear} \rightarrow \boxed{\text{ODE Solver}} \rightarrow \text{Linear} \rightarrow \text{Output}
\]

- ODE learns continuous transformation:


\frac{dh}{dt} = f(h(t), t)


- Efficient backpropagation via `torchdiffeq.odeint_adjoint`.

---

### 📊 Results

| Model            | Training Accuracy | Test Accuracy |
|------------------|-------------------|---------------|
| Standard NN      | 100.00%           | 100.00%       |
| Neural ODE       | 100.00%           | 100.00%       |
