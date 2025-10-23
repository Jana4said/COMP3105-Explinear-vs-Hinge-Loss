# Explinear vs Hinge Loss (COMP 3105)


A **Machine Learning optimization project** that compares the performance of **Explinear** and **Hinge** loss functions for binary classification.  
This project demonstrates how convex optimization techniques can be used to train and evaluate smooth vs. non-smooth loss functions.

---

## Features 
- Implements **Explinear** and **Hinge** loss functions from scratch  
- Uses both `scipy.optimize.minimize` and `cvxopt.solvers.qp` for optimization  
- Generates and visualizes synthetic datasets  
- Compares classification accuracy and decision boundaries  
- Includes reproducible setup through `requirements.txt`

---

## Core Concepts 
- Convex optimization in machine learning  
- Surrogate loss design  
- Margin-based classification  
- Comparison between smooth and non-smooth losses  
- Reproducible experimentation and evaluation  

---

## File Overview 
| File | Description |
|------|--------------|
| `A1codes.py` | Python implementation of Explinear and Hinge loss models |
| `A1report.pdf` | Report with experiment details and results |
| `requirements.txt` | Required dependencies for execution |

---

## Technologies Used 
- **Python 3.10+**  
- **NumPy**  
- **Matplotlib**  
- **CVXOPT**  
- **Pandas**

---


**Gana Said**   
Bachelor of Computer Science (AI Stream) – Carleton University  
📧 [janasaid@cmail.carleton.ca](mailto:janasaid@cmail.carleton.ca)

---

## How It Works
1. Generates a synthetic dataset for binary classification.  
2. Trains two models — one using **Explinear Loss**, one using **Hinge Loss**.  
3. Solves the optimization problem using **CVXOPT** and **SciPy**.  
4. Plots decision boundaries and compares accuracy.  
5. Summarizes all results and findings in the report.

---

 *This project demonstrates understanding of convex optimization, loss design, and the mathematical foundations of binary classification.* 
