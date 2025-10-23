import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import generateData
from A2helpers import generateData, linearKernel, polyKernel, gaussKernel
import pandas as pd

# Question 1 (a)
def minExpLinear(X, y, lamb):
    ...
    n, d = X.shape
    
    def objective(theta):
        w = theta[:d].reshape(-1, 1)
        w0 = theta[d]
        margins = y * (X @ w + w0)

        # ExpLinear loss
        loss_terms = np.zeros_like(margins)
        mask_leq_zero = margins <= 0
        mask_gt_zero = margins > 0
        loss_terms[mask_leq_zero] = 1 - margins[mask_leq_zero]
        loss_terms[mask_gt_zero] = np.exp(-margins[mask_gt_zero])

        total_loss = np.sum(loss_terms)
        regularization = (lamb / 2) * np.sum(w ** 2)
        return total_loss + regularization

    theta0 = np.zeros(d + 1)
    res = minimize(objective, theta0, method='BFGS')

    w = res.x[:d].reshape(-1, 1)
    w0 = res.x[d]
    return w, w0

# Question 1 (b)
def minHinge(X, y, lamb, stabilizer=1e-5):

    n, d = X.shape
    # Building P and q
    P = np.zeros((d + 1 + n, d + 1 + n))
    P[:d, :d] = lamb * np.eye(d)  
    P += stabilizer * np.eye(d + 1 + n)
    q = np.zeros((d + 1 + n, 1))
    q[d+1:d+1+n] = 1.0 

    # Building G and h 
    G1 = np.hstack([-y * X, -y, -np.eye(n)])
    h1 = -np.ones((n, 1))

    G2 = np.hstack([np.zeros((n, d+1)), -np.eye(n)])
    h2 = np.zeros((n, 1))
    G = np.vstack([G1, G2])
    h = np.vstack([h1, h2])


    #QP
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

    #Extract
    sol_vec = np.array(sol['x']).flatten()
    w = sol_vec[:d].reshape(-1, 1)
    w0 = sol_vec[d]
    
    return w, w0

# Question 1 (c)
def classify(Xtest, w, w0):
 
    scores = Xtest @ w + w0
    yhat = np.sign(scores)
    return yhat
# Question 1 (d)
def synExperimentsRegularize():

    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.0]
    gen_model_list = [1, 2, 3]

    train_acc_explinear = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    test_acc_explinear  = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    train_acc_hinge     = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    test_acc_hinge      = np.zeros((len(lamb_list), len(gen_model_list), n_runs))

    
    np.random.seed(57) #Group ID: A2-57

    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest   = generateData(n=n_test, gen_model=gen_model)

                #ExpLinear
                w, w0 = minExpLinear(Xtrain, ytrain, lamb)
                yhat_train = classify(Xtrain, w, w0)
                yhat_test  = classify(Xtest, w, w0)
                train_acc_explinear[i, j, r] = np.mean(yhat_train == ytrain)
                test_acc_explinear[i, j, r]  = np.mean(yhat_test == ytest)

                #Hinge
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                yhat_train = classify(Xtrain, w, w0)
                yhat_test  = classify(Xtest, w, w0)
                train_acc_hinge[i, j, r] = np.mean(yhat_train == ytrain)
                test_acc_hinge[i, j, r]  = np.mean(yhat_test == ytest)

    #Average
    train_acc_explinear_avg = np.mean(train_acc_explinear, axis=2)
    test_acc_explinear_avg  = np.mean(test_acc_explinear,  axis=2)
    train_acc_hinge_avg     = np.mean(train_acc_hinge,     axis=2)
    test_acc_hinge_avg      = np.mean(test_acc_hinge,      axis=2)
    #4*6 M
    train_acc = np.hstack([train_acc_explinear_avg, train_acc_hinge_avg])
    test_acc  = np.hstack([test_acc_explinear_avg,  test_acc_hinge_avg])

    return train_acc, test_acc

# Question 2(a)
def adjExpLinear(X, y, lamb, kernel_func):
    """
    Q2(a): Adjoint form of regularized Explinear loss
    """
    n, d = X.shape
    K = kernel_func(X, X)  
    def objective(theta):
        alpha = theta[:n].reshape(-1, 1)  # n * 1
        alpha0 = theta[n]                 
        margins = y * (K @ alpha + alpha0)
        
        #Explinear loss
        loss_terms = np.zeros_like(margins)
        mask_leq_zero = margins <= 0
        mask_gt_zero = margins > 0
        
        loss_terms[mask_leq_zero] = 1 - margins[mask_leq_zero]
        loss_terms[mask_gt_zero] = np.exp(-margins[mask_gt_zero])
        
        total_loss = np.sum(loss_terms)
        regularization = (lamb / 2) * (alpha.T @ K @ alpha).item()
        
        return total_loss + regularization
    
    theta0 = np.zeros(n + 1)
    res = minimize(objective, theta0, method='BFGS')
    
    a = res.x[:n].reshape(-1, 1)
    a0 = res.x[n]
    
    return a, a0
# Question 2 (b)

def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):

    n, d = X.shape
    K = kernel_func(X, X)  #n*n
    
    # Total
    total_vars = 2 * n + 1
    
    #P matrix 
    P = np.zeros((total_vars, total_vars))
    P[:n, :n] = lamb * K
    P += stabilizer * np.eye(total_vars)
    q = np.zeros((total_vars, 1))
    q[n+1:] = 1.0  
    
   
    # Building
    G1 = np.zeros((n, total_vars))
    for i in range(n):
        G1[i, :n] = -y[i] * K[i, :]  
        G1[i, n] = -y[i]            
        G1[i, n+1+i] = -1.0         
    h1 = -np.ones((n, 1))
    
    G2 = np.zeros((n, total_vars))
    G2[:, n+1:] = -np.eye(n)
    h2 = np.zeros((n, 1))
    G = np.vstack([G1, G2])
    h = np.vstack([h1, h2])
    
 
    #QP 
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    sol_vec = np.array(sol['x']).flatten()
    a = sol_vec[:n].reshape(-1, 1)   
    a0 = sol_vec[n]                  
    
    return a, a0
# Question 2 (c)
def adjClassify(Xtest, a, a0, X, kernel_func):
  
    K_test = kernel_func(Xtest, X)  # m * n
    scores = K_test @ a + a0
    yhat = np.sign(scores)
    return yhat
# Question 2 (d)
def synExperimentsKernel():

    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001  # fixed

    #Defining
    kernel_list = [
        linearKernel,
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5)
    ]
    gen_model_list = [1, 2, 3]

   
    train_acc_explinear = np.zeros((len(kernel_list), len(gen_model_list), n_runs))
    test_acc_explinear = np.zeros((len(kernel_list), len(gen_model_list), n_runs))
    train_acc_hinge = np.zeros((len(kernel_list), len(gen_model_list), n_runs))
    test_acc_hinge = np.zeros((len(kernel_list), len(gen_model_list), n_runs))

    
    np.random.seed(57)  #Group ID: A2-57

    for r in range(n_runs):
        for i, kernel_func in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                #Explinear
                try:
                    a, a0 = adjExpLinear(Xtrain, ytrain, lamb, kernel_func)
                    yhat_train = adjClassify(Xtrain, a, a0, Xtrain, kernel_func)
                    yhat_test = adjClassify(Xtest, a, a0, Xtrain, kernel_func)
                    train_acc_explinear[i, j, r] = np.mean(yhat_train == ytrain)
                    test_acc_explinear[i, j, r] = np.mean(yhat_test == ytest)
                except:
                    train_acc_explinear[i, j, r] = 0.0
                    test_acc_explinear[i, j, r] = 0.0

                #Hinge
                try:
                    a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel_func)
                    yhat_train = adjClassify(Xtrain, a, a0, Xtrain, kernel_func)
                    yhat_test = adjClassify(Xtest, a, a0, Xtrain, kernel_func)
                    train_acc_hinge[i, j, r] = np.mean(yhat_train == ytrain)
                    test_acc_hinge[i, j, r] = np.mean(yhat_test == ytest)
                except:
                    train_acc_hinge[i, j, r] = 0.0
                    test_acc_hinge[i, j, r] = 0.0

    # Average runs
    train_acc_explinear_avg = np.mean(train_acc_explinear, axis=2)
    test_acc_explinear_avg = np.mean(test_acc_explinear, axis=2)
    train_acc_hinge_avg = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge_avg = np.mean(test_acc_hinge, axis=2)

    
    train_acc = np.hstack([train_acc_explinear_avg, train_acc_hinge_avg])
    test_acc = np.hstack([test_acc_explinear_avg, test_acc_hinge_avg])

    return train_acc, test_acc
# Question 3 (a)
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n, d = X.shape
    
    K = kernel_func(X, X) 
    
  
    y_vec = y.flatten()
    K_transformed = np.outer(y_vec, y_vec) * K

    P = (1.0 / lamb) * K_transformed
    P += stabilizer * np.eye(n)  
    
    q = -np.ones((n, 1))
    

    G = np.vstack([-np.eye(n), np.eye(n)])  # 2n * n
    h = np.vstack([np.zeros((n, 1)), np.ones((n, 1))])  # 2n * 1
    

    A = y.reshape(1, -1).astype(float)
    b = np.zeros(1)
    

    #QP
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    
    a = np.array(sol['x'])  # n * 1
 
    eps = 1e-5

    sv_indices = np.where((a.flatten() > eps) & (a.flatten() < 1 - eps))[0]
    
    if len(sv_indices) > 0:
        
        best_idx = sv_indices[np.argmin(np.abs(a[sv_indices] - 0.5))]
        
        #bias
        k_i = K[best_idx, :]  
        b_value = y_vec[best_idx] - (1.0 / lamb) * np.sum(a.flatten() * y_vec * k_i)
    else:
        b_value = 0.0
    
    return a, b_value
# Question 3 (b)
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    """
    Q3(b): Predict labels using SVM dual form parameters
    """
    #kernel
    K_test = kernel_func(Xtest, X)  # m * n
    
    #scores
    scores = (1.0 / lamb) * (K_test @ (y * a)) + b
    yhat = np.sign(scores)
    
    return yhat
# Question 3 (c)
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    # Load and preprocess data
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.0  
    y = train_data[:, [0]]        
    
    y[y == 4] = -1
    y[y == 9] = 1
    
    n = X.shape[0]
    cv_acc = np.zeros((k, len(lamb_list), len(kernel_list)))
    np.random.seed(57)  # I forgot to change this one before writing the report I'm sorry (so it was 0 during testing)
    
    indices = np.random.permutation(n)
    fold_size = n // k

    for fold in range(k):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        Xtrain, ytrain = X[train_idx], y[train_idx]
        Xval, yval = X[val_idx], y[val_idx]
        
        for i, lamb in enumerate(lamb_list):
            for j, kernel_func in enumerate(kernel_list):
                try:
                    a, b_val = dualHinge(Xtrain, ytrain, lamb, kernel_func)
                    yhat = dualClassify(Xval, a, b_val, Xtrain, ytrain, lamb, kernel_func)
                    cv_acc[fold, i, j] = np.mean(yhat == yval)
                except Exception as e:
                    #debugging
                    cv_acc[fold, i, j] = 0.0
    
    #Average folds
    avg_acc = np.mean(cv_acc, axis=0)
    
    #Find the best
    best_idx = np.unravel_index(np.argmax(avg_acc), avg_acc.shape)
    best_lamb = lamb_list[best_idx[0]]
    best_kernel = kernel_list[best_idx[1]]
    
    return avg_acc, best_lamb, best_kernel
