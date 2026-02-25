"""
    Finite Element Solver for a 1D c-example
    Author: Bochra Mejri 
    Lecture: Regularization Theory: From Functional Analysis to Machine Learning 
    Level: Master students
"""

import numpy as np
import matplotlib.pyplot as plt

def Forward(x, f, nele):
    """
        Example 1.1.1: c-example
        Solve -y"(s) + x(s) y(s) = f(s) for s in (0,1) with Dirichlet BCs y(0) = y(1) = 0
        Discretization using linear splines on a uniform grid of (nele + 1) nodes 
        Inputs: 
            x     : function - coefficient function x(s)
            f     : function - right-hand side f(s)
            nele  : int - number of elements  
        Outputs: 
            nodes : ndarray - mesh nodes in (0,1)    
            y     : ndarray - FEM solution at nodes y(s)  
    """
    # Mesh Parameters
    h = 1.0 / nele                                                 # element size
    nodes = np.linspace(0.0, 1.0, nele+1)                          # mesh points

    # Local matrices 
    Ke = (1.0/h) * np.array([[1.0, -1.0], [-1.0, 1.0]])             # local stiffness matrix 
    Me = (h/6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])               # local mass matrix 

    # Global matrices 
    K = np.zeros((nele+1, nele+1))                                  # stiffness matrix  
    M = np.zeros((nele+1, nele+1))                                  # mass matrix 
    F = np.zeros(nele+1)                                            # load vector
    
    # Assemply loop 
    for e in range(nele):
        smid = 0.5 * (nodes[e] + nodes[e+1])                        # midpoint
        K[e:e+2, e:e+2] += Ke
        M[e:e+2, e:e+2] += x(smid) * Me 
        F[e:e+2] += f(smid) * (h/2.0) * np.array([1.0, 1.0])

    # Solve system with Dirichlet BCs
    G = K + M 
    y_int = np.linalg.solve(G[1:-1, 1:-1], F[1:-1])
    y = np.hstack(([0.0], y_int, [0.0]))

    return nodes, y 

# ------- Example -------
x = lambda s: 1.0
f = lambda s: np.exp(-s)
y_true = lambda s: - (np.exp(s) - np.exp(-s)) / (2 * (np.exp(2) - 1)) + 0.5 * s * np.exp(-s)
# FEM solution
nodes, y = Forward(x, f, nele=50)
# Analytic solution
y_val = np.array([y_true(s) for s in nodes])

# ------- Plot -------
plt.plot(nodes, y, label="FEM solution")
plt.plot(nodes, y_val, '--', label="Analytic solution")
plt.xlabel("s")
plt.ylabel("y(s)")
plt.title("c-example")
plt.legend()
plt.show()


\begin{algorithm}
\caption{Finite Element Solver}
    \begin{algorithmic}[1]
        \STATE Construct uniform mesh
        \FOR{each element}
            \STATE Compute local matrices
            \STATE Assemble global system
        \ENDFOR
        \STATE Apply Dirichlet boundary conditions
        \STATE Solve linear system
        \STATE Return solution
    \end{algorithmic}
\end{algorithm}         