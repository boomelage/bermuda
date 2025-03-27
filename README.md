# 🧮 Binomial Tree Specification for Bermudan Option Pricing

## 📘 Overview

This document specifies the full model, mathematics, and algorithmic structure for pricing **Bermudan options** using a **binomial tree**. A Bermudan option is exercisable at **discrete time points**, unlike European (only at maturity) or American (any time).

---

## 📐 Mathematical Model

### Parameters

| Symbol | Description |
|--------|-------------|
| $ S_0 $ | Initial asset price |
| $ K $   | Strike price |
| $ T $   | Time to maturity |
| $ r $   | Risk-free interest rate |
| $ \sigma $ | Volatility |
| $ n $   | Number of time steps |
| $ \Delta t = T/n $ | Time step size |
| $ u = e^{\sigma \sqrt{\Delta t}} $ | Up move factor |
| $ d = 1/u $ | Down move factor |
| $ p = \frac{e^{r \Delta t} - d}{u - d} $ | Risk-neutral probability |
| $ \mathcal{E} \subset \{0, 1, ..., n\} $ | Set of exercise times |

---

## 🔣 Closed Form Binomial Tree Algorithm

### Terminal Payoffs

At maturity $ t = T $, compute:

$$
V_i^n = \max(\phi(S_i^n), 0)
$$

Where $\phi(S)$is the option's intrinsic value:
- Call: $ \phi(S) = S - K $
- Put:  $ \phi(S) = K - S $

$$
S_i^n = S_0 \cdot u^i \cdot d^{n-i}
$$

---

### Backward Induction

For each time step $t = n-1$ down to $t = 0$, and each node $i \in \{0, 1, \ldots, t\}$:

$$
V_i^t = 
\begin{cases}
\max\left( \phi(S_i^t), e^{-r \Delta t} [p V_{i+1}^{t+1} + (1-p)V_i^{t+1}] \right), & \text{if } t \in \mathcal{E} \\
e^{-r \Delta t} [p V_{i+1}^{t+1} + (1-p)V_i^{t+1}], & \text{otherwise}
\end{cases}
$$

---

## 🏗️ Algorithm Steps

1. **Initialize tree parameters**: $ u, d, p, \Delta t $
2. **Build asset price tree** $ S_i^t $
3. **Initialize payoffs at $ t = n $**
4. **Iterate backward**:
   - Compute continuation value
   - If $ t \in \mathcal{E} $, check for early exercise
5. **Return** $ V_0^0 $ as the present option price

---

## 🔁 Exercise Schedule

For Bermudan options, early exercise is allowed **only at discrete times**.

Example:
exercise_times = $set([5, 10, 15])$ (i.e., steps where exercise is allowed)

---
## Example code

```python
import numpy as np

def bermudan_binomial_tree(
    S0, K, T, r, sigma, n, option_type="call", exercise_steps=set()
):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Initialize asset prices and option values
    S = np.zeros((n+1, n+1))
    V = np.zeros((n+1, n+1))
    
    for i in range(n+1):
        S[i, n] = S0 * (u ** i) * (d ** (n - i))
        V[i, n] = max(S[i, n] - K, 0) if option_type == "call" else max(K - S[i, n], 0)

    # Backward induction
    for t in reversed(range(n)):
        for i in range(t+1):
            S[i, t] = S0 * (u ** i) * (d ** (t - i))
            cont = discount * (p * V[i+1, t+1] + (1 - p) * V[i, t+1])
            exer = max(S[i, t] - K, 0) if option_type == "call" else max(K - S[i, t], 0)
            V[i, t] = max(exer, cont) if t in exercise_steps else cont

    return V[0, 0]
```
---

## 📉 Greek Extensions

You can compute **Greeks** using tree differentials:

- **Delta**: $ \Delta = \frac{V_{1}^{1} - V_{0}^{1}}{S_0 u - S_0 d} $
- **Gamma**, **Theta**: via finite differences
- **Vega**: recompute tree with bump in $ \sigma $

---

## 📚 References

- Cox, Ross, and Rubinstein (1979): *Option Pricing: A Simplified Approach*
- Hull, J. C. (2018): *Options, Futures, and Other Derivatives*

---

🧠 **Designed by Itose Research** | Mathematical Abstractions for Markets and Intelligence