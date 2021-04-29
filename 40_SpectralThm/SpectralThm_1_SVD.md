---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The Singular Value Decomposition

In the previous workbook, we saw that symmetric square matrices $A$ have a special decomposition called an _eigenvalue_ decomposition: $A = V\Lambda V^\top$, where $V$ is an orthogonal matrix satisfying $V^\top V = VV^\top = I$, whose columns are _eigenvectors_, and $\Lambda = \text{diag}(\lambda_1,\dots, \lambda_n)$ is a diagonal matrix containing the _eigenvalues_ of $A$. 

In this section, we see that _all_ matrices -- even non-square and non-symmetric -- have a similar decomposition called the _singular value decomposition_, or SVD. Let's first remind ourselves why we can't eigenvalue decomposition doesn't make sense for non-square matrices. Suppose $A$ is a $m\times n$ matrix. Then for any $v\in \mathbb{R}^n$, $Av$ is a vector in $\mathbb{R}^m$, and so the eigenvalue condition $Av = \lambda v$ does not make sense in this setting: the left-hand side is a $m$-dimensional vector, while $\lambda v$ is an $n$-dimensional vector.

Instead, for $m\times n$ matrices $A$, we consider instead a generalized version of the eigenvalue condition: vectors $v\in \mathbb{R}^n$, $u\in \mathbb{R}^m$ and a number $\sigma$ are called _right and left singular vectors_, and a _singular value_ if they satisfy:


$$
\begin{aligned}
Av = \sigma u && (1)\\
A^\top u = \sigma v && (2)
\end{aligned}
$$


Singular values and singular vectors are in fact closely related to eigenvalues and eigenvectors. Let's see why this is the case. Let's start with equation $(1)$, and multiply both sides by $A^\top$:


$$
Av = \sigma u \implies A^\top A v = \sigma A^\top u.
$$


Now, let's plug in equation $(2)$, which says that $A^\top u = \sigma v$. We get:


$$
A^\top A v = \sigma A^\top u = \sigma^2 v.
$$


This looks more like something we've seen before: if we set $B = A^\top A$ and $\lambda = \sigma^2$, this can be written as $Bv = \lambda v$. Therefore, the squared singular values and right singular vectors can be obtained by computing an eigenvalue decompostion of the symmetric matrix $A^\top A$. Using a similar derivation, we can also show that


$$
AA^\top u = \sigma^2 u
$$


from which we see that $u$ is really an eigenvector of the symmetric matrix $AA^\top$.

**Remark:** In our discussion above, we saw that the eigenvalues of $AA^\top$ and/or $A^\top A$ correspond to the _squared_ singular values $\sigma^2$ of $A$. This may seem odd, since we know that in general matrices may have positive or negative eigenvalues. However, this occurs specifically because $A^\top A$ and $AA^\top$ are always _positive semi-definite_, and therefore always have non-negative eigenvalues. To see why this is true, note that the smallest eigenvalue of $A^\top A$ are the minimum of the quadratic form $Q(x) = x^\top A^\top A x$, over all unit vectors $x$. Then:


$$
\lambda_{\text{min}} = \min_{\|x\|_2 =1} x^\top A^\top A x = \min_{\|x\|_2 =1} (Ax)^\top Ax = \min_{\|x\|_2 =1}\|Ax\|_2^2 \geq 0.
$$


A similar derivation shows that all the eigenvalues of $AA^\top$ are non-negative.

How many singular values/vectors do we expect to get for a given $m\times n$ matrix $A$? We know that the matrix $A^\top A$ is $n\times n$, which gives us $n$ eigenvectors $v_1,\dots, v_n$ (corresponding to $n$ right singular vectors of $A$), and $AA^\top$ is $m\times m$, giving us $m$ eigenvectors $u_1,\dots, u_m$ (corresponding to $m$ left singular vectors of $A$). The matrices $A^\top A$ and $AA^\top$ will of course not have the same number of eigenvalues, though they do always have the same _non-zero_ eigenvalues. The number $r$ of nonzero eigenvalues of $A^\top A$ and/or $AA^\top$  is exactly equal to the _rank_ of $A$, and we always have that $r \leq \min(m,n)$. 

Now let's collect the vectors $u_1,\dots, u_m$ into an $m\times m$ matrix $U = \begin{pmatrix} u_1 & \cdots & u_m\end{pmatrix}$ and likewise with $v_1,\dots, v_n$ into the $n\times n$ matrix $V = \begin{pmatrix}v_1 &\cdots & v_n\end{pmatrix}$. Note that since $U$ and $V$ come from the eigenvalue decompositions of the symmetric matrices $AA^\top$ and $A^\top A$, we have that $U$ and $V$ are always orthogonal, satisfying $U^\top U = UU^\top = I$ and $V^\top V = VV^\top = I$. 

Then let's define the $m\times n$ matrix $\Sigma$ as follows:


$$
\Sigma_{ij} = \begin{cases}\sigma_i & \text{if } i=j\\ 0  & \text{if } i\neq j\end{cases}
$$


That is, $\Sigma$ is a "rectangular diagonal" matrix, whose diagonal entries are the singular values of $A$ -- i.e. the square roots of the eigenvalues of $A^\top A$ or $AA^\top$. For example, in the $2\times 3$ case $\Sigma$ would generically look like


$$
\begin{pmatrix}\sigma_1 & 0 & 0 \\ 0 & \sigma_2 &0\end{pmatrix}
$$

and in the $3\times 2$ case it would look like


$$
\begin{pmatrix}\sigma_1 & 0  \\ 0 & \sigma_2 \\ 0 & 0\end{pmatrix}.
$$



Given the matrices $U, \Sigma$ and $V$, we can finally write the full singular value decomposition of $A$:

$$
A = U\Sigma V^\top.
$$


This is one of the most important decompositions in linear algebra, especially as it relates to statistics, machine learning and data science.

**Remark:** Sometimes you may see a slightly different form of the SVD: the rank of $A$ is $r\leq \min(n,m)$, we can actually remove the last $m-r$ columns of $U$ and $n-r$ column of $V$ (so that $U$ is $m\times r$ and $V$ is $n\times r$), and let $\Sigma$ be the $r\times r$ diagonal matrix $\text{diag}(\sigma_1,\dots,\sigma_r)$. The two forms are totally equivalent, since the last $m-r$ columns of $U$ are only multiplied by the $m-r$ zero rows at the bottom of $\Sigma$ anyway. This form is sometimes called the "compact SVD". In this workbook, we'll assume we're working with the "standard" version, introduced above, though the compact version is sometimes better to work with in practice, especially when the matrix $A$ is very low rank, with $r\ll m,n$.

## Computing the SVD in Python

Let's see some examples of computing the singular value decomposition in Python.

First, let's draw a random $m\times n$ matrix $A$ to use.

```{code-cell}
import numpy as np
np.random.seed(1)

m = 5
n = 3

A = np.random.normal(size=(m,n))
```

Next, let's compute the eigenvalue decompositions of $A^\top A = V\Lambda_1 V^\top$ and $AA^\top = U\Lambda_2 U^\top$.

```{code-cell}
AAT = np.dot(A,A.T)
ATA = np.dot(A.T,A)

Lambda1, V = np.linalg.eig(ATA)
Lambda2, U = np.linalg.eig(AAT)
```

Of course, since $A^\top A$ and $AA^\top$ are of different dimensions, $\Lambda_1$ and $\Lambda_2$ will also be of different dimensions. However, as we mentioned above, $\Lambda_1$ and $\Lambda_2$ should have the same _non-zero_ entries. Let's check that this is true.

```{code-cell}
print(Lambda1.round(8))
print(Lambda2.round(8))
```

Indeed, we get the same non-zero eigenvalues, but $\Lambda_2$ has 10 extra zero eigenvalues. Now let's form the matrix $\Sigma$, which will be $m\times n$ matrix with $\Sigma_{ii} = \sqrt{\lambda_i}$ and $\Sigma_{ij} = 0$ for $i\neq j$. 

```{code-cell}
Sigma = np.zeros((m,n))
for i in range(n):
    Sigma[i,i] = np.sqrt(Lambda1[i])
    
Sigma
```

Now we have our matrices $V,U$ and $\Sigma$; let's check that $A = U\Sigma V^\top$.

```{code-cell}
np.allclose(A, np.dot(U, np.dot(Sigma, V.T)))
```

Strangely, this doesn't give us the correct answer. The reason is that we have an issue with one of the signs of the eigenvectors: the eigenvalue is invariant to switching the signs of one of the eigenvectors (i.e. multiplying one of the columns of $V$ or $U$ by $-1$ ), but the SVD is not. Since we computed the eigenvalue decomposition of $A^\top A$ and $AA^\top$ separately, there was no guarantee that we would get the correct signs of the eigenvectors. It turns out in this case we can fix this by switching the sign of the third column of $V$.

```{code-cell}
V[:,2] *= -1
np.allclose(A, np.dot(U, np.dot(Sigma, V.T)))
```

Now everything works! However, this issue is a bit annoying in practice -- fortunately, we can avoid it by simply using numpy's build in SVD function, `np.linalg.svd`. Let's see how this works.

```{code-cell}
U, S, VT = np.linalg.svd(A)

Sigma = np.zeros((m,n)) # make diagonal matrix
for i in range(n):
    Sigma[i,i] = S[i]

np.allclose(A, np.dot(U, np.dot(Sigma, VT)))
```

Now that we've seen how the singular value decomposition works in Python, let's explore some interesting applications.

## Approximating matrices with the SVD

While the singular value decomposition appears frequently in statistics and machine learning, one of the most important uses of the SVD is in _low-rank approximation_. 

Before explaining the problem of low-rank approximation, let's first state a few useful facts. Let's assume that $A$ is an $m\times n$ matrix with rank $r$ (i.e. $r$ non-zero singular values) and that $A = U\Sigma V^\top$ is its singular value decomposition. Also, let's label $\sigma_1,\dots, \sigma_r$ as its (non-zero) singular values, and let $u_1,\dots,u_r$ be the first $r$ columns of $U$ and $v_1,\dots, v_r$ be the first $r$ columns of $V$. Throughout, we will assume that the singular values are ordered, so that $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r$. Then $A$ can be written as


$$
A = \sum_{i=1}^r \sigma_i u_iv_i^\top.
$$
In the problem of low-rank approximation, we want to find a matrix $\widehat{A}$ which has rank at most $k$, and approximates $A$ closely, i.e. $A\approx \widehat{A}$. Formally, the problem can be written as follows:


$$
\begin{aligned}
\min_{\widehat{A}}&\;\;\;\; \|A - \widehat{A}\|_F && (3)\\
\text{subject to}&\;\;\;\; \text{rank}(\widehat{A}) \leq k
\end{aligned}
$$


Low rank matrices are useful for a variety of reasons, but two important reasons are that 1) they can require less memory to store and 2) we can do faster matrix computations with low rank matrices. It turns out that the solution to the low-rank approximation problem (3) can be exactly constructed from the singular value decomposition. The famous [Eckart–Young–Mirsky theorem](https://en.wikipedia.org/wiki/Low-rank_approximation) states that the solution to the problem (3) is explicitly given by the matrix: 


$$
\widehat{A}_k = \sum_{i=1}^k \sigma_i u_iv_i^\top.
$$


That is, the _best possible rank $k$ approximation to $A$ is given by the matrix which keeps just the top $k$ singular values of $A$_. Another way to write $\widehat{A}_k$ is as follows: set $\Sigma_k$ to be the $m\times n$ matrix such that 


$$
[\Sigma_k]_{ij} = \begin{cases}\sigma_i & \text{if } i=j \text{ and } i\leq k\\ 0 & \text{otherwise}\end{cases}
$$


That is, $\Sigma_k$ is the same as $\Sigma$, except we set the singular values $\sigma_{k+1},\dots,\sigma_r$ to be equal to zero. Then $\widehat{A}_k = U\Sigma_kV^\top$. 

When can we expect $\widehat{A}_k$ to be a good approximation to $A$? It turns out that the error $\|\widehat{A}_k - A\|_F$ is exactly given by $\sqrt{\sum_{i=k+1}^r \sigma_i^2}$. Therefore, if the matrix $A$ has many small singular values, then it can be well approximated by $\widehat{A}_k$.

In the next section, we see an example of low rank approximation with compressing an image.

### An example with image compression

In this section, we look at a simple example of low rank approximation with image compression. Let's see an example image.

```{code-cell}
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_sample_images

dataset = load_sample_images()     
image = dataset.images[0] 

# Display image
fig = plt.figure(figsize=(15, 11))
plt.imshow(image, cmap = 'gray')
plt.axis('off')
plt.show()

# Print shape
print('Dimensions:', image.shape)
```

As we can see, this image is of dimension $(427, 640, 3)$, but let's simplify the problem a bit and make it grayscaled. We can do this by taking the average over the third axis.

```{code-cell}
image = image.mean(axis=2)

# Display image
fig = plt.figure(figsize=(15, 11))
plt.imshow(image, cmap = 'gray')
plt.axis('off')
plt.show()

# Print shape
print('Dimensions:', image.shape)
```

Now we can think of this image as a $427 \times 640$ matrix $A$. 

Let's compute the SVD of the image. 

```{code-cell}
U, s, Vt = np.linalg.svd(image, False)
```

We can check the rank of $A$ by checking how many non-zero singular values it has.

```{code-cell}
print('rank(A) = %s' % len(s[s>0]))
```

So $A$ is a rank $427$ matrix.

In the following we construct the outerproducts of the first eight singular vectors and values, i.e. the first eight terms $\sigma_i u_iv_i^\top$.

```{code-cell}
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,5))
axs = axs.flatten()
for i in range(8):
    axs[i].imshow(np.outer(U[:,i],Vt.T[:,i])*s[i], cmap = 'gray')
```

These don't appear to look like much, but when we _sum_ them, we can obtain approximations to the original image at various levels of quality.

```{code-cell}
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,5))
axs = axs.flatten()
idx = 0
for i in [1,10,20,40,60,100,200, 300]:
    outer_products = [np.outer(U[:,j],Vt.T[:,j])*s[j] for j in range(i)]
    reconstruction = np.sum(np.asarray(outer_products), axis=0)
    axs[idx].imshow(reconstruction, cmap = 'gray')
    axs[idx].set_title('rank k= %s' % i)
    idx += 1
    
fig.tight_layout()
```

As we can see, as soon as we get to rank $k=40$, we get a fairly good approximation to the original image, and by rank $k=200$ the approximation and the original image are nearly indistinguishable visually.

As we mentioned above, the error from the rank $k$ approximation is given by $\|\widehat{A}_k-A\|_F = \sqrt{\sum_{i=k+1}^{r}\sigma_i^2}$.  Let's plot this error as a function of $k$. 

```{code-cell}
sr = np.flip(s**2)
errors = np.cumsum(sr)
errors = np.flip(errors)
errors = np.sqrt(errors)

plt.plot(range(1,428), errors)
plt.xlabel('k')
plt.ylabel('Error of rank k approximation')
plt.show()
```

As we can see, the errors start large when $k$ is small, but quickly get smaller as we increase the rank of our approximation.

