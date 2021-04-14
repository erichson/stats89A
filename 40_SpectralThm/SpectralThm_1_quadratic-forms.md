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

# Quadratic forms

In this section, we investigate special functions called _quadratic forms_. These are functions of the form


$$
f(x) = x^\top Ax
$$


where $x\in \mathbb{R}^n$ and $A$ is an $n\times n$ matrix. To see why such functions are called quadratic forms, let's look at the case when $n=1$. In this case, $A = a$ and $x$ are both scalars, and so 


$$
f(x) = ax^2
$$


which is of course a usual quadratic. Importantly, depending on whether or not $a>0$ or $a<0$, this quadratic can be "pointing up" or "pointing down". For example, when $a=1$ we get

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt

xx = np.arange(-3,3.01,.01)

a = 1
plt.plot(xx, a*xx**2)
plt.title('ax^2 when a = %s' % a)
plt.show()
```

while when $a=-1$ we get

```{code-cell}
a = -1
plt.plot(xx, a*xx**2)
plt.title('ax^2 when a = %s' % a)
plt.show()
```

For $n>1$, we have the following expression for a quadratic form.


$$
f(x) = x^\top A x = \sum_{i,j=1}^n a_{ij}x_ix_j
$$


where $a_{ij}$ is the $(i,j)$th entry of $A$.

Let's look at a few special cases. For example, when the matrix $A$ is diagonal, then $a_{ij} = 0$ when $i\neq j$, and so we get


$$
f(x) = x^\top A x = \sum_{i=1}^n a_{ii}x_i^2
$$


which looks more like the 1-d quadratic we saw before.

## Visualizing quadratic forms

To get a better intuition for how quadratic forms work, let's focus on the $n=2$ case. In this case, our matrix $A$ is of the form


$$
A = \begin{pmatrix}a_{11} & a_{12}\\ a_{21} & a_{22}\end{pmatrix}
$$


and the associated quadratic form is the function


$$
f(x) = x^\top A x = \begin{pmatrix} x_1 & x_2\end{pmatrix}\begin{pmatrix}a_{11} & a_{12}\\ a_{21} & a_{22}\end{pmatrix}\begin{pmatrix}x_1\\ x_2\end{pmatrix} = a_{11}x_1^2 + a_{12}x_1x_2 + a_{21}x_2x_1 + a_{22}x_2^2
$$


Visualizing these functions is possible in $\mathbb{R}^2$ via a 3-d plot. Let's look at a simple example with 


$$
A_1= \begin{pmatrix}2 & 0 \\ 0&1\end{pmatrix}
$$

```{code-cell}
def f(x1, x2, A):
    return A[0,0]*x1**2 + A[0,1]*x1*x2 + A[1,0]*x2*x1 + A[1,1]*x2**2

x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)

X1, X2 = np.meshgrid(x1, x2)

A1 = np.array([[2,0],[0,1]])
X3 = f(X1,X2, A1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, X3, 50)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
plt.show()
```

Here we see that the quadratic form is a bowl shape, pointing upward. Let's see when happens when we make the entries of the matrix $A$ negative, i.e. using the matrix


$$
A_2= \begin{pmatrix}-2 & 0 \\ 0&-1\end{pmatrix}
$$

```{code-cell}
A2 = np.array([[-2,0],[0,-1]])
X3 = f(X1,X2,A2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, X3, 50)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
plt.show()
```

In this case, we get a downward shaped-bowl. These two cases are analogous to the simple 1-d examples, where $a>0$ or $a<0$. However, in 2d we also have a third case: where one of the entries is positive, and the other is negative. For example, let's consider the matrix


$$
A_3 = \begin{pmatrix}2&0\\ 0&-1\end{pmatrix}
$$

```{code-cell}
A3 = np.array([[2,0],[0,-1]])
X3 = f(X1,X2,A3)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, X3, 50)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
plt.show()
```

In this case, we get neither a bowl up nor a bowl down shape, but rather a hyperbolic surface. 

The last example we will see is a when one of diagonals is equal to zero. For example, the matrix


$$
A_4 = \begin{pmatrix}2 & 0\\ 0 &0\end{pmatrix}
$$


Notice that the function $f(x) = x^\top A_4 x = 2x_1^2$ and hence does not depend on the second coordinate of $x$ at all. Thus we should see that the plot of $f(x)$ is 'flat' along the $x_2$ axis. Let's check this.

```{code-cell}
A4 = np.array([[2,0],[0,0]])
X3 = f(X1,X2,A4)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, X3, 50)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
plt.show()
```

Indeed, we get a parabola-like shape as we vary $x_1$, but the plot of $f$ is constant as we vary $x_2$ while keeping $x_1$ fixed.

Another way to visualize quadratic forms is with looking at the _level curves_ of the function $f(x) = x^\top A x$. The _$\lambda$-level curve_ (or _level set_) of a function $f(x)$ is the set $\Omega_\lambda = \{x : f(x) = \lambda\}$. Level curves are also sometimes called contour plots. Let's see a few examples of them in python.

For example, let's look at $f(x)=x^\top A_1 x$. We can visualize the level curves of this function as follows.

```{code-cell}
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)

A1 = np.array([[2,0],[0,1]])

X1, X2 = np.meshgrid(x1, x2)
X3 = f(X1,X2, A1)

contours = plt.contour(X1, X2, X3)
plt.clabel(contours, inline=1, fontsize=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

Here the level curves of $f(x)$ form ellipses. We see that the values of the function are smallest near the origin, and grow larger as we move farther away from it. 

Let's see an example with the indefinite matrix $A_3$. 

```{code-cell}
A3 = np.array([[2,0],[0,-1]])
X3 = f(X1,X2, A3)

contours = plt.contour(X1, X2, X3)
plt.clabel(contours, inline=1, fontsize=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

For this matrix, the level curves of the quadratic form $x^\top A_3 x$ form hyperbolas.

## Positive definite, negative definite, and indefinite matrices

It turns out that the shapes of these surfaces can be characterized in terms of the _definiteness_ of the matrix $A$. 

Let $A$ be an $n\times n$ matrix. Then

- The matrix $A$ is called _positive definite_ if for any $x\neq 0$, we have that $f(x) = x^\top A x >0$. The matrix $A$ is called _positive semi-definite_ if for any $x\neq 0$, we have that $f(x) = x^\top A x \geq 0$.
- The matrix $A$ is called _negative definite_ if for any $x\neq 0$, we have that $f(x) = x^\top A x<0$. The matrix $A$ is called _negative semi-definite_ if for any $x\neq 0$, we have that $f(x) = x^\top A x \leq 0$.
- If the matrix $A$ is neither positive (semi-)definite nor negative (semi-)definite, then it is called _indefinite_.

In the above section, the matrix $A_1$ was an example of a _positive definite_ matrix, $A_2$ was an example of a _negative definite_ matrix and $A_3$ was an example of an _indefinite_ matrix.  The matrix $A_4$ was an example of a (strictly) _positive semi-definite_ matrix. To see why, note that for $x  = \begin{pmatrix}x_1\\ x_2\end{pmatrix}\neq 0$, we have


$$
x^\top A_1 x = 2x_1^2 + x_2^2 >0 \hspace{10mm}\text{and} \hspace{10mm} x^\top A_2 x = -2x_1^2 - x_2^2 <0. 
$$


On the other hand, for the matrix $A_3$ let's see what happens when we look at the vectors $e_1 = \begin{pmatrix}1\\0\end{pmatrix}$ and $e_1 = \begin{pmatrix}0\\1\end{pmatrix}$. Then


$$
e_1^\top A_3 e_1 = 2
$$


but


$$
e_2^\top A_3 e_2 = -1.
$$


Hence the function $f(x) = x^\top A_3 x$ can be either positive or negative depending on the input vector $x$.

Similarly, we have that $e_2^\top A_4 e_2= 0$, and so the function $f(x) = x^\top A_4 x$ cannot be positive definite.

Of course, the examples we saw here were all diagonal matrices, but we could just as easily look at matrices which aren't diagonal. 

Let's look at examples of positive definite and negative definite matrices which are not diagonal. To do this, a useful trick is to note that for any matrix $A$, the matrix $B_1 = A^\top A$ is always positive semi-definite, while, conversely, the matrix $B_2 = -A^\top A$ is always negative semi-definite. To see why this is the case, note that


$$
x^\top B_1 x = x^\top A^\top A x = (Ax)^\top Ax = \|Ax\|_2^2 \geq 0
$$


and


$$
x^\top B_2 x = -x^\top A^\top A x = -(Ax)^\top Ax = -\|Ax\|_2^2 \leq 0.
$$


Furthermore, the matrix $B_1$ (resp. $B_2$) will always be strictly positive (resp. negative) definite whenever $A$ is has full column rank, that is, whenever $A$ has linearly independent columns. Let's see a few examples.

To generate $B_1$ and $B_2$, we'll generate a random $2\times 2$ matrix and compute $B_1 = A^\top A$ and $B_2 = -A^\top A$. 

```{code-cell}
np.random.seed(1111)
A = np.random.normal(size=(2,2))
B1 = np.dot(A.T, A)
B2 = -B1
```

Now we can visualize the quadratic forms $x^\top B_1 x$ and $x^\top B_2 x$. 

```{code-cell}
X3 = f(X1,X2, B1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, X3, 50)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
plt.show()
```

Indeed, since $B_1$ is positive definite, we see that we get the familiar upward bowl shape.

Now let's look at $f(x) = x^\top B_2 x$.

```{code-cell}
X3 = f(X1,X2, B2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, X3, 50)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
plt.show()
```

Again, we get the usual downward bowl shape associated with a negative definite matrix.

Next, let's see an example of a _semi-definite_ matrix -- that is, one for which $x^\top A x = 0$ for some non-zero vector. Recall that the matrix $A$ will be will be (positive or negative) semi-definite if the columns of $A$ are not linearly independent. Let's use the following example:


$$
A = \begin{pmatrix} 1 & -2 \\ -1 & 2\end{pmatrix}
$$


Since the second column is $-2$ times the first column, this matrix will not be strictly positive or negative definite. Let's see what the function $f(x) = x^\top Ax$ looks like for this matrix.

```{code-cell}
A = np.array([[1,-2],[-1,2]])

X3 = f(X1,X2, A)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, X3, 50)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
plt.show()
```

Here we obtain an upward-pointing shape, but not quite the bowl shape that we had before. Indeed, it turns out that for any $x$ in  the subspace


$$
S = \{x = \begin{pmatrix}x_1\\ x_2\end{pmatrix} : x_1 = x_2\}
$$


we have that $x^\top A x = 0$. Visually, this is represented by the flat region where the function is equal to zero at the bottom of the bowl. Indeed, it turns out that for this matrix, this flat line at the bottom of the bowl is just the $x_2 = x_1$ line in the $(x_1,x_2)$ plane.

## Finding extrema of quadratic forms

For the purposes of this class, we are often interested in finding _extrema_ of quadratic forms $f(x) = x^\top A x$, that is maxima or minima of quadratic forms. For positive or negative definite matrices $A$, however, if $x$ is allowed to be an arbitrary vector, $f(x)$ can be made arbitrarily large or small, respectively. Therefore, it is convenient to restricting ourselves to unit vectors $x$. Therefore, we consider the following optimization problem:


$$
\max_{x}/\min_x f(x) = x^\top A x\\
\text{subject to}\;\;\; \|x\|_2^2 = x^\top x = 1
$$


In 2-d we can easy visualize the function $f(x) = x^\top A x$ for unit vectors $x$. This is because any unit vector $x\in \mathbb{R}^2$ lies on the unit circle, and is hence of the form $x(\theta) = \begin{pmatrix}\cos(\theta)\\ \sin(\theta)\end{pmatrix}$. Let's try plotting $f(x(\theta))$ as a function of the $\theta$.

First, let's draw a random $2\times 2$ matrix $A$, and construct the positive definite matrix $B = A^\top A$.

```{code-cell}
A = np.random.normal(size=(2,2))
B = np.dot(A.T,A)
```

Next, let's define the function $g(\theta) = f(x(\theta)) = x(\theta)^\top B x(\theta)$. 

```{code-cell}
def g(theta, B):
    x = np.array([np.cos(theta), np.sin(theta)])
    return np.dot(x, np.dot(B,x))

theta_range = np.arange(0, np.pi, .01)
plt.plot(theta_range, [g(theta,B) for theta in theta_range])
plt.xlabel('theta', fontsize=14)
plt.ylabel('f(x(theta))', fontsize=14)
plt.show()
```

Notice that this function is always positive -- as expected, since $B$ is positive definite. We should observe that $x(\theta)^\top B x(\theta)$ has two extrema: one maximum, and one minimum, both strictly greater than zero. The value of the function at these extrema are special: they are called the _eigenvalues_ of the matrix $B$. If $\theta_{min}, \theta_{max}$ are the values of $\theta$ at which the minimum/maximum occurs, then the vectors $x_{min} = \begin{pmatrix}\cos(\theta_{min})\\ \sin(\theta_{min})\end{pmatrix}$ and $x_{max} = \begin{pmatrix}\cos(\theta_{max})\\ \sin(\theta_{max})\end{pmatrix}$ are called the _eigenvectors_ of the matrix $B$. 

We can also look at an example when $B$ is negative definite.

```{code-cell}
B = -B
theta_range = np.arange(0, np.pi, .01)
plt.plot(theta_range, [g(theta,B) for theta in theta_range])
plt.xlabel('theta', fontsize=14)
plt.ylabel('f(x(theta))', fontsize=14)
plt.show()
```

This time, the function is always negative, again as we would expect from a negative definite matrix $B$. We again also have one maximum and one minimum, again corresponding to the _eigenvalues_ of the matrix $B$. 

Let's also look at an example where the matrix $B$ is indefinite. For example, consider the matrix


$$
B = \begin{pmatrix}1 & 2\\ -1 & -4 \end{pmatrix}
$$


```{code-cell}
B = np.array([[1,2], [-1,-4]])

theta_range = np.arange(0, np.pi, .01)
plt.plot(theta_range, [g(theta,B) for theta in theta_range])
plt.xlabel('theta', fontsize=14)
plt.ylabel('f(x(theta))', fontsize=14)
plt.show()
```

In this case, the function $f(x) = x^\top B x$ is sometimes positive and sometimes negative. $f$ still has one maximum and one minimum; this time the maximum is strictly positive, while the minimum is strictly negative. 

At this point, it should be intuitive the _eigenvalues_ (i.e. the extrema of the function $f(x) = x^\top B x$) are always non-negative for a positive semi-definite matrix, always non-positive for a negative semi-definite matrix, and both positive and negative for an indefinite matrix. 

### Bonus: Lagrange multipliers

If you're familiar with the method of Lagrange multipliers, this problem can be reframed as optimizing the following:


$$
L(x,\lambda) = x^\top A x - \lambda (x^\top x - 1)
$$


If we take the derivative of this function with respect to $x$ and set it equal to $0$, we get


$$
0 = \nabla_x L(x,\lambda) = 2Ax - 2\lambda x \implies Ax = \lambda x
$$


Thus the maxima of the constrained optimization problem (finding the maximum of $f(x)=x^\top A x$ subject to $\|x\|_2 = 1$) are the solutions of the equation $Ax = \lambda x$. Pairs $(\lambda, x)$ which satisfy this equation are important: they are called _eigenvalues_ and _eigenvectors_ of the matrix $A$.

### Another bonus: completing the square

Sometimes, we are given a function which looks a bit like a quadratic form, say a function that looks like


$$
f(x) = x^\top A x - 2b^\top x
$$


where here $b$ is some vector. Because functions quadratic forms are convenient to work with, we would like to rearrange $f$ so that it looks more like a pure quadratic form (i.e. without the linear term). It turns out that there is a version of completing the square that works even in higher dimensions. Indeed, if $A$ is an invertible matrix, then we can write


$$
f(x) = x^\top A x - 2b^\top x = (x-A^{-1}b)^\top A (x-A^{-1}b) - b^\top A^{-1} b
$$


If we use the change of variable $z= x-A^{-1}b$ and let $c = -b^\top A^{-1}b$, then we can write this as


$$
f(z) = z^\top A z + c
$$


which looks much more convenient to work with. While we won't use this trick immediately, it is frequently useful and worth remembering.