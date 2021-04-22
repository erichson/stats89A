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

# Inverses Revisited

In this section, we extend our previous study of inverses (left, right and generalized), and show how to compute them in the general case. We will also see that there is a connection between left and right inverses and the projection matrices that we've studied earlier in this chapter.

## Left Inverses

In the previous chapter, we saw that a linear map $f(x) = Ax$ from $\mathbb{R}^n$ to $\mathbb{R}^m$ (where $A$ is an $m\times n$ matrix) is _injective_ if and only if the columns of $A$ are linearly independent. This, in turn, means that if $f$ is injective, we must have that $n\leq m$. Therefore, injective linear maps are represented by "tall" matrices $A$ with more rows than columns. 

Let's see how we can compute left inverses for such matrices in general. For a matrix $A$ with linearly independent columns, the matrix


$$
A_{\text{left}} = (A^\top A)^{-1}A^\top
$$
 

is always a left inverse for $A$. This is easy to verify, since


$$
A_{\text{left}}A = (A^\top A)^{-1}A^\top A = I
$$


Let's see an example. First, we will draw a random $m\times n$ matrix.

```{code-cell}
import numpy as np

m = 20
n = 10

A = np.random.normal(size=(m,n))
```

Now let's define the matrix $A_{\text{left}}$. 

```{code-cell}
ATA = np.dot(A.T, A)
A_left = np.dot(np.linalg.inv(ATA), A.T)
```

Let's check that $A_{\text{left}}A$ is indeed equal to the identity matrix.

```{code-cell}
np.dot(A_left, A).round(8)
```

Indeed, we get the identity back.

On the other hand, $A_{\text{left}}$ is not generally a right inverse for $A$ -- indeed, we observe


$$
AA_{\text{left}} = A(A^\top A)^{-1}A^\top
$$


The matrix on the right should be familiar: it is precisely the orthogonal projection onto the column space of $A$. 

While this left inverse is special in the sense that it induces the orthogonal projection matrix, it is not unique. Indeed, for any $m\times n$ matrix $C$ for which $C^\top A$ is invertible, we have that the matrix


$$
A_{\text{left},C} = (C^\top A)^{-1}C^\top 
$$


is also a left inverse for $A$. It is also easy to verify this fact:


$$
A_{\text{left},C}A = (C^\top A)^{-1}C^\top A = I
$$


Let's see an example with the same matrix $A$ that we defined above. When $A$ is a random $m\times n$ matrix, it turns out that for (almost) any other random $m\times n$ matrix $C$, the matrix $C^\top A$ will be invertible, hence allowing us to define a left inverse $A_{\text{left},C}$. Let's check that this is in fact the case.

```{code-cell}
C = np.random.normal(size=(m,n))
CTA = np.dot(C.T,A)
A_left_C = np.dot(np.linalg.inv(CTA), C.T)
```

This should again give us a left inverse, but let's verify.

```{code-cell}
np.dot(A_left_C, A).round(8)
```

As expected, $A_{\text{left},C}A = I$. Of course, $A_{\text{left},C}$ is very must different than $A_{\text{left}}$, thus demonstrating that $A$ has _many_ left inverses.

Similarly, $A_{\text{left},C}$ is not a right inverse (in general) for $A$, but it does induce a projection:


$$
AA_{\text{left},C} = A(C^\top A)^{-1}C^\top
$$


As we saw earlier in this chapter, this matrix is an _oblique projection onto the column space of $A$_.

### Not all tall matrices are injective

From our discussion so far, you may be tempted into think that "tall" matrices, i.e. $m\times n$ matrices $A$ with $n\leq m$,  are always injective. However, this is far from the case: there are many tall matrices which _do not_ have linearly independent columns, and hence do not represent injective linear functions. For example, consider the following matrix:


$$
A = \begin{pmatrix}1 & -2\\ 3/2 & -3 \\ -4 & 8\end{pmatrix}
$$


While this matrix indeed has more rows than columns, the function $f(x) = Ax$ is not injective. To see this, note that the second column of $A$ is equal to $-2$ times the first column, and thus the columns are not linearly independent. Equivalently, this means that the matrix $A^\top A$ is not invertible (nor is any matrix of the form $C^\top A$), and so the formulas for $A_{\text{left}}$ and $A_{\text{left},C}$ are not well defined.

We can also see this directly from the definition of an injective function. Recall that for $f(x)$ to be injective, we need to have that $f(x) = f(y)$ implies $x=y$. However, this is not the case for the function $f(x) = Ax$. For example, consider the vectors $x = \begin{pmatrix}2\\1\end{pmatrix}, y=\begin{pmatrix}-1\\ -1/2\end{pmatrix}$. Let's compute $f(x)$ and $f(y)$ for these two vectors.

```{code-cell}
A = np.array([[1,-2], [3./2, -3], [-4, 8]])
x = np.array([2,1])
y = np.array([-1, -1./2])

fx = np.dot(A,x)
fy = np.dot(A,y)

print('f(x) = %s' % fx)
print('f(y) = %s' % fy)
```

 Indeed, both $f(x)$ and $f(y)$ are equal to the zero vector. But clearly $y\neq x$, and so $f(x)$ cannot be injective. 

## Right Inverses

Recall that a linear map $f(x) = Ax$ is _surjective_ if and only if the columns of $A$ span all of $\mathbb{R}^m$. This means that if $f$ is surjective, we must have that $n\geq m$ -- i.e. $A$ must be "wide", with more columns than rows. If $f$ is surjective, then it will always have at least one right inverse. 

Let's see how we can find a right inverse for $A$ with columns spanning $\mathbb{R}^m$. The matrix 


$$
A_{\text{right}} = A^\top(AA^\top)^{-1}
$$


is always a right inverse for $A$. This is again easy to verify, since


$$
AA_{\text{right}} = AA^\top(AA^\top)^{-1} = I
$$


Let's again see an example.

```{code-cell}
m = 10
n = 20 # now more columns than rows

A = np.random.normal(size=(m,n))
```

Let's compute $A_{\text{right}}$.

```{code-cell}
AAT = np.dot(A, A.T)
A_right = np.dot(A.T, np.linalg.inv(AAT))
```

Now we can check that $AA_{\text{right}}$ does in fact give us the identity:

```{code-cell}
np.dot(A, A_right).round(8)
```

As expected, we get the identity.

On the other hand, $A_{\text{right}}$ is not in general a left inverse for $A$. Indeed, we see


$$
A_{\text{right}}A = A^\top(AA^\top)^{-1}A
$$


This matrix, while not the orthogonal projection onto the column space that we've seen before, is still a projection. Let's verify by checking that $P^2 = P$, for $P=A^\top (AA^\top)^{-1}A$. 


$$
P^2 = (A^\top (AA^\top)^{-1}A)^2 = A^\top \underbrace{(AA^\top)^{-1}AA^\top}_{I} (AA^\top)^{-1}A = A^\top (AA^\top)^{-1}A = P
$$


Indeed, the matrix $A^\top (AA^\top)^{-1}A$ is the orthogonal projection onto the _row space_ of $A$; that is, the orthogonal projection onto the span of the rows of $A$. This is easy to see by simply replacing $A$ with $A^\top$ in the formula for the projection onto the column space.

As one might guess, this is not the only right inverse for such a matrix $A$. Indeed, for any matrix $C$ such that $AC^\top$ is invertible, we have that 


$$
A_{\text{right},C} = C^\top (AC^\top)^{-1}
$$


is a right inverse for $A$. This is again easy to check:


$$
AA_{\text{right},C} = AC^\top(AC^\top)^{-1} = I
$$


Let's see another example.

```{code-cell}
C = np.random.normal(size=(m,n))
ACT = np.dot(A, C.T)
A_right_C = np.dot(C.T, np.linalg.inv(ACT))
```

Now we can verify that $AA_{\text{right},C} = I$:

```{code-cell}
np.dot(A, A_right_C).round(8)
```

And of course, $A_{\text{right},C}$ is very much distinct from $A_{\text{right}}$, therefore demonstrating that $A$ in general has many right inverses.

It is, however, not generally a left inverse, since


$$
A_{\text{right},C}A = C^\top(AC^\top)^{-1}A
$$


will not generally be equal to the identity matrix. Instead, $C^\top(AC^\top)^{-1}A$ is an _oblique projection onto the row space of $A$_. 

### Not all wide matrices are surjective

Like for injective linear functions, not all "wide" $m\times n$ matrices $A$ (i.e. those with $n\geq m$) represent surjective linear functions. Indeed, if the columns of $A$ don't span all of $\mathbb{R}^m$, then $f(x) = Ax$ will not be surjective, and therefore won't have a right inverse. For example, we can consider the same matrix $A$ as before, but transposed:


$$
A = \begin{pmatrix}1 &  3/2 & -4\\ -2 & -3  & 8\end{pmatrix}
$$


Note that each column is a scalar multiple of each of the other columns, and therefore the columns do not span $\mathbb{R}^2$. To see that $f(x) = Ax$ is not surjective directly, consider the vector $y = \begin{pmatrix}1\\ 2\end{pmatrix}$. Is there any vector $x$ such that $f(x) = Ax = y$? The answer is no: this is because the vector $Ax = \begin{pmatrix}1 &  3/2 & -4\\ -2 & -3  & 8\end{pmatrix}\begin{pmatrix}x_1 \\x_2\\x_3\end{pmatrix} = \begin{pmatrix} x_2 + 3x_2/2 - 4x_3\\ -2x_1 - 3x_1 + 8x_3\end{pmatrix}$ is always of the form $\begin{pmatrix}\alpha\\-2\alpha\end{pmatrix}$. The vector $y = \begin{pmatrix} 1 \\ 2\end{pmatrix}$ is clearly not of this form, and so there is no $x$ for which $f(x) = y$. Therefore $f$ cannot be surjective.

## Generalized Inverses

In chapter 2, we saw that functions can also have special complementary functions called _generalized inverses_. For a function $f$, a generalized inverse is a function $g$ which satisfies 


$$
f \circ g \circ f = f \hspace{10mm} \text{and} \hspace{10mm} g\circ f \circ g = g
$$


In the context of linear functions of the form $f(x) = Ax$, a generalized inverse is a matrix $A_g$ satisfying


$$
AA_g A = A \hspace{10mm} \text{and} \hspace{10mm} A_g A A_g = A_g
$$


Of course, if $f(x) = Ax$ is has either a left or right inverse (i.e. is either injective or surjective), then its left or right inverses are automatically generalized inverses. This is easy to see; for example, if $A_{\text{left}}$ is a left inverse for $A$, then


$$
A\underbrace{A_{\text{left}}A}_{I} = A \hspace{10mm} \text{and} \hspace{10mm} \underbrace{A_{\text{left}}A}_IA_{\text{left}} = A
$$


The same argument works for an arbitrary right inverse $A_{\text{right}}$. More interesting examples appear for matrices $A$ which are neither injective nor surjective, and therefore do not have left or right inverses. However, such matrices can still have generalized inverses. Let's see an example.

To construct a matrix $A$ for which $f(x) = Ax$ is neither injective nor surjective, we use the following method: we generate a "tall" matrix $A$, but in such a way that the columns of $A$ are linear dependent.

```{code-cell}
m = 10

a1 = np.random.randn(m)
a2 = np.random.randn(m)
a3 = np.random.randn(m)
a4 = a1 - a2 + 3*a3 # the 4th column is a linear combination of the other columns

A = np.stack([a1,a2,a3,a4], axis=1)
```

Now $A$ is a $10\times 4$ matrix, but does not represent an injective linear function because the fourth column of $A$ is a linear combination of the first three columns. Therefore, $A^\top A$ will not be invertible, and so we won't have a left inverse for $A$. On the other hand, we can construct a generalized inverse $A_g$ for $A$. To do this, we will use a _singular value decomposition_ of $A$. We haven't covered this decomposition yet, so for now don't worry about the details.

```{code-cell}
U,S,VT = np.linalg.svd(A, full_matrices=False)
S_inv = np.linalg.pinv(np.diag(S))

A_g = np.dot(VT.T, np.dot(S_inv, U.T))
```

Now let's verify that $AA_g A = A$:

```{code-cell}
AAgA = np.dot(A, np.dot(A_g, A))
np.allclose(AAgA, A)
```

Similarly, let's verify that $A_g A A_g = A_g$.

```{code-cell}
AgAAg = np.dot(A_g, np.dot(A, A_g))
np.allclose(AgAAg, A_g)
```

Indeed, it is. Therefore, $A_g$ satisfies the conditions of a generalized inverse of $A$, even though $f(x) = Ax$ is neither injective nor surjective. 