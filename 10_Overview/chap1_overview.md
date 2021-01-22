---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Linear Algebra for Data

Stripped to its mathematical essentials, data science represents a blending of concepts from linear algebra, geometry, calculus, and probability theory.  Among these, linear algebra plays a crucial role, because the heart of any data analysis consists of one or multiple data matrices, and linear algebra is all about manipulating matrices. Hence, to learn or extract knowledge from data, a data scientist needs to have a working understanding of matrix and vector operations. But a data scientist also needs to know about data preparation and data cleaning as well as data visualization. A profound knowledge just about statistical modeling and inference is not sufficient for being a modern data scientist. In this course you will learn to understand the mechanics behind many tools that are used in data science by learning about linear algebra. 

Let's start at the beginning, with data. In order to learn from data, you need to collect some data (this can be a difficult and tedious job) or someone needs to give you some data. Typically, data are collect in the form of a table where the columns represent different variables of interest and the rows contain measurements for the subjects. Before you can start to ''manipulate'' or ''crunch'' your data, you need to read the raw data into your environment. In python, we can use the pandas library to do so.  

```{code-cell}
import pandas as pd
data = pd.read_csv('data.csv')
```

It is a good idea to print the first few rows (say the first three, `n=3`) of the data frame to quickly test if your data frame has the right type of data in it. 

```{code-cell}
data.head(n=3)
```

By looking at the head of this data frame it is clear that the columns of the data matrix store measurements for height, weight and age. However, this data frame doesn't specify the units of the measurements. Anyway, you can guess that we used the metric system for height and weight, i.e., centimeters and kilo grams. 

Next, you want to check how many measurements you have collected. 

```{code-cell}
data.shape
```

This dataframe contains 3 different measurements for 5 subjects. This is a pretty small data frame, of course. Later, we will deal with much bigger data frames.

Now, how can we learn from data? A very intuitive approach is to start by visualizing the data. Data visualization is broad field and you often it takes a lot of time to create informative plots. Here is one way how you can visualize the data.

```{code-cell}
import seaborn as sns

g = sns.PairGrid(data, diag_sharey=False)
g.map_upper(sns.scatterplot, s=45)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
```

There are lot's of things going on here, so you should spent some time to look at these plots. You should be able to describe some of the key take always in your own words.

* We only have a very few data points, so don't read too much into the data.
* The data are roughly normal distributed. Is this plausible? 
* It seems that there is a relationship between height and weight. Is the relationship positive or negative?
* What do you think, is there a relationship between height and age and weight and age? 
* Does the data look plausible, i.e., does the data match with your intuition?

It is an important first step to gain some intuition for the data. But typically, we also want to summarize the data using some `quantitative' statistics. Linear algebra provides us many tools to compute such statistics in a very concise and efficient manner. Therefore, we need to represent the actual informational content of the data frame as a data matrix. 

A data matrix is is simply a rectangular array of numerical entries.

$$
\mathbf{A} = \begin{bmatrix} 
175 & 75 & 21 \\
182 & 85 & 25\\
173 & 72 & 21 \\
165 & 60 & 23 \\
191 & 92 & 22 \\
\end{bmatrix}
\quad
$$

This data matrix, which we denote as $\mathbf{A}$, has 3 columns and 5 rows and 15 entries. 

> For now it is fine if you think about a matrix as a table that has numerical values in it, i.e., a data matrix. But, a matrix can also have more general entries such as complex numbers or functions. We will talk about matrices and their properties in much more detail later in this course. 


In Python, we can transform a data frame into a NumPy array to represent the above data matrix.

```{code-cell}
import numpy as np
A = data.to_numpy()
print(A)
```

Numpy arrays are elegant and powerful objects that allow us to to perform a range of operations on the data matrix.

> Can you transform a data frame that contains string variables into a Numpy array?

Note, that the main disadvantage of matrices is that they do not provide any meta information about the rows or columns. For this reason, data scientists typically prefer to wrap a data matrix as a dataframe that has additional information. For instance, we can also create a data frame that takes the data matrix `A` as input as well as an additional list that specifies the column names. 

```{code-cell}
import pandas as pd
df = pd.DataFrame(A, columns=['height', 'weight', 'age'])
print(df)
```

> You should be comfortable to work with data frames and also know how to transform data frames into NumPy arrays.

One of the simplest statistics that we often want to compute is the mean of a variable. We can do this by using the following numpy routine. 

```{code-cell}
print(np.mean(A, axis=0))
```

But did you know that NumPy is using fast vectorized array operations for computing the mean? Using linear algebra allows you to efficiently compute the mean even if you deal with billion of data points. You get the same answer as above by performing a matrix transpose, a matrix vector multiplication and a scalar division.

```{code-cell}
print(np.ravel(A.T.dot(np.ones((A.shape[0], 1))) / A.shape[0]))
```

We will talk much more about these concepts later. Another important concept that you will learn a lot about are vectors. A vector is an ordered set of numbers, which we can write as

$$
\mathbf{x} = \begin{bmatrix} 
				x_{1}  \\
                               x_{2} \\
                               \vdots \\
                               x_{n}
           \end{bmatrix}.
$$

The $x$'s denote numbers and are called components or entries of the (column) vector $\mathbf{x}$. Note, we use bold letters to denote the vector in order to better distinguish between the components and vector itself. Here the vector has $n$ components and hence we say that the vector is of order $n \times 1$. As a more concrete example for a column vector, we can extract the first column of the matrix $\mathbf{A}$.

```{code-cell}
x = A[:,0]
print(x)
```

We can also extract rows of the matrix $\mathbf{A}$, which we call row vectors. Here is an example.

```{code-cell}
x = A[1,:]
print(x)
```

> If we are only interested in vectors and operations on vectors, then it matters less whether we distinguish between row and column vectors, but starting with Chapter 2, we will be more careful about this distinction, since it is often important.

 In this course you will also learn how you can explore relationships between different variables. For instance, you might be interested in investigating whether there is a positive relationship between the height and weight of a person? From your Data 100 class you know that you can use the the correlation coefficient to do so, but in this course you will learn more about how to compute the length and angles between different columns in your data matrix and how these operations can be scaled to large-scale data matrices.

 For instance, to quantify whether there is some relationship between height and weight we can compute the  (cosine of the) angle $ \gamma$ between two vectors to provide a notion of closeness. 

$$
 \gamma = \cos(\theta) = \frac{x \cdot y}{\|x\|_2\|y\|_2} .
$$

 Computing $\gamma$ requires two key ingredients, i.e., we need to learn about basic operations on vectors. First, we need to be able to compute a dot product between two vectors.

$$
 x \cdot y = \sum_{i=1}^{n}x_iy_i  .
$$

Secondly, we need to be able to compute the (euclidean) norm of a vector. 

$$
\|x\|_2 = \left(\sum_{1=1}^{n} x_i^2 \right)^{1/2} = \left(x \cdot x\right)^{1/2}.
$$

Again, we will talk in great detail about these concepts, but for now we can just use routines provided by NumPy to do the computations. Before we do so, it is a good idea to mean center all the columns. That is, because the measurements that we collected have different units. We know already how we can compute the mean of every column and we simply have to subtract the mean now. 

```{code-cell}
Z = A - np.ravel(A.T.dot(np.ones((A.shape[0], 1))) / A.shape[0])
print(Z)
```
We store the results in a new variable Z and use this matrix to compute the (cosine of the) angle between the first and second column of the matrix.


```{code-cell}
x = Z[:,0]  # extract first column, height
y = Z[:,1]  # extract second column, weight

xy = x.dot(y) # compute dot product between x and y
x_norm = np.linalg.norm(x) # euclidean norm of x
y_norm = np.linalg.norm(y) # euclidean norm of y

gamma = xy / (x_norm * y_norm)
print(gamma)
```

As expected, we see that the vectors are very close, which in turn suggest that there might exist some relationship between height and weight. 

In practice, you don't need to reinvent the wheel. We get the same answer by using a routine from SciPy

```{code-cell}
from scipy import spatial
gamma = 1 - spatial.distance.cosine(Z[:,0], Z[:,1])
print(gamma)
```

Now we can also quickly check whether there is some relationship between height and age. 

```{code-cell}
from scipy import spatial
gamma = 1 - spatial.distance.cosine(Z[:,0], Z[:,2])
print(gamma)
```

You might not be surprised that you get the same answers when you are computing the correlation coefficients.

```{code-cell}
print(np.corrcoef(A, rowvar=False)[0,1]) # correlation coef. between height and weight
print(np.corrcoef(A, rowvar=False)[0,2]) # correlation coef. between height and age
```

It can be seen that the two vectors are not very close and this is somewhat plausible from a practical point of view, i.e., there is no reason to expect that there is a relationship between height and age.

To further investiage the relationship between variables you can formulate linear regression model. For example, we might want to explore whether height and age impacts the weight of a person. You most likely have learned about linear regression models before, so we skip all the details here. Given a data frame you can estimate the parameters of a model and some other statistics using the statsmodel library. 


```{code-cell}
import statsmodels.formula.api as sm

result = sm.ols(formula="weight ~ height + age", data=data).fit()
print(result.summary())
```

First, we can see that the height has a statistical significant effect on the weight, i.e., the estimated parameters is positive and the p-value is small. In contrast, the p-value for age is not significant. In this course, we will not dwell too much on inference, but the underlying computations that are required to obtain the estimates are based on tools from linear algebra. We will talk a lot about the mechanics of linear regression in this course.

> The above regression routines returns the following error: "The condition number is large, 4.25e+03. This might indicate that there are
strong multicollinearity or other numerical problems." Do you know what a condition number is and what multicollinearity means? If not, then you should take this course to learn all about it.
