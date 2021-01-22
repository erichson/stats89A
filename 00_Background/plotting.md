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



# MatPlotLib 

## Basic Plots

Plotting with MatPlotLib: we have to import another library.

```{code-cell} 
import numpy
import matplotlib.pyplot as plt
%matplotlib inline
```

Define some points to plot

```{code-cell} 
x1 = [-2,-1,0,1,2]
x2 = [4,1,0,1,4]
```

Do a first plot.

```{code-cell} 
fig = plt.figure()
plt.axhline(linewidth=1, color='black', linestyle='--')
plt.axvline(linewidth=1, color='black', linestyle='--')
plt.plot(x1,x2, lw=2, color='red')
```

We can redefine our points using `np.linspace`. This allows us to get many more points.

```{code-cell} 
import numpy as np
x1_highres = np.linspace(-2,2,40)
x2_highres = x1_highres**2
```


```{code-cell}
fig = plt.figure()
plt.plot(x1,x2)
plt.plot(x1_highres,x2_highres)
```

Lots of arguments can be given to plotting functions.  It's worth knowing how to use them.

```{code-cell} 
fig = plt.figure()

##
## Comment and uncomment each of the following in turn to see how the plot changes
##

plt.plot(x1,x2)
plt.plot(x1_highres,x2_highres)

plt.plot(x1_highres,x2_highres,color="blue",linestyle="solid")
plt.plot(x1_highres,x2_highres,color="green",linestyle="dashed")
plt.plot(x1_highres,x2_highres,color="green",linestyle="dashdot")
plt.plot(x1_highres,x2_highres,color="red",linestyle="dotted")

plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=3)
plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='o')
plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker=',')
plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='v')
plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='^')
plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='s')
plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='p')
plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='*')

plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=3,marker='o',markerfacecolor='blue')

plt.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=3,marker='o',markerfacecolor='blue',markersize=5)
```

You can also set titles and axis labels for plots.

```{code-cell} 
fig = plt.figure()

plt.plot(x1,x2)
plt.plot(x1_highres,x2_highres)
plt.title("$y=x^2$")

plt.xlabel("x-value")
plt.ylabel("y-value")
```

The thing above between the dollar signs is [LaTeX](https://matplotlib.org/3.1.1/tutorials/text/usetex.html), which is nice for equations, but dont worry if you dont know it. You can use text only in your labels and titles if you like.




## Matplotlib Cheatsheet

This section will serve as a cheat sheet for `matplotlib`, in addition to the intro to `matplotlib` included in last week's lab. You've probably already seen a lot of these plots in last week's lab and homework, or even earlier in this lab.

Note: some of the functionality mentioned here was done in last week's lab using explicitly defined axes and figure objects. While this allows greater flexibility, often times it is more convenient to just plot using plt (which uses an implicit axes object in the background). This is enough if you just need one plot per cell. In future homeworks in labs, feel free to use whichever method feels more natural for you.



Simple $y$ vs $x$ line plot using `plt.plot`.

```{code-cell} 
xs = np.arange(-5, 6)
ys = xs**2

plt.plot(xs, ys)

plt.title("parabola")
plt.xlabel("x value")
plt.ylabel("y value")
```

We can also do a scatterplot with `plt.scatter`.

```{code-cell} 
xs = np.arange(-5, 6)
ys = xs**2

plt.scatter(xs, ys)

plt.title("scatter parabola -- no line")
plt.xlabel("x value")
plt.ylabel("y value")
```

We can plot multiple functions on the same graph

```{code-cell} 
xs = np.arange(20)
y1 = np.sin(2*np.pi*xs/20)
y2 = np.cos(2*np.pi*xs/20)

plt.plot(xs, y1)
plt.plot(xs, y2)

plt.title("Two sinusoids")
plt.xlabel("x value")
plt.ylabel("y value")
```

We can also create a legend using `plt.legend`.

```{code-cell} 
xs = np.arange(20)
y1 = np.sin(2*np.pi*xs/20)
y2 = np.cos(2*np.pi*xs/20)

plt.plot(xs, y1, label='sin')
plt.plot(xs, y2, label='cos')
plt.legend()

plt.title("Two sinusoids: with a legend")
plt.xlabel("x value")
plt.ylabel("y value")
```

Label our axes and graph using `plt.xlabel`, `plt.ylabel`, and `plt.title`.

```{code-cell} 
xs = np.arange(20)
y1 = np.sin(2*np.pi*xs/20)
y2 = np.cos(2*np.pi*xs/20)

plt.plot(xs, y1, label='$\sin$') # We can use latex in between $$
plt.plot(xs, y2, label='$\cos$') # We can use latex in between $$
plt.legend()

plt.ylabel('$y$-axis') # We can use latex in between $$
plt.xlabel('$x$-axis') # We can use latex in between $$
plt.title('Graph of $y=\sin(x)$ and $y=\cos(x)$')  # We can use latex in between $$.
```

Scale our axes using `plt.xlim` and `plt.ylim`

```{code-cell} 
xs = np.arange(20)
y1 = np.sin(2*np.pi*xs/20)
y2 = np.cos(2*np.pi*xs/20)

plt.plot(xs, y1, label='sin')
plt.plot(xs, y2, label='cos')
plt.legend()

plt.ylabel('$y$-axis') # We can use LaTeX in between $$
plt.xlabel('$x$-axis') # We can use LaTeX in between $$
plt.title('Graph of $y=\sin(x)$ and $y=\cos(x)$')  # We can use LaTeX in between $$.

plt.xlim(-10, 30)
plt.ylim(-5, 5)
```

Notice how the $x$-axis seems to be more stretched out than the $y$-axis? Most of the time this is fine, but if you want to plot something where the $x$-$y$ aspect ratio really matters, you can do the following:

```{code-cell} 
xs = np.arange(20)
y1 = np.sin(2*np.pi*xs/20)
y2 = np.cos(2*np.pi*xs/20)

plt.plot(xs, y1, label='sin')
plt.plot(xs, y2, label='cos')
plt.legend()

plt.ylabel('$y$-axis')
plt.xlabel('$x$-axis')
plt.title('Graph of $y=\sin(x)$ and $y=\cos(x)$')  # We can use LaTeX in between $$.

plt.xlim(-10, 30)
plt.ylim(-10, 30)

plt.gca().set_aspect('equal')  # <--- do this to fix aspect ratio
```

We can also use `plt.scatter` to plot a bunch of 2D points. Let's first generate a bunch of 2D points between $[0, 1) \times [0, 1)$. Note that each column of this matrix is a 2-dimensional vector that contains the $x$- and $y$-coordinates of a point.

```{code-cell} 
points = np.random.rand(2, 10)
print('points:\n{}'.format(points))
print()
print('points has shape', points.shape)
```

We can slice the first row of points to obtain all the $x$ coordinates. We can slice the second row of points to obtain all the $y$ coordinates.

```{code-cell} 
xs = points[0,:]
ys = points[1,:]

print('xs (shape: {}):\n{}'.format(xs.shape, xs))
print()
print('ys (shape: {}):\n{}'.format(ys.shape, ys))
```

Once we have all the `xs` and `ys`, we can use `plt.scatter`. Note that `plt.scatter` will match the $i$th element of `xs` with the $i$th element of `ys`. So good thing we generated `xs` and `ys` together using matrix! This will ensure the coordinates are aligned.

```{code-cell} 
plt.scatter(xs, ys)

plt.title("Our beautiful scatter plot")
plt.xlabel("$x$ value")
plt.ylabel("$y$ value")
```
