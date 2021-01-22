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

# (CH0): Python 101

Most things a computer does has been given to it by someone as a set of instructions. Each instruction performs a task and a set of instructions that does something useful is called a program. When you execute a program, the computer will work through this list from the top to the bottom and (hopefully) return the desired output at the end. In order to do so we need to have an agreed language that both ourselves and the computer can understand. Programming languages provide sets of simplified commands that the computer knows how to interpret so that it can be given instructions to do something useful. These commands are a range of English words and mathematical notation, for example `if`, `for`, `+` and `-` are all commands that are commonly used in programming languages. You will meet several different languages/packages within your data science degree, but the skills you learn in this course will be transferable to all of these.

In this course we will be using [Python][00], a language that is well suited to mathematical problems. Python is also the most popular language for data science. (Python is also widely used for web development and Quora, Pinterest and Spotify use Python for their backend web development for example.) The syntax is simple to read and understand, and there exist many powerful libraries for plotting, machine learning, deep learning, and scientific computing. Most important, Python is free and open-source. For these reasons we use Python in this course and we design the course in such a way that it can be viewed as a tutorial that introduces tools provided by Numpy, SciPy, Pandas, Scikit-Learn and Matplotlib. However, we expect that you have a basic programming skills and are familiar with concepts like variables, loops, conditional statements, functions etc.  While Python is an [*interpreted*][01] language and cannot compete in speed with [*compiled*][01] languages, we will learn how to take advantage of extensions such as [*Cython*][02] to write C extensions for Python. 

[00]: <https://wiki.python.org/moin/BeginnersGuide/Overview> "What is Python"
[01]: <https://www.freecodecamp.org/news/compiled-versus-interpreted-languages/> "Interpreted vs compiled"
[02]: <https://cython.org/> "Writing C extensions for Python as easy as Python itself"


Python is an interpreted language and to run a python program you need a python interpreter that looks at what you have written and executes the instructions line–by–line in turn. If you like to run all the exercises on your local machine, you can get [Anaconda][10]. Anaconda is a free, easy-to-install package manager, environment manager, and Python distribution with a collection of 1,500+ open source packages with free community support. Anaconda is platform-agnostic, so you can use it whether you are on Windows, macOS, or Linux.

[10]: <https://www.anaconda.com/> "Anaconda"
[11]: <https://colab.research.google.com> "Google Colab"
