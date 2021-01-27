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

# (CH2): Algebra, algebras, and functions

An algebra is a specific mathematical structure defined by a specific set of symbols and a specific set of rules for manipulating those symbols.
For example, linear algebra, as opposed to elementary high school algebra or Boolean algebra or abstract algebra.

In this chapter, we will give several examples of algebras that will be useful for the study of linear algebra, including the algebra of sets, and the algebra of transformations/functions.
We want to illustrate several things.

- How to perform basic set operations in python.

- How to read a python data frame and perform basic operations on it. 

- How basic set theory operations relate to basic data frame operations. 

- Basic operations on functions and how they can be implemented with data frame operations (including composition of functions, associativity of composition, idempotent functions, and two views of idempotence)

- Surjective/injective functions and right/left inverses, including non-uniqueness, and examples with data frames

- Generalized inverses and inverses and examples with data frames

We will illustrate these concepts with a data frame we read in, and we'll ask you to work through them on different tables that you read in.
