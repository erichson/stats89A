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

# Functions and Compositions

In this section, we will continue with data frames, and we will use them to discuss functions and the composition of functions.  
Topics to be covered include the following.

- Binary relations, functions, and defining functions in terms of basic data frame operations.
- Image/preimage of functions and restrictions of functions.
- Composition of functions, and how to use data frame operations to compose two functions together.
- Associativity of function composition.
- Identity versus idempotence.

We will again work with the `students` dataset, which we import below.


```{code-cell} 
import pandas as pd
from datasets import students
students.head()
```


## Functions via binary relations

Functions seem simple enough.
You probably know that a function is (something like) a rule that assigns to each input one output.
That's not incorrect, but it is not the full story.
When we use functions in data science, we typically want to combine and compose them in various ways.
In that case, it is better to think of them as black box subroutines, and it is more important to be careful about the domain and range of the function.
To do that, let's start with a slightly more general notion, and then let's get more specific to different types of functions.

A binary relation is defined in the following way.
Given two sets, $A$ and $B$, a binary relation between $A$ and $B$ is a triple $\langle A, R, B \rangle$, where $ R \subset A \times B$ is any set of ordered paris from $A \times B$.
For example, we could have $A = $`StudentID` and $B = $`grade` and have $\langle a,b \rangle \in R$ iff student $a$ has the grade $b$.

We can view this relation using the students data frame by simply selecting the columns StudentID and Grade:

```{code-cell}
students[['StudentID', 'Grade']]
```

Note that this is _not_ a function, since a student is in general in multiple classes, but it is a binary relation.
For example, for `StudentID == 101`, we see the following.

```{code-cell}
tmp = students[['StudentID', 'Grade']]
student101 = tmp[tmp['StudentID'] == 101]
student101
```

Note also that we could have a student enrolled in no classes, and we could have a class with no students in it. 
Sometimes that could be encoded in the data frame, and sometimes it would not be, e.g., if the data frame only has students who are enrolled in at least one class. 
However, that is worth keeping in mind.

On the other hand, we could construct a function from `StudentID` to `Grade`. 
To do this, we need to find a way to choose a grade value to assign for students in multiple classes. 
Below, we do this by taking $f(StudentID)$ to be the maximum grade that student with ID ```StudentID``` has in any class. 
To do this, we again need the group by operation introduced in the previous workbooks. 
(Note: we need to use the option `as_index=False` to make sure Pandas doesn't try to use StudentID as the index -- this just makes the tables look nice.)

```{code-cell} 
students_to_grades = students.groupby('StudentID', as_index=False)[['StudentID', 'Grade']].max('Grade')
students_to_grades
```

This is a function.
For example, for `StudentID == 101` we see the following.

```{code-cell}
tmp = students_to_grades[['StudentID', 'Grade']]
student101 = tmp[tmp['StudentID'] == 101]
student101
```

To verify that this a function in code, we can check to make sure every student ID appears only once:

```{code-cell}
students_to_grades[['StudentID']].value_counts()
```

However, this was not the only way we could have constructed this function. 
For example, we also could have constructed a function from `StudentID` to `Grade` by selecting the minimum grade, or the average grade.


## Image/preimage and restriction

If a function is a rule that maps points from a set $A$ to set $B$, then the _image_ of $f$ is the subset of $B$ that gets mapped to by some element $a \in A$, and the preimage of $f$ is the subset of $f$ is the subset of $A$ that maps to something. 
Think, _range_ and _domain_. 

We can also have the image of a subset of $A$. 
For example, consider the function $f:$`StudentID` $\to$ `Grade`  that we defined above and the subset `sophomore` $\subseteq$  `StudentID` of sophomore students. We can compute the image of the subset `sophomore` under the function `students_to_grades `, denoted $f(\text{sophomore})$ which we obtain with the following code:

```{code-cell}
sophomore = students[students['Year'] == 'Sophomore']['StudentID']
students_to_grades[students_to_grades['StudentID'].isin(sophomore)][['Grade']]
```

Note that the resulting set is a subset of the set `Grade`. 
Likewise, we could consider the pre-image of some subset of grades under this function. 
For example, let's consider the subset of grades `[84, 76, 81, 95]`. 
We can obtain the pre-image of this set under the function $f$, denoted $f^{-1}(\{84,76,81,95\})$, which is just the set of students who have one of these grades as their maximum grade:

```{code-cell}
students_to_grades[students_to_grades['Grade'].isin([84, 76, 81, 95])][['StudentID']]
```

Alternatively, we can obtain the set of all students whose highest grade is in the 80s as a pre-image under the function `students_to_grades`

```{code-cell}
grades_in_80s = list(range(80,90)) #[80, 81, 82, ... , 88, 89]
students_to_grades[students_to_grades['Grade'].isin(grades_in_80s)][['StudentID']]
```

Given that, a useful notion is the _restriction_ of a function. 
Informally, this is the same function, except it is defined on a subset of $A$ and/or $B$. 
Let's look the function `students_to_grades` restricted to the set of sophomores, which we denote by $f|_{\text{sophomore}}$.

```{code-cell}
sophomore_students_to_grades = students_to_grades[students_to_grades['StudentID'].isin(sophomore)]
sophomore_students_to_grades
```

Notice that when we have a _partial function_ -- i.e. a function which is not defined on the entire domain -- we can construct a total function simply by restricting it to the elements of its domain which it is defined on.


## Composition of functions

Something that one often wants to do is to compose two functions. 
Informally, this means that we apply a second function on the output of the first function. 
In symbols, if $f:A\to B$ and $g:B\to C$, then the composition of these function is a new function $g\circ f : A\to C$ defined by $(g\circ f)(a) = g(f(a))$. 

When working with function composition, we want to think of the functions as balck boxes, and we are interested in how those black boxes compose, i.e., we are not interested in the particular inputs/outputs of the functions. 
Thus, the main gotcha when working with function composition has to do with when the function isn't defined for all elements of the input domain and/or doesn't map to all elements of the output domain. 
Let's see this with an example.

First, let's consider the function from `StudentID` to `Year` :

```{code-cell}
students_to_year = students.groupby('StudentID', as_index=False)[['StudentID','Year']].first()
students_to_year
```

This is a well-defined function, because each student only has a single year. 
On the other hand, what happens if we try to map back from `Year` to `StudentID`? 
Since there is more than one student in each year, there is not a unique way to define such a function. 

For example, for each year, we could select the student with the smallest student ID:

```{code-cell} 
year_to_students1 = students.groupby('Year',  as_index=False)[['Year', 'StudentID']].min('StudentID')
year_to_students1
```

However, it would also be perfectly valid to select the student in each year with the largest student ID number:

```{code-cell} 
year_to_students2 = students.groupby('Year',  as_index=False)[['Year','StudentID']].max('StudentID')
year_to_students2
```

Let's see what happens when we compose these functions with our function `students_to_year`. 
It turns out that we can do this again using _joins_, similar to how we used them to perform set intersections. In particular, we can perform a left join, which is essentially just an inner join between the domain of `year_to_students1` and the co-domain of `students_to_year`. 
Since both of these sets are subsets of the set `year`, such a join makes sense. 
Below we see how to do this in Python:

```{code-cell}
students_to_students1 = students_to_year.merge(year_to_students1, how="left", on="Year").drop("Year", axis=1)
students_to_students1
```

We can do the same composition with `year_to_students2`:

```{code-cell}
students_to_students2 = students_to_year.merge(year_to_students2, how="left", on="Year").drop("Year", axis=1)
students_to_students2
```

Above we perform the join on the column "Year", and then drop the column "Year" (so that we are just left with the mapping taking `students` to `students`). We'll see several more examples of function composition in the following sections.


## Associativity of function composition

An important property of function composition is that it is _associative_, meaning that if we have functions $f,g,h$, we have that
$
h\circ(g\circ f) = (h\circ g)\circ f
$.
In words, this means that when we want to compose three functions together, we can either compute the composition of $g$ and $f$ first, and then compose $h$ on the left, or compute the composition of $h$ and $g$ first, and then compose $f$ on the right. 
Let's see how this works with our students dataset. 

To do this, let's make another function mapping `StudentID` to `Major`. 

```{code-cell}
students_to_major = students.groupby('StudentID', as_index=False)[['StudentID','Major']].first()
students_to_major
```

Next, we'll make a function which maps from `Major` to `Year`. We can do this in many ways, since there are students of many different years in each major. For this example, we'll just pick the first `Year` after performing a group by within each major.

```{code-cell}
major_to_year = students.groupby('Major', as_index=False)[['Major', 'Year']].first()
major_to_year
```

For our last function, we'll use the function `year_to_students1` defined in the previous section. 

```{code-cell}
year_to_students1
```

 We're going to compute the composition `year_to_students1` $\circ$(`major_to_year` $\circ$ `students_to_major`) -- which is a function mapping `StudentID` to `StudentID` -- and show that it is equal to  (`year_to_students1` $\circ$`major_to_year`) $\circ$ `students_to_major`.

Let's start off by computing (`major_to_year` $\circ$ `students_to_major`), which we can do with a left join:

```{code-cell}
mty_comp_stm = major_to_year.merge(students_to_major, how="left", on="Major").drop("Major", axis=1)[['StudentID', 'Year']]
mty_comp_stm
```

Notice that this gives us a function from `StudentID` to `Year`, but it is of course a different function than the function `students_to_year` we defined before. Next, we compose with `year_to_students1` on the left, to get the composition `year_to_students1` $\circ$(`major_to_year` $\circ$ `students_to_major`):

```{code-cell}
composition1 =  mty_comp_stm.merge(year_to_students1, how="left", on="Year").drop("Year", axis=1)
print('year_to_students1 o (major_to_year o students_to_major) is given by')
composition1
```

Next, let compute the composition the other way; namely, let's compute (`year_to_students1` $\circ$`major_to_year`) $\circ$ `students_to_major`. We'll start by computing  (`year_to_students1` $\circ$`major_to_year`), which is a function from `Major` to `StudentID` :

```{code-cell}
yts_comp_mty = year_to_students1.merge(major_to_year, how="left", on="Year").drop("Year", axis=1)[['Major', 'StudentID']]
yts_comp_mty
```

Next, we'll compose this with `students_to_major` on the right. 

```{code-cell}
composition2 = students_to_major.merge(yts_comp_mty, how="left", on="Major").drop("Major", axis=1)
print('(year_to_students1 o major_to_year) o students_to_major is given by')
composition2
```

As we can see, we do in fact have that  `year_to_students1` $\circ$ (`major_to_year` $\circ$ `students_to_major`) =  (`year_to_students1` $\circ$`major_to_year`) $\circ$ `students_to_major` -- that is, we've shown the associative property!


## Function compositions yield the Identity

In the previous two section, we composed functions and obtained new function mapping the set `StudentID` to itself. 
However, none of these compositions gave us the _identity function_ on the set `StudentID` -- namely, each student ID was not mapped to itself. 
It is, however, very easy to compose two functions and obtain the identity function. 
In this section, we give a few examples of such a composition. 

First, let's consider the function `students_to_major` which we've used in the previous sections. 
Is there a function that we compose on the right to obtain the identity? 
Let's try and find one. 
Below, we define a function`major_to_students`, which assigns to each major the student with the smallest student ID. 

```{code-cell}
major_to_students = students.groupby('Major', as_index=False)[['Major','StudentID']].min('StudentID')
major_to_students
```

Now, what happens if we compose these to get a function `students_to_major` $\circ$ `major_to_students`, from `Major` to `Major`? 
Let's see:

```{code-cell}
major_to_major = major_to_students.merge(students_to_major, how="left", on="StudentID").drop("StudentID", axis=1)
major_to_major
```

This time we do indeed get the identity function on the set `Major`! 

What happens if we try to "reverse" the operation, and instead compute the composition `major_to_students` $\circ$ `students_to_major`, which this time is a function between `StudentID` and `StudentID`?

```{code-cell}
students_to_students2 = students_to_major.merge(major_to_students, how="left", on="Major").drop("Major", axis=1)
students_to_students2
```

This is distinctly _not_ the identity function. 
So, we can compose in one direction and get the identity, but not the other. 
As we will see later, this is related to the fact that the set `Major` is "smaller" than the set `StudentID`. 

Here is another example, where we _can_ get the identity function on students. 
Let's consider the function `students_to_grades` that we defined above, which maps students to their highest grade. 
Next, let's define a function `grades_to_students` which maps grades to students. 
This is not necessarily a well-defined function in all cases, since multiple students could have the same grade. However, we can do this in this case by simply flipping the columns of the `students_to_grades` table:

```{code-cell}
grades_to_students = students_to_grades[['Grade', 'StudentID']]
grades_to_students
```

Now, let's compose these function to get `grades_to_students` $\circ$ `students_to_grades`, and see if it gives us the identity function on `StudentID`:

```{code-cell}
students_to_students3 = students_to_grades.merge(grades_to_students, how="left", on="Grade").drop("Grade", axis=1)
students_to_students3
```

Finally, this gave us an identity function on `StudentID`! 
However, we should be entirely surprised by this: indeed, this worked because each student happened to have a _unique_ highest grade, which meant this grade could be uniquely assigned to each student, and thus we can translate between grades and students without losing any information. 
We can see this is true by flipping the composition in the other direction, and looking at the function `students_to_grades` $\circ$ `grades_to_students`:

```{code-cell}
grades_to_grades = students_to_grades.merge(grades_to_students, how="left", on="StudentID").drop("StudentID", axis=1)
grades_to_grades
```

Indeed, this gives us the identity on (a subset of) the set `Grade`. 
Which subset in particular is this? 
It is simply _range_ of our function `students_to_grades`. 


## Idempotent/projection functions

An idempotent function, also known as a projection function, is a function that yields the same answer when it is applied twice as when it is applied once. 
In other words, a function $f:A\to A$ is _idempotent_ if for all $a\in A$, $f(a) = f(f(a))$. 

We can illustrate it with basic set operations (as last time, with set union and intersection), and we can illustrate it with more general functions with a data frame.

Let's see a simple example. 
Consider the function $f$ mapping `StudentID` to `StudentID`, defined in the following way. 
For each student, the function $f$ returns the student in his/her major with the highest average grade. 
Let's compute this function using python. 
First, we define a new data frame with each student's average grades:

```{code-cell}
students_avg_grades = students.groupby("StudentID", as_index=False).mean('Grade')
majors = students.groupby("StudentID", as_index=False)[['StudentID', 'Major']].first()
students_avg_grades['Major'] = majors['Major']
students_avg_grades
```

Next, we can compute the function $f$ from  `StudentID` to `StudentID` defined above.

```{code-cell}
max_grade_by_major = students_avg_grades.groupby("Major", as_index=False)[["Major", "StudentID"]].max("Grade")
stdnt_to_stdnt_highest_grade = students_avg_grades.merge(max_grade_by_major, how="left", on="Major")[['StudentID_x', 'StudentID_y', 'Major']]
stdnt_to_stdnt_highest_grade.columns = ['StudentID', 'f(StudentID)', 'Major']
stdnt_to_stdnt_highest_grade[['StudentID', 'f(StudentID)']]
```

Note that we did a bit of clean up in the cell above to make sure that the column names were interpretable (Pandas will automatically assign new column names when we join two tables with the same column name). 

Now, let's use another join to compute $f(f(\text{StudentID}))$.

```{code-cell}
stdnt_to_stdnt_highest_grade2 = stdnt_to_stdnt_highest_grade.merge(max_grade_by_major, how="left", on="Major")[['StudentID_x', 'f(StudentID)', 'StudentID_y']]
stdnt_to_stdnt_highest_grade2.columns = ['StudentID', 'f(StudentID)', 'f(f(StudentID))']
stdnt_to_stdnt_highest_grade2
```

Indeed, we see that the columns `f(StudentID)` and `f(f(StudentID))` are equal, and so the function $f$ is in fact idempotent.

An interesting property of idempotent function is that they are always equal to the identity function on their image. 
This make sense:  if $b \in \text{Image}(f)$, then $b = f(a)$ for some $a\in A$, and thus $b = f(a) = f(f(a)) = f(b)$. 
Let's verify this with code, by restricting the function $f$ to its image.

```{code-cell}
stdnt_to_stdnt_highest_grade[stdnt_to_stdnt_highest_grade['StudentID'].isin(stdnt_to_stdnt_highest_grade['f(StudentID)'])][['StudentID', 'f(StudentID)']]
```

As expected, when we restrict $f$ to $\text{Image}(f)$ we do indeed get the identity function back.


