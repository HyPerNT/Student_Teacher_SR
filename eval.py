"""RPN Calculator for SR implementations

Author: Brenton Candelaria

The aim of this file is to provide functionality in interpreting strings of text representing math functions in RPN and provide numerical
evaluations as an output. This file provides two functions that both work over NumPy arrays: Eval(), and EvalPt(). Both expect a string
to be evaluated, but also an array to evaluate on. EvalPt() expects a vector to parse and evaluate, and Eval() works over an array of such
vectors, deferring to EvalPt() for evaluation.

In implementation, complex results/those that occur outside of a function's domain (such as lg(0)) are returned as np.nan from EvalPt()
This behavior can be changed in the future, but suffices to tell the user that the computation could not be computed gracefully.
"""

import math
import numpy as np
from language import * # Utilize the basic context-free grammar for math
import traceback # For some light error-logging, in the worst case


def EvalPt(expr: str, x: np.ndarray) -> np.ndarray:
    """Evaluates a unit vector x using the expression str

        Parameters
        ----------
            expr : A (syntactically correct) expression to be parsed in RPN, like "X{0}X{1}+"
            x : A numpy array that corresponds to a vector, each component being a scalar variable in the function

        Returns
        -------
            np.ndarray : An output vector of length len(x)+1, appending the result to the end
                May append np.nan on failed computations
    """
    stack = [] # Tracks numbers on the stack
    orig_expr = expr # Track original expr, since we overwrite it during parsing
    x1 = 0 # First arg, if any
    x2 = 0 # Second arg, if any
    i = 0 # Tracks position in overall expression, after edits (dropping braces during parsing)
    k = i # Tracks position in original expression, sans edits

    # Surround in try-catch, we might be handed invalid syntax and cannot handle it otherwise
    try:
        # Parse to end of the expr
        while i < len(expr):
            symbol = expr[i] # Get next symbol to process
            i += 1
            k += 1 
            # Inc i and k since we're now processing it

            if symbol not in non_literals:  # Evaluate vars, constants
                if symbol in constants: # Eval constants (e, pi, 0, 1)
                    if symbol.isalpha(): # On letter constants, append their value to the stack
                        if symbol == "p":
                            stack.append(math.pi)
                        else:
                            stack.append(math.e)
                    else: # Append the constant (0 or 1) to the stack
                        stack.append(int(symbol))
                else: # Encountered a variable, process it
                    first = expr.index("{")
                    last = expr.index("}")
                    stack.append(x.item(int(expr[first+1:last]))) # Extract the indexed var in the expression, add to stack
                    expr = expr.replace('{', '', 1)
                    expr = expr.replace('}', '', 1) # Replace braces in expr so next instance of braces can be correctly found
                    i += last-first-1
                    k = i+2
                    # Update i to reflect new position in expr, k to reflect updated position in orig_expr
            
            else: # On unary or binary operators
                if symbol in binary:
                    # Get args
                    x1 = stack.pop()
                    x2 = stack.pop()

                    # Pre-check type of args in case sh*t hit the fan and we haven't caught it yet
                    if (type(x1) is complex) or (type(x2) is complex):
                        return np.append(x, np.nan)

                    # Conv to float and work with it
                    x1 = float(x1)
                    x2 = float(x2)

                    # "Good code documents itself"
                    if symbol == "+":
                        stack.append(x1 + x2)
                    elif symbol == "-":
                        stack.append(x1 - x2)
                    elif symbol == "*":
                        stack.append(x1 * x2)
                    elif symbol == "/":
                        # Check for div by 0
                        if (x2 == 0):
                            return np.append(x, np.nan)
                        stack.append(x1 / x2)
                    elif symbol == "^":
                        # Prevent the CPU from cooking itself
                        if (abs(x2) >= 20) or (x1 == 0 and x2 <= 0) or (x1 < 0):
                            return np.append(x, np.nan)
                        stack.append(x1 ** x2)

                else: # Unary operators
                    q = stack.pop()

                    # Same pre-check
                    if type(q) is complex:
                        return np.append(x, np.nan)

                    q = float(q)

                    # "Good code documents itself"
                    if symbol == "~":
                        stack.append(q * -1)
                    elif symbol == ">":
                        stack.append(q + 1)
                    elif symbol == "<":
                        stack.append(q - 1)
                    elif symbol == "S":
                        stack.append(float(math.sin(q)))
                    elif symbol == "C":
                        stack.append(math.cos(q))

                    # Check domains on arctrig fns, ln
                    elif symbol == "s":
                        if q < -1 or q > 1:
                            return np.append(x, np.nan)
                        stack.append(math.asin(q))
                    elif symbol == "c":
                        if q < -1 or q > 1:
                            return np.append(x, np.nan)
                        stack.append(math.acos(q))
                    elif symbol == "t":
                        stack.append(math.atan(q))
                    elif symbol == "A":
                        stack.append(abs(q))
                    elif symbol == "L":
                        if q <= 0:
                            return np.append(x, np.nan)
                        stack.append(math.log(q))
        
        # Finished parsing the fn! Get the result (might err here if there's nothing to pop)
        q = stack.pop()
        if (type(q) is complex):
            return np.append(x, np.nan)

        return np.append(x, float(q))

    except Exception:
        # Lazy error handling, give stack trace and also show where in the fn we went wrong
        traceback.print_exc()
        print(f"Stack: {stack}\nSymbol: {symbol}\n" +
              f"expr={orig_expr}\n{' '*(k+4) + '^'}\
              \nEval:{str(x1) + str(symbol) + str(x2)}\n\n\n")

        # A wrong answer is better than no answer
        return np.append(x, np.nan)


def Eval(expr: str, x: np.ndarray) -> np.ndarray:
    """Evaluate an array of pts on the same expression, returning the full array with the corresponding results.

        Parameters
        ----------
            expr : A (syntactically correct) expression to be parsed in RPN, like "X{0}X{1}+"
            x : A numpy array that corresponds to a list of vectors, like those accepted by EvalPt()

        Returns
        -------
            np.ndarray : An output array with shape (n, m+1) with appended column of results
    """
    newarr = np.array([])
    i = 0
    for index in range(len(x)):
        i += 1
        x2 = EvalPt(expr, x[index]) # Get ans from EvalPt
        newarr = np.append(newarr, x2) # Append
    s = list(x.shape)
    s[1] += 1
    newarr.shape = s # Correct shape to "look like" the old arr
    return newarr
