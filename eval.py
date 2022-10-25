import math
import numpy as np
from language import *
import traceback


def EvalPt(expr: str, x: np.ndarray) -> np.ndarray:
    stack = []
    orig_expr = expr
    i = 0
    x1 = 0
    x2 = 0
    k = i
    try:
        while i < len(expr):
            symbol = expr[i]
            i += 1
            k += 1
            if symbol not in non_literals:  # evaluate
                if symbol in constants:
                    if symbol.isalpha():
                        if symbol == "p":
                            stack.append(math.pi)
                        else:
                            stack.append(math.e)
                    else:
                        stack.append(int(symbol))
                else:
                    first = expr.index("{")
                    last = expr.index("}")
                    stack.append(x.item(int(expr[first+1:last])))
                    expr = expr.replace('{', '', 1)
                    expr = expr.replace('}', '', 1)
                    i += last-first-1
                    k = i+2
            else:
                if symbol in binary:
                    x1 = stack.pop()
                    x2 = stack.pop()
                    if (type(x1) is complex) or (type(x2) is complex):
                        return np.append(x, np.nan)
                    x1 = float(x1)
                    x2 = float(x2)
                    if symbol == "+":
                        stack.append(x1 + x2)
                    elif symbol == "-":
                        stack.append(x1 - x2)
                    elif symbol == "*":
                        stack.append(x1 * x2)
                    elif symbol == "/":
                        if (x2 == 0):
                            return np.append(x, np.nan)
                        stack.append(x1 / x2)
                    elif symbol == "^":
                        if (abs(x2) >= 20) or (x1 == 0 and x2 <= 0):
                            return np.append(x, np.nan)
                        stack.append(x1 ** x2)
                else:
                    q = stack.pop()
                    if type(q) is complex:
                        return np.append(x, np.nan)
                    q = float(q)
                    if symbol == "~":
                        stack.append(q * -1)
                    elif symbol == ">":
                        stack.append(q + 1)
                    elif symbol == "<":
                        stack.append(q - 1)
                    elif symbol == "S":
                        stack.append(math.sin(q))
                    elif symbol == "C":
                        stack.append(math.cos(q))
                    elif symbol == "s":
                        if q < -1 or q > 1:
                            return np.append(x, np.nan)
                        stack.append(math.asin(q))
                    elif symbol == "c":
                        if q < -1 or q > 1:
                            return np.append(x, np.nan)
                        stack.append(math.acos(q))
                    elif symbol == "t":
                        if q <= -(math.pi/2) or q >= (math.pi/2):
                            return np.append(x, np.nan)
                        stack.append(math.atan(q))
                    elif symbol == "A":
                            return np.append(x, np.nan)
                    elif symbol == "L":
                        if q <= 0:
                            return np.append(x, np.nan)
                        stack.append(math.log(q))
        q = stack.pop()
        if (type(q) is complex):
            return np.append(x, np.nan)
        return np.append(x, float(q))
    except Exception:
        traceback.print_exc()
        print(f"Stack: {stack}\nSymbol: {symbol}\n" +
              f"expr={orig_expr}\n{' '*(k+4) + '^'}\
              \nEval:{str(x1) + str(symbol) + str(x2)}\n\n\n")
        return np.append(x, np.nan)


def Eval(expr: str, x: np.ndarray) -> np.ndarray:
    newarr = np.array([])
    i=0
    for xi in x:
        i+=1
        x2 = EvalPt(expr, xi)
        newarr = np.append(newarr, x2)
        if np.isnan(x2[len(x2)-1]):
            s = list(x.shape)
            s[1] += 1
            i = len(newarr)/s[1]
            s[0] = int(i)
            newarr.shape = s
            return newarr
    s = list(x.shape)
    s[1] += 1
    newarr.shape = s
    return newarr
