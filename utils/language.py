"""Definitions for different atoms in our language"""

unary = ['~', 'A', '>', '<']
trig = ['S', 'C']
inverse_trig = ['s', 'c', 't']
log = ['L']
extended_unary = unary + trig + inverse_trig + log
binary = ['+', '-', '*', '/', '^']
constants = ['1', '0', 'p', 'e']
fns = extended_unary + binary
non_literals = extended_unary + binary
