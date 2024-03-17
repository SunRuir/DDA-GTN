
"""
This code is a function that calculates the sum and mean of a list.
It is used in cross validation to calculate the final evaluation metrics for n-fold cross validation.
"""


import torch

def average_list(list_input):
    average = 0
    for i in range(len(list_input)):
        average = (average * i + list_input[i]) / (i + 1)
    return average



def sum_list(list_input):
    summ = 0
    for i in range(len(list_input)):
        summ = summ + list_input[i]
    return summ