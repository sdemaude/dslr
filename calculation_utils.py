import math

# Custom calculation functions for a given Pandas Serie(column)

def ft_count(column):
    return len(column)
        
def ft_mean(column):
    sum = 0
    for i in range(len(column)):
        sum += column.iloc[i]
    return sum / len(column)

def ft_std(column):
    return math.sqrt(ft_var(column))

def ft_min(column):
    return column.iloc[0]

def ft_max(column):
    return column.iloc[-1]
        
def ft_25(column):
    n = len(column)
    pos = n // 4
    if n % 4 != 0:
        median = column.iloc[pos]
    else:
        median = (column.iloc[pos - 1] + column.iloc[pos]) / 2
    return median

def ft_50(column):
    n = len(column)
    pos = n // 2
    if n % 2 != 0:
        median = column.iloc[pos]
    else:        
        median = (column.iloc[pos - 1] + column.iloc[pos]) / 2
    return median

def ft_75(column):
    n = len(column)
    pos = n // 4 * 3
    if n % 4 != 0:
        median = column.iloc[pos]
    else:
        median = (column.iloc[pos - 1] + column.iloc[pos]) / 2
    return median

def ft_var(column):
    # Subtract the mean from each value and square the result
    squared_differences = (column - ft_mean(column)) ** 2
    # Calculate the variance
    variance = squared_differences.sum() / len(column)
    return variance

def ft_skew(column):
    return column.skew()

def ft_kurt(column):
    return column.kurt()

def ft_range(column):
    return ft_max(column) - ft_min(column)