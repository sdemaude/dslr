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
    pos = (n - 1) / 4
    lower = int(pos)
    upper = lower + 1

    if pos == lower:
        return column.iloc[int(pos)]
    else:
        # Interpolate between the two closest values
        weight = pos - lower
        return (1 - weight) * column.iloc[lower] + weight * column.iloc[upper]

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
    pos = (n - 1) / 4 * 3
    lower = int(pos)
    upper = lower + 1

    if pos == lower:
        return column.iloc[int(pos)]
    else:
        weight = pos - lower
        return (1 - weight) * column.iloc[lower] + weight * column.iloc[upper]

def ft_var(column):
    # Subtract the mean from each value and square the result
    squared_differences = (column - ft_mean(column)) ** 2
    variance = squared_differences.sum() / (len(column) - 1)
    return variance

def ft_skew(column):
    n = len(column)
    mean = ft_mean(column)
    std = ft_std(column)
    skewness = (n * ((column - mean) ** 3).sum()) / ((std ** 3) * (n - 1) * (n - 2))
    return skewness

def ft_kurt(column):
    return column.kurtosis()

def ft_range(column):
    return ft_max(column) - ft_min(column)