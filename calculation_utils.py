import numpy as np

def ft_count(column):
    return column.size

def ft_sum(column):
    sum = 0
    for value in column:
        sum += value
    return sum
        
def ft_mean(column):
    return ft_sum(column) / column.size

def ft_std(column):
    return np.sqrt(ft_var(column))

def ft_min(column):
    return column.iat[0]

def ft_max(column):
    return column.iat[-1]

def ft_quantile(column, div, part):
    (pos, rest) = np.divmod((column.size - 1) * part, div)
    if rest == 0:
        return column.iat[pos]
    return (column.iat[pos] * (div - rest) + column.iat[pos + 1] * rest) / div

def ft_25(column):
    return ft_quantile(column, 4, 1)

def ft_50(column):
    return ft_quantile(column, 2, 1)

def ft_75(column):
    return ft_quantile(column, 4, 3)

def ft_var(column):
    n = ft_count(column)

    return (1 / (n - 1)) * ft_sum((column - ft_mean(column)) ** 2)
    #return (sum_squared * n - sum**2) / (n**2 - n)

def ft_skew(column):
    n = ft_count(column)

    return n / ((n - 1) * (n - 2)) * ft_sum(((column - ft_mean(column)) / ft_std(column)) ** 3)
    #return (n**2 * sum_cubed - 3 * n * sum * sum_squared + 2 * sum**3 ) / (n * sum_squared - sum**2) ** (3/2)


def ft_kurt(column):
    n = ft_count(column)

    return n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * ft_sum(((column - ft_mean(column)) / ft_std(column)) ** 4) - 3 * (n - 1)**2 / ((n - 2) * (n - 3))
    #return (n**3 * sum_quartic - 4 * n**2 * sum * sum_cubed - 3 * n**2 * sum_squared**2 + 12 * n * sum**2 * sum_squared - 6 * sum**4 )/(n**2 * sum_squared**2 - 2 * n * sum**2 * sum_squared + sum**4 ) 
    
def ft_range(column):
    return ft_max(column) - ft_min(column)