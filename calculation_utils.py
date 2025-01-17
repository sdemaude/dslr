import math
# Custom calculation functions for a given Pandas Serie(column)

def ft_count(column):
    return column.count()
        
def ft_mean(column):
    col_data = column.dropna()
    sum = 0
    for i in range(len(col_data)):
        sum += col_data.iloc[i]
    return sum / len(col_data)

def ft_std(column):
    #return math.sqrt(ft_var(column))
    return column.std()

def ft_min(column):
    col_data = column.dropna()
    min = col_data.iloc[0]
    for i in range(len(col_data)):
        if col_data.iloc[i] < min:
            min = col_data.iloc[i]
    return min

def ft_max(column):
    col_data = column.dropna()
    max = col_data.iloc[0]
    for i in range(len(col_data)):
        if col_data.iloc[i] > max:
            max = col_data.iloc[i]
    return max
        
def ft_25(column):
    return column.quantile(0.25)

def ft_50(column):
    return column.median()

def ft_75(column):
    return column.quantile(0.75)

def ft_var(column):
    return column.var()

def ft_skew(column):
    return column.skew()

def ft_kurt(column):
    return column.kurt()

def ft_range(column):
    return ft_max(column) - ft_min(column)