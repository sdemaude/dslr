import sys
import numpy as np
import pandas as pd
import calculation_utils as calc

# Truncate column names and add a dot if the name is too long
def format_column_name(name):
    truncated = name[:10]  # Truncate to 10 characters
    if len(name) > 10:     # Check if the name was truncated
        truncated = truncated.rstrip() + '...'
    return truncated 

# Compute statistics for a given column   
def stats_computation(col_data):
    stats = {}

    stats['count'] = calc.ft_count(col_data)                    # count
    stats['mean'] = calc.ft_mean(col_data)                      # mean
    stats['std'] = calc.ft_std(col_data)                        # std
    stats['min'] = calc.ft_min(col_data)                        # min
    stats['25%'] = calc.ft_25(col_data)                         # 25th percentile
    stats['50%'] = calc.ft_50(col_data)                         # 50th percentile (median)
    stats['75%'] = calc.ft_75(col_data)                         # 75th percentile
    stats['max'] = calc.ft_max(col_data)                        # max

    stats['var'] = calc.ft_var(col_data)                        # variance
    stats['skew'] = calc.ft_skew(col_data)                      # skewness
    stats['kurt'] = calc.ft_kurt(col_data)                      # kurtosis
    stats['range'] = calc.ft_range(col_data)                    # range

    return stats

# Custom implementation of pandas.DataFrame.describe() function
def ft_describe(csv_file):
    # Read the CSV file and catch any errors
    try: 
        df = pd.read_csv(csv_file)
    except:
        print('error: could not read .csv file', file=sys.stderr)
        return

    df = df.rename(columns=lambda x: format_column_name(x))
    
    # Define statistics to calculate
    stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'var', 'skew', 'kurt', 'range']
    results = {}
    
    # Compute stats for float columns
    for column in df.select_dtypes(include=['float']).columns:
        col_data = df[column].dropna()  # Exclude NaN values
        if col_data.empty:
            results[column] = [np.nan] * len(stats) 
            continue
        stats = stats_computation(col_data)
        results[column] = list(stats.values())
    
    # Create the result DataFrame to display the stats
    result_df = pd.DataFrame(results, index=stats)
    return result_df

# Check if the user provided a CSV file
if len(sys.argv) != 2:
    print('usage: describe.py [input_file].csv', file=sys.stderr)
else :
    output = ft_describe(sys.argv[1])
    if output is not None:
        print(output)
