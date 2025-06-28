import sys
import pandas as pd
import calculation as calc

def format_column_name(name):
    truncated = name[:10]
    if len(name) > 10:
        truncated = truncated.rstrip() + '...'
    return truncated 

def stats_computation(column):
    stats = {}

    stats['count'] = calc.ft_count(column)
    stats['mean'] = calc.ft_mean(column)
    stats['std'] = calc.ft_std(column)
    stats['min'] = calc.ft_min(column)
    stats['25%'] = calc.ft_25(column)
    stats['50%'] = calc.ft_50(column)
    stats['75%'] = calc.ft_75(column)
    stats['max'] = calc.ft_max(column)

    stats['var'] = calc.ft_var(column)
    stats['skew'] = calc.ft_skew(column)
    stats['kurt'] = calc.ft_kurt(column)
    stats['range'] = calc.ft_range(column)

    return stats

def ft_describe(data_frame):

    stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'var', 'skew', 'kurt', 'range']
    results = {}

    for column_name in data_frame.columns:
        column = data_frame[column_name].dropna()
        column = column.sort_values()
        stats = stats_computation(column)
        results[column_name] = list(stats.values())
    
    result_df = pd.DataFrame(results, index=stats)
    return result_df

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('usage: describe.py [input_file].csv [-d]', file=sys.stderr)
    else :
        if len(sys.argv) == 3 and sys.argv[2] != '-d':
            print('usage: describe.py [input_file].csv [-d]', file=sys.stderr)
            return

        try: 
            csv_file = sys.argv[1]
            df = pd.read_csv(csv_file)        
        except:
            print('error: could not read .csv file', file=sys.stderr)
            return

        df = df.rename(columns=format_column_name)

        df_numeric = df.select_dtypes(include=['float']).dropna(axis=1, how='all')

        print('\nft_describe function:\n')  
        print(ft_describe(df_numeric))

        if len(sys.argv) == 3:
            true_describe = df_numeric.describe()

            true_describe.loc['var'] = df_numeric.var()
            true_describe.loc['skew'] = df_numeric.skew()
            true_describe.loc['kurt'] = df_numeric.kurt()
            true_describe.loc['range'] = df_numeric.max() - df_numeric.min()

            print('\npandas describe function:\n')
            print(true_describe)

if __name__ == '__main__':
    main()