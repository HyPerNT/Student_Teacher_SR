from utils import *
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

def isCorrect(x, y):
    return (x and y > 0) or (not x and y < 0)

def print_summary(df):
    anomalies = df.loc[df['anomaly'] == -1]
    mistakes = df.loc[df['correct'] == False]
    false_pos = mistakes.loc[mistakes['anomaly'] == -1]
    false_neg = mistakes.loc[mistakes['anomaly'] == 1]

    print(f'False Negative Rate: {false_neg.size/df.size :.5f}\nFalse Positive Rate: {false_pos.size/df.size :.5f}')
    print(f'Accuracy: {100 * (1 - mistakes.size/df.size):.3f}%\n')
    print(f'Pre-Scrub outlier rate: {len(df[df.truth == False])/len(df) :.5f}, {len(df)} Data Points')
    cleaned_df = pd.merge(df, anomalies, indicator=True, how='left').query('_merge=="left_only"').drop('_merge', axis=1)
    print(f'Post-Scrub outlier rate: {len(cleaned_df[cleaned_df.truth == False])/len(cleaned_df) :.5f}, {len(cleaned_df)} Data Points')
    mistakes = cleaned_df.loc[cleaned_df['correct'] == False]
    false_pos = mistakes.loc[mistakes['anomaly'] == -1]
    false_neg = mistakes.loc[mistakes['anomaly'] == 1]
    print(f'Percent of clean data after scrub: {100 * (1 - mistakes.size/cleaned_df.size):.3f}%\n')
    return cleaned_df

def fn_to_df(fn):
    q = np.reshape(fn.y_train, (-1, 1))
    tmp = np.append(fn.x_train, q, axis=1)
    predictors = ['x'+str(i) for i in range(fn.dim)]
    cols = predictors + ['y']
    df = pd.DataFrame(tmp, columns=cols)
    mask = np.equal(fn.y_train, np.array([eval.EvalPt(fn.fn, x)[-1] for x in fn.x_train]))
    return df, mask

def main():
    outlier_rate = 0.1
    fn = mystery_function("x{0}S", 1, gen_data=True, sample_size=10_000, scale=10, center=5, outlier_rate=outlier_rate)
    df, mask = fn_to_df(fn)
    print(f'Number of bad data points: {len(mask)-np.sum(mask)}')

    cols = df.columns
    model = IsolationForest()
    model.fit(df[cols])

    df['anomaly']=model.predict(df[cols])
    df['correct']=[True if isCorrect(mask[i], df['anomaly'][i]) else False for i in range(len(mask))]
    df['truth']=[x for x in mask]
    print_summary(df)


    fn = mystery_function("x{0}S", 1, gen_data=True, sample_size=10_000, scale=10, center=5, outlier_rate=2*outlier_rate)
    df, mask = fn_to_df(fn)

    df['anomaly']=model.predict(df[cols])
    df['correct']=[True if isCorrect(mask[i], df['anomaly'][i]) else False for i in range(len(mask))]
    df['truth']=[x for x in mask]
    print_summary(df)

if __name__ == "__main__":
    main()