import numpy as np

def reject_outliers(features_table: np.ndarray[np.ndarray], values_to_be_predicted: np.ndarray, upper_percentile: int, lower_percentile: int, outlierConstant: float) -> (np.ndarray[np.ndarray], np.ndarray):
    columns = len(features_table[0])
    for column in range(columns):
        column_values = [row[column] for row in features_table]
        upper_quartile = np.percentile(column_values, upper_percentile)
        lower_quartile = np.percentile(column_values, lower_percentile)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        low, high = (lower_quartile - IQR, upper_quartile + IQR)

        indices_to_delete = []
        for index, value in enumerate(column_values):
            if not (value >= low and value <= high):
                indices_to_delete.append(index)
        
        features_table = np.delete(features_table, indices_to_delete, 0)
        values_to_be_predicted = np.delete(values_to_be_predicted, indices_to_delete, 0)
        

    return features_table, values_to_be_predicted

def gradient_descent(m_now, b_now, x_values, y_values,  learning_rate):
    if len(x_values) != len(y_values):
        raise ValueError('X and Y values amounts must be equal.')

    m_gradient = 0
    b_gradient = 0
    n = len(x_values)

    for i in range(n):
        x = x_values[i]
        y = y_values[i]

        m_gradient += -(2/n) * (x * (y - (m_now * x + b_now)))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate
    loss_value = lr_loss_function(x_values[0], y_values[0], m, b)

    return m, b, loss_value

def lr_loss_function(x, y, m, b):
    return (y - (m * x + b)) ** 2
