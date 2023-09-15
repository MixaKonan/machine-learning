import numpy as np

def reject_outliers(features_values: list[list], y: list, upper_percentile: int, lower_percentile: int, outlierConstant: float) -> (list[list], list):
    feature_amount = len(features_values[0])
    new_x_set = []
    new_y = y
    for current_feature_index in range(feature_amount):
        current_feature_values = [features[current_feature_index] for _, features in enumerate(features_values)]
        upper_quartile = np.percentile(current_feature_values[1], upper_percentile)
        lower_quartile = np.percentile(current_feature_values[1], lower_percentile)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

        indices_to_delete = []
        for index, feature_value in enumerate(current_feature_values):
            if not (feature_value >= quartileSet[0] and feature_value <= quartileSet[1]):
                indices_to_delete.append(index)

        new_x_set.append(np.delete(features_values, indices_to_delete))
        new_y = np.delete(new_y, indices_to_delete)

    return new_x_set, new_y

def lr_gradient_descent(m_now, b_now, x_values, y_values,  learning_rate):
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
