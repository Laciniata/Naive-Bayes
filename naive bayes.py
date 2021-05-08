import pandas as pd
import numpy as np


# tested
def data_init():
    """
    初始化数据和属性集。

    Returns: (数据, 属性集)
    """

    train_set = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
                 ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
                 ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
                 ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
                 ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
                 ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
                 ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
                 ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
                 ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
                 ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
                 ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
                 ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
                 ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
                 ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
                 ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
                 ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
                 ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
                 ]
    test_set = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]]
    attributes = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
    is_discrete = [True, True, True, True, True, True, False, False]
    return train_set, test_set, attributes, is_discrete


# tested
def generate_dataframe(data: list, attributes: list):
    """
    将数据集转化为DataFrame类型。

    Args:
        data (list): 数据集
        attributes (list): 属性集

    Returns: pd.DataFrame
    """
    column = attributes.copy()
    if len(data[0]) == len(attributes) + 1:
        column.append('类型')
    frame = pd.DataFrame(data, columns=column)
    return frame


def get_result_frame(data: pd.DataFrame):
    """

    Args:
        data ():

    Returns:

    """
    col = data.columns.tolist()
    col.remove('类型')
    row_count = data['类型'].value_counts()
    row_number = row_count.shape[0]
    zero_data = np.zeros((row_number, len(col)))
    frame = pd.DataFrame(zero_data, columns=col, index=row_count.index)
    return frame


def train_naive_bayes(train: pd.DataFrame, test: pd.DataFrame):
    result = get_result_frame(train)
    for attribute in test.columns:
        value = test.iloc[0][attribute]
        # probability =


if __name__ == '__main__':
    train_set_array, test_set_array, attributes, is_discrete = data_init()
    train_set = generate_dataframe(train_set_array, attributes)
    test_set = generate_dataframe(test_set_array, attributes)
    train_naive_bayes(train_set, test_set)
