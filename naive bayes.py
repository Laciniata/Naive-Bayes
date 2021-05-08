import pandas as pd
import numpy as np
import scipy.stats


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
    test_set = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]
                ]
    attributes = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
    is_discrete = [True, True, True, True, True, True, False, False]
    return train_set, test_set, attributes, is_discrete


# tested
def generate_dataframe(data: list, attributes: list, one_dimensional: bool = False):
    """
    将数据集转化为DataFrame类型。

    Args:
        data (list): 数据集
        attributes (list): 属性集

    Returns: pd.DataFrame
    """
    column = attributes.copy()
    if one_dimensional:
        if len(data) == len(attributes) + 1:
            column.append('类型')
        frame = pd.DataFrame([data], columns=column)
    else:
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
    col = data.columns
    row_count = data['类型'].value_counts()
    row_number = row_count.shape[0]
    zero_data = np.zeros((row_number, len(col)))
    frame = pd.DataFrame(zero_data, columns=col, index=row_count.index)
    return frame


def calculate_type_prior_probability(data: pd.DataFrame):
    counts = data['类型'].value_counts()
    tpp = pd.Series(index=counts.index, dtype='float64')
    for i in tpp.index:
        tpp[i] = (counts[i] + 1) / (counts.sum() + counts.size)
    return tpp


def train_naive_bayes(train: pd.DataFrame, test: pd.DataFrame, is_discrete_frame):
    result_frames = []
    result_frame = get_result_frame(train)
    types = result_frame.index.tolist()
    type_prior_probability = calculate_type_prior_probability(train)
    for i, test_sample in test.iterrows():
        result_frame_temp = result_frame.copy(deep=True)
        for attribute in test.columns:
            value = test_sample[attribute]
            # 对每种类别，计算概率
            # 离散情况
            if is_discrete_frame.loc[0, attribute]:
                for t in types:
                    # 获得该类型的train视图，再获得该类型下该取值的视图，二者行数求比值得到概率
                    frame_of_type = train[train['类型'] == t]
                    frame_of_type_and_value = frame_of_type[frame_of_type[attribute] == value]
                    probability = (frame_of_type_and_value.shape[0] + 1) / (
                            frame_of_type.shape[0] + train[attribute].unique().size)  # 拉普拉斯修正
                    result_frame_temp.loc[t, attribute] = probability
            # 连续情况
            else:
                for t in types:
                    # 统计平均值和方差，代入正态分布
                    frame_of_type = train[train['类型'] == t]
                    mean = frame_of_type.loc[:, attribute].mean()
                    std = frame_of_type.loc[:, attribute].std()
                    probability = scipy.stats.norm(mean, std).pdf(value)
                    result_frame_temp.loc[t, attribute] = probability
        result_frame_temp['类型'] = type_prior_probability  # 类先验概率
        result_frames.append(result_frame_temp)
    return result_frames


def judge_naive_bayes(result_frames: 'list[pd.DataFrame]'):
    judge_results = []
    for result_frame in result_frames:
        result_frame['概率'] = result_frame.prod(axis=1)
        judge_result = result_frame['概率'].idxmax()
        judge_results.append(judge_result)
    return judge_results


if __name__ == '__main__':
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)

    train_set_array, test_set_array, attributes, is_discrete = data_init()
    train_set = generate_dataframe(train_set_array, attributes)
    test_set = generate_dataframe(test_set_array, attributes)
    is_discrete_frame = generate_dataframe(is_discrete, attributes, True)
    result_frames = train_naive_bayes(train_set, test_set, is_discrete_frame)
    judge_results = judge_naive_bayes(result_frames)
    print(judge_results)
