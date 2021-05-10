import pandas as pd
import numpy as np
import scipy.stats


def data_init():
    """
    初始化数据和属性集。

    Returns:
        train_set (list): 训练集
        test_set (list): 测试集
        attributes (list): 属性集
        is_discrete (list): 属性是否离散
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


def generate_dataframe(data: list, attributes: list, one_dimensional: bool = False):
    """
    将数据集转化为DataFrame类型。

    Args:
        data (list): 数据集
        attributes (list): 属性集
        one_dimensional (bool = False): 是否是一维列表，用于处理只有单个样例的集合，例如is_discrete

    Returns:
        frame (pd.DataFrame): 数据集
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
    初始化概率表，行标签为类别，列标签为属性，表格内数据为概率。

    Args:
        data (pd.DataFrame): 训练集

    Returns:
        frame (pd.DataFrame): 概率表
    """
    col = data.columns
    row_count = data['类型'].value_counts()
    row_number = row_count.shape[0]
    zero_data = np.zeros((row_number, len(col)))
    frame = pd.DataFrame(zero_data, columns=col, index=row_count.index)
    return frame


def calculate_type_prior_probability(data: pd.DataFrame):
    """
    计算类先验概率，包含拉普拉斯修正。

    Args:
        data (pd.DataFrame): 训练集

    Returns:
        tpp (pd.Series): 类先验概率，index为类别
    """
    counts = data['类型'].value_counts()
    tpp = pd.Series(index=counts.index, dtype='float64')
    for i in tpp.index:
        tpp[i] = (counts[i] + 1) / (counts.sum() + counts.size)
    return tpp


def train_naive_bayes(train: pd.DataFrame, test: pd.DataFrame, is_discrete_frame: pd.DataFrame):
    """
    训练朴素贝叶斯分类器，并对测试集进行判别

    Args:
        train (pd.DataFrame): 训练集
        test (pd.DataFrame): 测试集
        is_discrete_frame (pd.DataFrame): 属性是否离散

    Returns:
        result_frames (pd.DataFrame): 概率表
    """
    # TODO:待优化。拆分成训练和判别两个函数，训练部分对所有属性所有取值计算，存储在使用MultiIndex的DataFrame中，避免重复计算
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
    """
    计算朴素贝叶斯概率的连乘结果，并返回分类判别结果

    Args:
        result_frames ('list[pd.DataFrame]'): 概率表，在本函数中添加一个'概率'列

    Returns:
        judge_results (list): 分类判别结果
    """
    judge_results = []
    for result_frame in result_frames:
        result_frame['概率'] = result_frame.prod(axis=1)
        judge_result = result_frame['概率'].idxmax()
        judge_results.append(judge_result)
    return judge_results


def split_data_set(data: list, attributes: list):
    """
    将带标签的数据集分为属性（无标签）和标签两部分。

    Args:
        data (list): 数据集
        attributes (list): 属性集

    Returns:
        data_without_type (pd.DataFrame): 属性（无标签）
        category_label (list): 标签
    """
    data_frame = pd.DataFrame(data)
    data_without_type = data_frame.iloc[:, 0:data_frame.shape[1] - 1]
    data_without_type.columns = attributes
    category_label = data_frame.iloc[:, data_frame.shape[1] - 1].tolist()
    return data_without_type, category_label


def calculate_precision(train: pd.DataFrame, test: pd.DataFrame, is_discrete_frame: pd.DataFrame,
                        category_label: list):
    """
    使用朴素贝叶斯对测试集分类，并计算精度。

    Args:
        train (pd.DataFrame): 训练集
        test (pd.DataFrame): 测试集（不带标签）
        is_discrete_frame (pd.DataFrame): 属性是否离散
        category_label (list): 测试集标签

    Returns:
        precision_of_classify (float): 精度
    """
    result_frames = train_naive_bayes(train, test, is_discrete_frame)
    judge_results = judge_naive_bayes(result_frames)
    print('原始标签:\n', category_label)
    print('测试集判别结果:\n', judge_results)
    wrong_count = 0
    for i in range(len(category_label)):
        if judge_results[i] != category_label[i]:
            wrong_count += 1
            print('------------------------------')
            print('第{}个数据预测错误，数据如下：\n'.format(i), train.loc[i], '\n\n朴素贝叶斯概率如下：\n', result_frames[i], sep='')
    print('------------------------------')
    precision_of_classify = 1 - wrong_count / len(category_label)
    return precision_of_classify


if __name__ == '__main__':
    # 设置ipython中pandas表格显示宽度
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    # 用测1样本判别
    train_set_array, test_set_array, attributes, is_discrete = data_init()
    train_set = generate_dataframe(train_set_array, attributes)
    test_set = generate_dataframe(test_set_array, attributes)
    is_discrete_frame = generate_dataframe(is_discrete, attributes, True)
    result_frames = train_naive_bayes(train_set, test_set, is_discrete_frame)
    judge_results = judge_naive_bayes(result_frames)
    print("测1判别结果：", judge_results)
    # 用训练集判别
    data_without_type, category_label = split_data_set(train_set_array, attributes)
    precision = calculate_precision(train_set, data_without_type, is_discrete_frame, category_label)
    print("精度:", precision)
