class Config(object):
    # 数据集路径
    dataSource = '../data/goods_zh.txt'
    stopWordSource = '../data/stopword.txt'

    # 分词后保留大于等于最低词频的词
    miniFreq = 1

    # 统一输入文本序列的定长，取了所有序列长度的均值。超出将被截断，不足则补0
    sequenceLength = 200
    batchSize = 64
    epochs = 5

    numClasses = 2
    # 训练集的比例
    rate = 0.8

    # 生成嵌入词向量的维度
    embeddingSize = 150

    # 卷积核数
    numFilters = 30

    # 卷积核大小
    filterSizes = [2, 3, 4, 5]
    dropoutKeepProb = 0.5

    # L2正则系数
    l2RegLambda = 0.1