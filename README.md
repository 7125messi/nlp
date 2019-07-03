# nlp

# ChineseTextClassifier
中文商品评论短文本分类器

### **运行环境：**

tensorflow2.0
python3



### 数据集：

京东商城评论文本，10万条，标注为0的是差评，标注为1的是好评。
路径：`data/goods_zh.txt`



### **已实现的模型：**

1. Transfromer
2. word2vec+textCNN
3. fastext
4. word2vec+LSTM/GRU
5. word2vec+LSTM/GRU+Attention
6. word2vec+Bi_LSTM+Attention
