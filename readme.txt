https://blog.csdn.net/a1ccwt/article/details/153263761?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%BC%80%E6%BA%90%E6%95%85%E9%9A%9C%E8%AF%8A%E6%96%AD%E9%A1%B9%E7%9B%AE&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-153263761.142^v102^pc_search_result_base2&spm=1018.2226.3001.4187

1.load_data.ipynb - 把原始数据载入的。
use.py

2.pre_process_data.ipynb - 读入数据、筛选指定工况、滑动窗口采样、保存为npz格式的数据集
make_data.py已整理

3.exact_feature.ipynb - 提取时域、频域、时频域特征（简易版）
#后续优化-把时频特征加进去

4.classic_ml.ipynb - 加载经典模型的训练结果。
#11.22优化-加载异样本的训练和测试集合,对比了实验


