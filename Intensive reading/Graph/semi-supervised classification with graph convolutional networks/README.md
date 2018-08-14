## semi-supervised classification with graph convolutional networks
[1-st order approximation GCN](https://arxiv.org/abs/1609.02907)
===================================================================================
    半监督分类问题：给定图中一小个带标签的集合，通过在损失函数中添加图拉普拉斯正则来将标签信息在图中平滑扩散。
    PS： 该方法基于一个基本假设，即图中相连的节点更可能共享相同的标签（该基本假设太强，图中的边可能包含一些附加的信息）。
    
    文章的两个主要贡献：
    1） 基于传统的谱图卷积一阶近似设计了一种直接在图上操作的特征逐层传播规则。
    2） 将其应用于半监督节点分类任务中。
    
    谱卷积：
    在一个图中，输入一个n维图信号x（每个图中的节点一个标量），其谱卷积由如下公式表示：
    
<div align=center><img src="https://latex.codecogs.com/gif.latex?g_{\theta&space;}\star&space;x&space;=&space;Ug_{\theta&space;}U^{\top&space;}x" title="g_{\theta }\star x = Ug_{\theta }U^{\top }x" /><\div>
  
    其中，g_{\theta}=diag(\theta)代表滤波器，\theta表示傅里叶域中的n维参数。
    U为特征分解标准图拉普拉斯矩阵（normalized graph Laplacian）所得的特征向量组成的矩阵，其中：

<div align=center><img src="https://latex.codecogs.com/gif.latex?L^{norm}=I_{n}-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}=U\Lambda&space;U^{\top}" title="L^{norm}=I_{n}-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}=U\Lambda U^{\top}" /><\div>
  
    \Lambda表示特征值矩阵，Ux可视为傅里叶变换，\tilde{x}U^{\top}则为傅里叶逆变换。
    由于特征分解计算复杂度高（o(n^2)），g_{\theta}(\Lambda)被通过使用切比雪夫多项式进行了K阶近似：

<div align=center><img src="https://latex.codecogs.com/gif.latex?g_{\theta'}\star&space;x\approx&space;\sum_{k=0}^{K}\theta'_{k}T_{k}(\tilde{L})x" title="g_{\theta'}\star x\approx \sum_{k=0}^{K}\theta'_{k}T_{k}(\tilde{L})x" /><\div>
  
     其中，\tilde{L}为rescaled的L_{norm}。
