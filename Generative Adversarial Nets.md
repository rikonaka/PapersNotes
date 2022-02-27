# 论文重要内容节选

**1. Introduction**

The promise of deep learning is to discover rich, hierarchical models that represent probability distributions over the kinds of data encountered in artificial intelligence applications, such as natural images, audio waveforms containing speech, and symbols in natural language corpora.

> 深度学习的目的是发现富的、分层的模型来表示所有在人工智能程序中出现数据的一个概率分布，像自然图像、音频波谱、自然语言符号。

So far, the most striking successes in deep learning have involved discriminative models, usually those that map a high-dimensional, rich sensory input to a class label.

> 深度学习最成功的地方，在于它其中的鉴别模型，通常用于将一个高维的、富感官的输入映射到一个分类标签。

These striking successes have primarily been based on the and dropout algorithms, using piecewise linear units which have a particularly well-behaved gradient .

> 这些最成功的地方主要是基于反向传播算法和 dropout 算法，其中通常使用具有很好梯度性质的分段线性单元。

Deep generative models have had less of an impact, due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context. We propose a new generative model estimation procedure that sidesteps these difficulties. 

> 对比上面的深度鉴别网络，深度生成模型没有很大的影响力，因为其难以近似很多在最大似然估计和相关策略中出现的棘手的概率计算，还因为难以在生成上下文的时候利用分段线性单元的优点。所以提出了新的生成模型估计方法来避免这些困难问题。

In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution.

> 在提出的对抗网络架构中，生成模型用于代表一个攻击者：一个鉴别模型应该学会分别这个样本是来自样本分布（model distribution）还是来自数据分布（data distribution）。

The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistinguishable from the genuine articles.

> 生成模型可以被认为一群造假者，试图生成虚假的货币而且使用它不需要检测，而鉴别模型则可以认为是警察，用于检测假币。在这个博弈中的对抗，将会提高造假币的方法直到假币无法和真品中分辨出来。

This framework can yield specific training algorithms for many kinds of model and optimization algorithm. In this article, we explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron, and the discriminative model is also a multilayer perceptron.

> …这篇文章中我们探讨一个特殊的案例，生成模型通过随机噪声和多层感知机来生成样本，鉴别模型同样也是多次感知机。

We refer to this special case as adversarial nets. In this case, we can train both models using only the highly successful backpropagation and dropout algorithms and sample from the generative model using only forward propagation. No approximate inference or Markov chains are necessary.

> …我们同时训练两个模型，仅仅只使用很成功的反向传播算法和 dropout 算法，然后样本来自生成模型，也仅仅使用前向传播算法。马尔可夫链和近似推理是不需要的。

**2. Related work**

An alternative to directed graphical models with latent variables are undirected graphical models with latent variables, such as restricted Boltzmann machines (RBMs), deep Boltzmann machines (DBMs) and their numerous variants. The interactions within such models are represented as the product of unnormalized potential functions, normalized by a global summation/integration over all states of the random variables.

> 具有潜在变量的有向图模型的替代方案是具有潜在变量的无向图模型，如受限的玻尔兹曼机（RBMs），深度玻尔兹曼机（DBMs）和它的多种变种（PS：受限玻尔兹曼机可以看作是只有两层结构的神经网络，而深度玻尔兹曼机则是多个受限玻尔兹曼机的结合）。模型中的交互可以被认为是未归一化的势函数的乘积，并通过一个对全局所有状态的求和/积分来归一化。

This quantity (the partition function) and its gradient are intractable for all but the most trivial instances, although they can be estimated by Markov chain Monte Carlo (MCMC) methods. Mixing poses a significant problem for learning algorithms that rely on MCMC.

> 这些量（配分函数）和它的梯度对于但是不重要的实例来说是棘手的，尽管它们可以通过 MCMC 估算。对于那些依赖 MCMC 的学习算法来说，mixing poses（怎么翻译）是一个很重要的问题。

Deep belief networks (DBNs) are hybrid models containing a single undirected layer and several directed layers. While a fast approximate layer-wise training criterion exists, DBNs incur the computational difficulties associated with both undirected and directed models.

> 深度信念网络（DBNs）是一个混合模型，包含了一个无向层和多个有向层。当使用一个快速逐层近似的标准训练时，DBNs 将会同时在无向和有向模型上产生计算困难。

Alternative criteria that do not approximate or bound the log-likelihood have also been proposed, such as score matching and noise-contrastive estimation (NCE). Both of these require the learned probability density to be analytically specified up to a normalization constant. Note that in many interesting generative models with several layers of latent variables (such as DBNs and DBMs), it is not even possible to derive a tractable unnormalized probability density. Some models such as denoising auto-encoders and contractive autoencoders have learning rules very similar to score matching applied to RBMs.

> 替代的标准不需要近似（do not approximate）或者使用对数近似边界（bound the log-likelihood）的方法也被提出了，比如分数匹配（score matching）和噪声对比估计（NCE）。它们两个都需要一个已知的概率密度来分析指定的一个归一化常量。注意，在很多有趣的有很多层隐变量的生成模型中（如 DBNs 和 DBMs），是不可能给出一个易于处理的非归一化的概率密度的。一些模型诸如去噪自动编码器（denoising autoen-coders）和收缩自动编码器（contractive autoencoders）会学习规则非常相似于分数匹配的 RBMs。

In NCE, as in this work, a discriminative training criterion is employed to fit a generative model. However, rather than fitting a separate discriminative model, the generative model itself is used to discriminate generated data from samples a fixed noise distribution. Because NCE uses a fixed noise distribution, learning slows dramatically after the model has learned even an approximately correct distribution over a small subset of the observed variables.

> 在 NCE 中，每一个鉴别训练标准都对应一个生成模型。然而，与其适应每一个不同的鉴别模型，生成模型自己被用于鉴别来自样本混合了噪声分布的生成数据。因为 NCE 使用了混合噪声分布，在模型在一个可观测变量的一小部分上得到一个近似正确的分布之后，学习会突然变得很慢。

Finally, some techniques do not involve defining a probability distribution explicitly, but rather train a generative machine to draw samples from the desired distribution. This approach has the advantage that such machines can be designed to be trained by back-propagation. Prominent recent work in this area includes the generative stochastic network (GSN) framework, which extends generalized denoising auto-encoders: both can be seen as defining a parameterized Markov chain, i.e., one learns the parameters of a machine that performs one step of a generative Markov chain.

> 最后，一些技术没有参与定义一个明确的概率分布，而是训练一个生成器来生成一个想要分布的样本。本文提出的方法对于上述的机器有优势，包括设计可以通过反向传播来训练。在这个区域最近杰出部分工作的包括生成随机网络（GSN）架构，扩展了正则噪声自动编码器，可以被认为是定义了一个参数化的马尔可夫链，通过执行一步的生成马尔可夫链来学习机器的参数。

Compared to GSNs, the adversarial nets framework does not require a Markov chain for sampling. Because adversarial nets do not require feedback loops during generation, they are better able to leverage piecewise linear units, which improve the performance of backpropagation but have problems with unbounded activation when used in a feedback loop. More recent examples of training
a generative machine by back-propagating into it include recent work on auto-encoding variational Bayes and stochastic backpropagation.

> 对比 GSNs，对抗网络框架在采样的时候不需要马尔可夫链。因为对抗网络在生成的时候不需要回馈循环，这样可以更好的利用分段线性单元的特性，这些分段线性单元将会提高反向传播的性能，当使用回馈循环时候，无限激活会产生很多的问题。最近很多通过反向传播训练生成器的例子都包含了自动编码的贝叶斯变种和随机反向传播。

**3. Adversarial nets**

The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons. To learn the generator’s distribution $p_g$ over data $x$, we define a prior on input noise variables $p_z(z)$, then represent a mapping to data space as $G(z; \theta_g)$, where $G$ is a differentiable function represented by a multilayer perceptron with parameters $\theta_g$.

> 对抗模型架构简单易行，当模型都是多层感知机的时候。为了学习到关于数据 $x$ 的生成器的分布 $p_g$，我们定义一个先验在输入噪声变量 $p_z(z)$ 上，其中 $G$ 是被参数 $\theta_g$ 表示的多层感知器的一个可微方程（这个 $G$ 从上文中理解就是将噪声映射到一个数据空间的函数，即将噪声转换成一个数据如图片）。

We also define a second multilayer perceptron $D(x; \theta_d)$ that outputs a single scalar. $D(x)$ represents the probability that $x$ came from the data rather than $p_g$. We train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$.

> 我们同时定义第二个多层感知器 $D(x; \theta_d)$，这个东西只会输出单个的标量值，代表了 $x$ 是来自数据还是生成器的分布 $p_g$ 的概率。我们训练这个 $D$ 来同时最大化训练数据正常标签和来自 $G$ 的样本正确标签的概率（这个 $D$ 是来鉴别 $G$ 生成的数据）。

We simultaneously train $G$ to minimize $\log(1 − D(G(z)))$:
In other words, $D$ and $G$ play the following two-player minimax game with value function $V(G, D)$:
$$\underset{G}{min}\,\underset{D}{max}V(D, G)=E_{x\sim p_{data}(x)}[\log D(x)]+E_{z\sim p_z(z)}[\log(1 − D(G(z)))]$$

> 同时训练 $G$ 来最小化 $\log(1 − D(G(z)))$ 这个值：
> 换句话说，$G$ 和 $D$ 是在玩如下的两个选手的最小最大游戏：

$$\underset{G}{min}\,\underset{D}{max}V(D, G)=\underset{G}{min}\,\underset{D}{max}E_{x\sim p_{data}(x)}[\log D(x)]+\underset{G}{min}\,\underset{D}{max}E_{z\sim p_z(z)}[\log(1 − D(G(z)))]$$

$$\underset{G}{min}\,\underset{D}{max}E_{x\sim p_{data}(x)}[\log D(x)]=\underset{D}{max}E_{x\sim p_{data}(x)}[\log D(x)]$$

$$\underset{G}{min}\,\underset{D}{max}E_{z\sim p_z(z)}[\log(1 − D(G(z)))]$$

> 这个等式，先看第一个 $E_{x\sim p_{data}(x)}$，这个期望的意义在于让鉴别器鉴别从真实数据分布中选出的值 $x$ 正确的概率最大。
> 而第二个 $E_{z\sim p_z(z)}$ 这个期望的意义在于让 鉴别器 $D$ 鉴别从生成器 $G$ 中生成的数据的成功概率的概率足够小，因为 $\underset{D}{max}[1-D(G(Z))]$ 要大，那么 $D(G(Z))$ 就得小，同时限制这个 $G$ 的范围要最小，让 $G$ 尽量小一些。
> 这个数学公式的意义就在于，让 $D$ 当输入为真实数据的时候，输出概率最大，输入为生成数据的时候，输出概率最小。

In the next section, we present a theoretical analysis of adversarial nets, essentially showing that the training criterion allows one to recover the data generating distribution as $G$ and $D$ are given enough capacity, i.e., in the non-parametric limit.

> 下一章中，我们提供了一个对抗网络的理论分析，本质上表明，当 $G$ 和 $D$ 有足够容量的时候，训练标准将允许恢复数据生成分布。

See Figure 1 for a less formal, more pedagogical explanation of the approach. In practice, we must implement the game using an iterative, numerical approach. Optimizing $D$ to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting.

> 在实践中，我们必须使用迭代和数值方法来实现这个博弈过程。在内部训练循环中完成优化 $D$  在计算上是禁止的，同时在有限数据集上可能导致过拟合。

Instead, we alternate between $k$ steps of optimizing $D$ and one step of optimizing $G$. This results in $D$ being maintained near its optimal solution, so long as $G$ changes slowly enough.

> 相反，我们在优化 $G$ 的 $k$ 步和优化 $D$ 的一步之间交替。这导致 $D$ 保持在其最佳解决方案附近，只要 $G$ 变化足够慢。

This strategy is analogous to the way that SML/PCD training maintains samples from a Markov chain from one learning step to the next in order to avoid burning in a Markov chain as part of the inner loop of learning. The procedure is formally presented in Algorithm 1.

> 这种策略类似于 SML/PCD 训练从一个学习步骤到下一个学习步骤维护来自马尔可夫链的样本的方式，以避免作为学习内部循环的一部分在马尔可夫链中燃烧。

In practice, equation 1 may not provide sufficient gradient for $G$ to learn well. Early in learning, when $G$ is poor, $D$ can reject samples with high confidence because they are clearly different from the training data. In this case, $\log (1 − D(G(z)))$ saturates. Rather than training $G$ to minimize $\log (1 − D(G(z)))$ we can train $G$ to maximize $\log D(G(z))$. This objective function results in the same fixed point of the dynamics of $G$ and $D$ but provides much stronger gradients early in learning.

> 在实践中，等式1可能不会为 $G$ 提供足够的梯度来很好的学习。在学习的早期，当 $G$ 很弱的时候，$D$ 可以高置信度地拒绝样本，因为其和训练数据明显的不同。这种情况下，$\log (1 − D(G(z)))$ 是饱和的。与其训练 $G$ 来最小化 $\log (1 − D(G(z)))$ 我们可以训练 $G$ 来最大化 $\log D(G(z))$。这个目标函数导致 $G$ 和 $D$ 的动态相同的固定点，但在学习早期提供了更强的梯度。

![Figure 1](https://img-blog.csdnimg.cn/7483bd4c12f8480199997eb720305336.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 1: Generative adversarial nets are trained by simultaneously updating the discriminative distribution ($D$, blue, dashed line) so that it discriminates between samples from the data generating distribution (black, dotted line) $p_x$ from those of the generative distribution $p_g(G)$ (green, solid line). The lower horizontal line is the domain from which $z$ is sampled, in this case uniformly. The horizontal line above is part of the domain of $x$. The upward arrows show how the mapping $x = G(z)$ imposes the non-uniform distribution $p_g$ on transformed samples. $G$ contracts in regions of high density and expands in regions of low density of $p_g$. (a) Consider an adversarial pair near convergence: $p_g$ is similar to $p_{data}$ and $D$ is a partially accurate classifier. (b) In the inner loop of the algorithm $D$ is trained to discriminate samples from data, converging to $D^∗(x) = \frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$. \(c\) After an update to $G$, gradient of $D$ has guided $G(z)$ to flow to regions that are more likely to be classified as data. (d) After several steps of training, if $G$ and $D$ have enough capacity, they will reach a point at which both cannot improve because $p_g = p_{data}$. The discriminator is unable to differentiate between the two distributions, i.e. $D(x) = \frac12$.

> 图像1：生成对抗网络被训练的同时更新鉴别分布（蓝线）鉴别来自真实数据生成分布的样本（黑线） $p_x$ 和来自生成分布（绿线） $p_G(x)$ 的数据。底下黑色的水平线是对 $z$ 采样的域。上面的水平线则是 $x$ 的一部分域。向上的箭头显示的是映射 $x=G(z)$ 强加非均匀分布 $p_g$ 到转换样本上。$G$ 在 $p_g$ 高密度区域收缩，低密度区域扩张。（a）考虑一个接近收敛的对抗：此时 $p_g$ 和 $p_{data}$ 是相似的，而 $D$ 是一个部分准确的分类器。（b）在算法 $D$ 的内部循环来训练鉴别器鉴别来自真实数据的样本，收敛于 $D^*(x)$ 这里。（c）在更新 $G$ 之后，$D$ 的梯度将会指引 $G(z)$ 落在最可能被分类为真实数据的区域。（d）在经过几步的训练之后，如果 $G$ 和 $D$ 有足够的容量，它们将会达到一个点，这个点不再能够提高因为此时 $p_g$ 已经等于 $p_{data}$ 了。此时这个鉴别器将不在能够分辨出两个分布，即 $D(x)=\frac 12$。

**4. Theoretical Results**

The generator $G$ implicitly defines a probability distribution $p_g$ as the distribution of the samples $G(z)$ obtained when $z\sim p_z$. Therefore, we would like Algorithm 1 to converge to a good estimator of $p_{data}$, if given enough capacity and training time. The results of this section are done in a non-parametric setting, e.g. we represent a model with infinite capacity by studying convergence in the space of probability density functions.

> 生成器 $G$ 隐式的定义了一个概率分布 $p_g$ 来作为样本 $G(z)$ 的分布当 $z\sim p_z$ 的时候。因此，我们在算法1中可以收敛于一个对于 $p_{data}$ 好的估计器，如果有足够的容量和训练时间。本章的结果是在非参数设置中完成的。我们通过研究概率密度函数空间中的收敛性来表示具有无限容量的模型。

We will show in section 4.1 that this minimax game has a global optimum for $p_g=p_{data}$. We will then show in section 4.2 that Algorithm 1 optimizes Eq 1, thus obtaining the desired result.

> 这个最小最大的博弈过程有个全局的优化目标就是让 $p_g=p_{data}$。

![Algorithm 1](https://img-blog.csdnimg.cn/b84261897d4c49da94ee1aef5263d57a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
> 小批量随机梯度下降训练用于生成对抗网络，$k$ 是一个超参数，其中在实验中使用 $k=1$ 作为一个最简单易行的选项。
> 整个算法有个大的迭代次数（这个算法是先更新鉴别器，再更新生成器），简单来说，就是：
> 1. 从生成模型中得到 $m$ 个样本；
> 2. 从真实数据模型中获得 $m$ 个样本；
> 3. 之后用这些个样本计算损失函数的梯度；
> 4. 然后通过提高其随机梯度来更新鉴别器（这一步代表了下面等式的前半部分的目标）；
> $$\underset{G}{min}\,\underset{D}{max}V(D, G)=\underset{D}{max}E_{x\sim p_{data}(x)}[\log D(x)]+\underset{G}{min}\,\underset{D}{max}E_{z\sim p_z(z)}[\log(1 − D(G(z)))]$$
> 5. 上面1-4步循环 $k$ 次；
> 6. 再从生成模型中获得 $m$ 个噪声样本；
> 7. 之后根据这新的 $m$ 个噪声样本，通过降低其随机梯度来更新生成器（这一步对应了上面说到的先暂时的让 $D(x)$ 最大，然后固定 $D(x)$，之后再来更新 $G(z)$ 这个过程，也就是这两个优化目标，不是同时优化而是分批次的优化，先让 $D$ 走 $k$ 步，然后再让 $G$ 走 $1$ 步，然后循环）。

**4.1 Global Optimality of $p_g = p_{data}$**

We first consider the optimal discriminator $D$ for any given generator $G$.
Proposition 1. For $G$ fixed, the optimal discriminator $D$ is
$$D^*_G(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$$

> $G$ 固定了，然后优化 $D$（这个优化目标就是 $p_{data}(x)=p_g(x)$，如果达到完美相等了，最终 $D^*_G(x)=\frac 12$，于是我们的优化目标就是让这个值要接近 $\frac 12$）。

> 之后的证明过程就是在证明在任意给出 $G$ 的情况下， $V(G, D)$ 在 $D^*_G(x)$ 处取得最大值，$D^*_G(x)$ 就是上面那个等式。

Theorem 1. The global minimum of the virtual training criterion $C(G)$ is achieved if and only if $p_g = p_{data}$. At that point, $C(G)$ achieves the value $−\log 4$.

> 其中 $C(G)=\underset{D}{max}V(G, D)$。

> 定理1。全局关于虚拟训练标准 $C(G)$ 达到最小时，当且仅当 $p_g=p_{data}$ 时。同时在这个点上 $C(G)$ 达到值 $-\log 4$（然后接下来的都是关于这个定理的证明我们就不管了）。

**4.2 Convergence of Algorithm 1**

Proposition 2. If $G$ and $D$ have enough capacity, and at each step of Algorithm 1, the discriminator is allowed to reach its optimum given $G$, and $p_g$ is updated so as to improve the criterion

$$\mathbb{E}_{x\sim p_{data}}[\log D^*_G(x)]+\mathbb{E}_{x\sim p_g}[\log (1-D^*_G(x)]$$

then $p_g$ converges to $p_{data}$

> 定理2。如果 $G$ 和 $D$ 有足够的容量，而且在算法1的每一步中，鉴别器都被允许在给定 $G$ 的情况下达到最优值，并更新 $p_g$ 以改进下式标准
> $$\mathbb{E}_{x\sim p_{data}}[\log D^*_G(x)]+\mathbb{E}_{x\sim p_g}[\log (1-D^*_G(x)]$$
> 之后 $p_g$ 将收敛到 $p_{data}$（之后的内容都是在证明这个结论就不详说了）

In practice, adversarial nets represent a limited family of $p_g$ distributions via the function $G(z; \theta_g)$, and we optimize $\theta_g$ rather than $p_g$ itself. Using a multilayer perceptron to define $G$ introduces multiple critical points in parameter space. However, the excellent performance of multilayer perceptrons in practice suggests that they are a reasonable model to use despite their lack of theoretical guarantees.

> 在实践中，对抗网络只表示一个 $p_g$ 有限的家族分布，通过方程 $G(z; \theta_g)$，之后我们优化 $\theta_g$ 而不是 $p_g$（这句话的大概意思就是我们用 $G(z; \theta_g)\sim p_g$ 来近似表示 $p_g$，之后我们只需优化生成网络的参数 $\theta_g$ 就完事，而不在需要去优化 $p_g$ 本身，因为 $G(z; \theta_g)$ 只是 $p_g$ 的一个很小的子集，优化 $\theta_g$ 比优化 $p_g$ 简单）。使用一个多层感知机来定义 $G$ 在参数空间引入多个临界点。因为多层感知机在实践中性能的优越性，所以采用多层感知机是一个合理的模型，而它们的缺点也只是缺乏理论的保证。

**5 Experiments**

We trained adversarial nets an a range of datasets including MNIST, the Toronto Face Database (TFD), and CIFAR-10. The generator nets used a mixture of rectifier linear activations and sigmoid activations, while the discriminator net used maxout activations. Dropout was applied in training the discriminator net. While our theoretical framework permits the use of dropout and other noise at intermediate layers of the generator, we used noise as the input to only the bottommost layer of the generator network.

> 训练用到了 MNIST，TFD 和 CIFAR-10 这三个数据集。然后生成器用到一个混合的线性整流器和 sigmoid 激活函数，而鉴别网络用的是 maxout 激活函数。同时在训练鉴别器中还使用了 dropout 方法。理论上框架允许使用 dropout 和其他噪声在生成器的中间层，但是这里仅使用噪声在生成器的最低层作为输入。

We estimate probability of the test set data under $p_g$ by fitting a Gaussian Parzen window to the samples generated with $G$ and reporting the log-likelihood under this distribution. The $\sigma$ parameter of the Gaussians was obtained by cross validation on the validation set.

> 我们通过将 Gaussian Parzen 窗口拟合到使用 $G$ 生成的样本并报告此分布下的对数似然来估计测试集数据在 $p_g$ 下的概率。然后 Gaussian 分布的 $\sigma$ 参数通过在验证数据集上的交叉验证来获得。

This procedure was introduced in Breuleux et al. and used for various generative models for which the exact likelihood is not tractable. Results are reported in Table 1. This method of estimating the likelihood has somewhat high variance and does not perform well in high dimensional spaces but it is the best method available to our knowledge. Advances in generative models that can sample but not estimate likelihood directly motivate further research into how to evaluate such models.

> 这种处理方式被用于很多生成模型，因为其准确的似然不好被驾驭。这种估计似然的方法存在一些高的方差和在高维空间性能不是很好，但是这是已知最好的方法。生成模型的优势可以采样但是不会对直接估计似然做进一步的研究。

In Figures 2 and 3 we show samples drawn from the generator net after training. While we make no claim that these samples are better than samples generated by existing methods, we believe that these samples are at least competitive with the better generative models in the literature and highlight the potential of the adversarial framework.

> 图2和图3显示的是来自生成网络在训练之后生成的图片，这些生成样本比其他生成模型生成的样本还要好。

![Figure 2](https://img-blog.csdnimg.cn/f7e78cf544d04015a795472ea8294852.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 2: Visualization of samples from the model. Rightmost column shows the nearest training example of the neighboring sample, in order to demonstrate that the model has not memorized the training set. Samples are fair random draws, not cherry-picked. Unlike most other visualizations of deep generative models, these images show actual samples from the model distributions, not conditional means given samples of hidden units. Moreover, these samples are uncorrelated because the sampling process does not depend on Markov chain mixing. a) MNIST b) TFD c) CIFAR-10 (fully connected model) d) CIFAR-10 (convolutional discriminator and “deconvolutional” generator)

> 前面五列都是生成样本，而最右边黄色框那一列，则是的是相邻样本最近的一个训练样本（一个是生成的一个是用于训练的）。为了保证模型不会记忆任何训练数据，样本是公平随机的画出，而不是特意挑选的。不像大多数的生成模型可视化，这些图片显示来自模型分布实际的样本。这些样本不相关是因为样本处理不依赖于马尔可夫链。

![Figure 3](https://img-blog.csdnimg.cn/16f46866531c4516b609f96e6128b5b9.png)
Figure 3: Digits obtained by linearly interpolating between coordinates in $z$ space of the full model.

> 通过在完整模型的 $z$ 空间中的坐标之间进行线性插值获得的数字（不懂）。

**6. Advantages and disadvantages**

This new framework comes with advantages and disadvantages relative to previous modeling frameworks. The disadvantages are primarily that there is no explicit representation of $p_g(x)$, and that $D$ must be synchronized well with $G$ during training (in particular, $G$ must not be trained too much without updating $D$, in order to avoid “the Helvetica scenario” in which $G$ collapses too many values of $z$ to the same value of $x$ to have enough diversity to model $p_{data}$), much as the negative chains of a Boltzmann machine must be kept up to date between learning steps. The advantages are that Markov chains are never needed, only backprop is used to obtain gradients, no inference is needed during learning, and a wide variety of functions can be incorporated into the model. Table 2 summarizes the comparison of generative adversarial nets with other generative modeling approaches.

> 新的架构带来了前人模型未具有的一些优势和劣势。劣势最重要的是没有一个关于 $p_g(x)$ 的表达式，而且 $D$ 必须和 $G$ 同步得很好在训练中（特别的，相较于更新 $D$，$G$ 不能训练得太多，为了避免 the Helvetica scenario 也就是其中 $G$ 将过多的 $z$ 值折叠为相同的 $x$ 值，从而有足够的多样性来建模 $p_{data}$），这有点像波尔兹曼机。而优势则是不再需要马尔科夫链，仅仅使用反向传播来获得梯度，不再需要推理在学习的过程中，同时广泛的函数种类可以被合并在模型中。

The aforementioned advantages are primarily computational. Adversarial models may also gain some statistical advantage from the generator network not being updated directly with data examples, but only with gradients flowing through the discriminator. This means that components of the input are not copied directly into the generator’s parameters. Another advantage of adversarial networks is that they can represent very sharp, even degenerate distributions, while methods based on Markov chains require that the distribution be somewhat blurry in order for the chains to be able to mix between modes.

> 前述的优势最主要的是计算性。对抗模型也许也许会从生成网络中获得一些数据样本中没有的统计优势，仅仅是梯度流过鉴别器。这意味着这些输入的组件不是直接复制进去生成器的参数中。其他的优势关于对抗网络的还有他们可以表示非常尖锐而且退化的分布，而其他基于马尔科夫链则要求这些分布有点模糊以便链能在模型之间混合。

# 论文的研究背景
**1. 本论文解决什么问题？（能否通过一个示例来说明拟解决的问题）**

本篇论文主要提出了一个包含生成网络和鉴别网络的对抗网络来训练深度生成模型，从而避免以前单独深度生成网络中存在的最大似然估计和相关策略出现的棘手概率计算问题。

**2. 关于该问题，目前的相关工作有哪些？这些相关工作有何优缺点？（综述相关工作）**

相关工作主要有受限的玻尔兹曼机（RBMs），深度玻尔兹曼机（DBMs）和它的多种变种，还有深度信念网络（DBNs）和分数匹配（score matching）和噪声对比估计（NCE）和生成随机网络（GSN）架构。

其中玻尔兹曼机（RBMs），深度玻尔兹曼机（DBMs）和它的多种变种的优点文中没说，但是缺点是对上述模型的梯度求导比较困难，虽然可以通过 MCMC 来完成。

深度信念网络（DBNs）优点文中也未说明，缺点是当使用一个快速逐层近似的标准训练时，DBNs 将会同时在无向和有向模型上产生计算困难。

分数匹配（score matching）和噪声对比估计（NCE）优点文中也未说明，缺点是需要一个已学习的概率密度来分析指定的一个归一化常量，但是很多情况下是不可能给出一个易于处理的非归一化的概率密度的。 在NCE 中，每一个鉴别训练标准都对应一个生成模型，无法很好的做到自适应，且需要一个已知的概率密度来分析指定的一个归一化常量，这通常是不容易给出的，而且在模型在一个可观测变量的一小部分上得到一个近似正确的分布之后，学习会突然变得很慢。

生成随机网络（GSN）架构优点文章也未说明，缺点是使用了马尔科夫链（Markov chain）和回馈循环（feedback loops），无法很好的利用分段线性单元特性，在使用回馈循环的时候，分段线性单元会存在激活失去约束的情况。

# 论文的主要研究内容
**1. 针对已有工作的不足之处，本文提出了什么方法？（该方法为何有效？）该方法的基本思路是什么？主要创新点在哪？**

提出了对抗网络的思想来训练生成器。因为该方法并不求解出一个显式的数学表达式来表示生成模型，而是通过深度神经网络的方法来模拟逼近一个真实分布，通过对抗网络来训练生成模型，从而使其更好的贴合真实数据，在这种无法求出确切数学表达式的情况下，使用深度神经网络更具优势。创新点就是不再要求生成模型求解出一个显式的数学表达式，而是通过深度神经网络来模拟这个数学表达式。

**2. 阐述本文提出方法的技术细节**

关于本文提出的方法的技术细节就是算法1中的迭代过程，简单来说如下所示：
![Algorithm 1](https://img-blog.csdnimg.cn/b84261897d4c49da94ee1aef5263d57a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
小批量随机梯度下降训练用于生成对抗网络，$k$ 是一个超参数，其中在实验中使用 $k=1$ 作为一个最简单易行的选项。
整个算法有个大的迭代次数（这个算法是先更新鉴别器，再更新生成器），简单来说，就是：
1. 从生成模型中得到 $m$ 个样本；
2. 从真实数据模型中获得 $m$ 个样本；
3. 之后用这些个样本计算损失函数的梯度；
4. 然后通过提高其随机梯度来更新鉴别器（这一步代表了下面等式的前半部分的目标）；
$$\underset{G}{min}\,\underset{D}{max}V(D, G)=\underset{D}{max}E_{x\sim p_{data}(x)}[\log D(x)]+\underset{G}{min}\,\underset{D}{max}E_{z\sim p_z(z)}[\log(1 − D(G(z)))]$$
5. 上面1-4步循环 $k$ 次；
6. 再从生成模型中获得 $m$ 个噪声样本；
7. 之后根据这新的 $m$ 个噪声样本，通过降低其随机梯度来更新生成器（这一步对应了上面说到的先暂时的让 $D(x)$ 最大，然后固定 $D(x)$，之后再来更新 $G(z)$ 这个过程，也就是这两个优化目标，不是同时优化而是分批次的优化，先让 $D$ 走 $k$ 步，然后再让 $G$ 走 $1$ 步，然后循环）。

# 论文的实验结果
**1. 阐述本文的实验内容**

实验采用了 MNIST，TFB 和 CIFAR-10 数据集，以下为实验结果。

In Figures 2 and 3 we show samples drawn from the generator net after training. While we make no claim that these samples are better than samples generated by existing methods, we believe that these samples are at least competitive with the better generative models in the literature and highlight the potential of the adversarial framework.

> 图2和图3显示的是来自生成网络在训练之后生成的图片，这些生成样本比其他生成模型生成的样本还要好。

![Figure 2](https://img-blog.csdnimg.cn/f7e78cf544d04015a795472ea8294852.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 2: Visualization of samples from the model. Rightmost column shows the nearest training example of the neighboring sample, in order to demonstrate that the model has not memorized the training set. Samples are fair random draws, not cherry-picked. Unlike most other visualizations of deep generative models, these images show actual samples from the model distributions, not conditional means given samples of hidden units. Moreover, these samples are uncorrelated because the sampling process does not depend on Markov chain mixing. a) MNIST b) TFD c) CIFAR-10 (fully connected model) d) CIFAR-10 (convolutional discriminator and “deconvolutional” generator)

> 前面五列都是生成样本，而最右边黄色框那一列，则是的是相邻样本最近的一个训练样本（一个是生成的一个是用于训练的）。为了保证模型不会记忆任何训练数据，样本是公平随机的画出，而不是特意挑选的。不像大多数的生成模型可视化，这些图片显示来自模型分布实际的样本。这些样本不相关是因为样本处理不依赖于马尔可夫链。

![Figure 3](https://img-blog.csdnimg.cn/16f46866531c4516b609f96e6128b5b9.png)
Figure 3: Digits obtained by linearly interpolating between coordinates in $z$ space of the full model.

> 通过在完整模型的 $z$ 空间中的坐标之间进行线性插值获得的数字（不懂）。

**2. 本文方法的有效性是如何通过实验进行验证的？**

对比生成出来图片的和其他生成模型之间，可以明显发现本文提出的模型具有更好的生成质量。

# 论文存在的不足之处
**1. 通过阅读此论文，你能否找到本文工作存在的不足之处？**

The disadvantages are primarily that there is no explicit representation of $p_g(x)$, and that $D$ must be synchronized well with $G$ during training (in particular, $G$ must not be trained too much without updating $D$, in order to avoid “the Helvetica scenario” in which $G$ collapses too many values of $z$ to the same value of $x$ to have enough diversity to model $p_{data}$), much as the negative chains of a Boltzmann machine must be kept up to date between learning steps.

> 劣势最重要的是没有一个关于 $p_g(x)$ 的表达式，而且 $D$ 必须和 $G$ 同步得很好在训练中（特别的，相较于更新 $D$，$G$ 不能训练得太多，为了避免 the Helvetica scenario 也就是其中 $G$ 将过多的 $z$ 值折叠为相同的 $x$ 值，从而有足够的多样性来建模 $p_{data}$），这有点像波尔兹曼机。

不足之处个人认为是 $D$ 和 $G$ 的训练没有一个自适应的过程，还是得依靠经验来调整 $D$ 和 $G$ 之间的协调。 

**2. 试阐述解决这些不足之处的基本思路？**

无...

