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
$$\underset{G}{min}\,\underset{D}{max}V(D, G)=E_{x\sim p_{data}(x)}[\log D(x)]+E_{z\sim p_z(z)}[\log(1 − D(G(z)))]. $$

> 同时训练 $G$ 来最小化 $\log(1 − D(G(z)))$ 这个值：
> 换句话说，$G$ 和 $D$ 是在玩如下的两个选手的最小最大游戏：

$$\underset{G}{min}\,\underset{D}{max}V(D, G)=\underset{G}{min}\,\underset{D}{max}E_{x\sim p_{data}(x)}[\log D(x)]+\underset{G}{min}\,\underset{D}{max}E_{z\sim p_z(z)}[\log(1 − D(G(z)))]. $$

$$\underset{G}{min}\,\underset{D}{max}E_{x\sim p_{data}(x)}[\log D(x)]=\underset{D}{max}E_{x\sim p_{data}(x)}[\log D(x)]$$

> 这个等式，先看第一个 $E_{x\sim p_{data}(x)}$，这个期望的意义在于让鉴别器鉴别从真实数据分布中选出的值 $x$ 正确的概率最大，同时让从噪声中生成的数据的鉴别最小

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

**2. 阐述本文提出方法的技术细节**

# 论文的实验结果
**1. 阐述本文的实验内容**

**2. 本文方法的有效性是如何通过实验进行验证的？**

# 论文存在的不足之处
**1. 通过阅读此论文，你能否找到本文工作存在的不足之处？**

**2. 试阐述解决这些不足之处的基本思路？**


