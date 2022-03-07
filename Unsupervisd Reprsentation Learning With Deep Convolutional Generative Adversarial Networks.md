# 论文重要内容节选
**Abstract**

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

> 近几年中，使用卷积网路的有监督的学习大量应用在计算机视觉任务中。而相对的，卷积网络的无监督学习却受到很少的关注。在本片论文的工作中，我们为 CNNs 建立一个有监督学习到无监督学习成功的桥梁。我们介绍我们这一类方法叫 DCGANs，通过某些架构限制来，证明它们是无监督学习的有力候选人。在不同的图片数据集上训练，我们将展示令人信服的证据，我们的深度卷积对抗对在生成器和判别器中学习了从对象部分到场景的表示层次结构。之后，我们使用学习到的特征来对新的任务 - 证明它们对一般图像表示的适用性。

**1 Introduction**

Learning reusable feature representations from large unlabeled datasets has been an area of active research. In the context of computer vision, one can leverage the practically unlimited amount of unlabeled images and videos to learn good intermediate representations, which can then be used on a variety of supervised learning tasks such as image classification.

> 从大量未标记的数据集中学得一个可重复使用的特征表示是一个热门研究领域。在计算机视觉语境中，这可以使用几乎无限的未标记图片和视频来学习得一个好的中间表示，这可以被用于很多监督学习的变种中，诸如图像分类。

We propose that one way to build good image representations is by training Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), and later reusing parts of the generator and discriminator networks as feature extractors for supervised tasks.

> 这篇论文提出了一种通过训练对抗网络的方法来建立一个好的图像表示，之后重复使用其中的生成器和鉴别器网络部分作为特征提取器用于监督任务。

GANs provide an attractive alternative to maximum likelihood techniques. One can additionally argue that their learning process and the lack of a heuristic cost function (such as pixel-wise independent mean-square error) are attractive to representation learning. GANs have been known to be unstable to train, often resulting in generators that produce nonsensical outputs. There has been very limited published research in trying to understand and visualize what GANs
learn, and the intermediate representations of multi-layer GANs.

> GANs 提供了一个吸引人的关于最大似然技术的替代品。其中它的学习过程和一个启发式的损失函数对于表示学习来说很具有吸引力。GANs 曾经被认为是不适合训练的，经常认为在生成器中生成了荒谬的输出。只有很有限的文章在试图理解和图像化 GANs 学到了什么，多层 GANs 的中间表示是什么。

In this paper, we make the following contribution
* We propose and evaluate a set of constraints on the architectural topology of Convolutional
GANs that make them stable to train in most settings. We name this class of architectures Deep Convolutional GANs (DCGAN).
* We use the trained discriminators for image classification tasks, showing competitive performance with other unsupervised algorithms.
* We visualize the filters learnt by GANs and empirically show that specific filters have learned to draw specific objects.
*  We show that the generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples.

> 这篇文章的贡献
> * 提出和评估了一系列的约束在卷积 GANs 的架构拓扑上使之在大都数的设定中都处于稳定。之后命名这一类架构未 Deep Convolutional GANs (DCGAN)。
> * 我们使用训练过的鉴别器在图像分类任务中，显示出其具有竞争力的性能对比其他非监督算法。
> * 我们可视化了 GANs 学到的过滤器然后凭经验显示特殊的过滤器已经学会画出特定的对象。
> * 我们展示生成器具有有趣的向量算术特性，可以轻松的操作生成样本的许多语义质量。

**2 Related work**

**2.1 Representation Learning from Unlabeled Data**

Unsupervised representation learning is a fairly well studied problem in general computer vision research, as well as in the context of images. A classic approach to unsupervised representation learning is to do clustering on the data (for example using K-means), and leverage the clusters for improved classification scores. In the context of images, one can do hierarchical clustering of image patches (Coates & Ng, 2012) to learn powerful image representations. Another popular method is to train auto-encoders (convolutionally, stacked (Vincent et al., 2010), separating the what and where components of the code (Zhao et al., 2015), ladder structures (Rasmus et al., 2015)) that encode an image into a compact code, and decode the code to reconstruct the image as accurately as possible. These methods have also been shown to learn good feature representations from image pixels. Deep belief networks (Lee et al., 2009) have also been shown to work well in learning hierarchical representations.

> 无监督的表示学习是一个相当好的问题，在普通计算机视觉研究中，其内容为图片。一个经典的无监督表示学习是对数据做一个聚类（如 K-means），之后利用簇来提高分类的分数。如果内容为图片的时候，我们可以对图像补丁做一个分成的聚类来学习一个强有力的图像表示。其他流行的方法是训练一个自动编码器，分离代码的内容和位置组件，阶梯结构，编码一个图片成为紧凑的代码，之后试图精确的解码这个代码来重组图像。这种方法已被证明可以从图像像素中学习出一个好的特征表示。深度信念网络也被证明在学习一个分层表示时工作得很好。

**2.2 Generating Natural Images**

Generative image models are well studied and fall into two categories: parametric and non-parametric.

> 生成图像模型主要有两类：参数化的和非参数化的。

The non-parametric models often do matching from a database of existing images, often matching patches of images, and have been used in texture synthesis (Efros et al., 1999), super-resolution (Freeman et al., 2002) and in-painting (Hays & Efros, 2007).

> 无参数的模型经常从已有图片的数据中做匹配，还经常用作匹配图片的部分区域，还被用做纹理合成（texture synthesis），超分辨率和画中画。

Parametric models for generating images has been explored extensively (for example on MNIST digits or for texture synthesis (Portilla & Simoncelli, 2000)). However, generating natural images of the real world have had not much success until recently. A variational sampling approach to generating images (Kingma & Welling, 2013) has had some success, but the samples often suffer from being blurry. Another approach generates images using an iterative forward diffusion process (Sohl-Dickstein et al., 2015). Generative Adversarial Networks (Goodfellow et al., 2014) generated images suffering from being noisy and incomprehensible. A laplacian pyramid extension to this approach (Denton et al., 2015) showed higher quality images, but they still suffered from the objects looking wobbly because of noise introduced in chaining multiple models. A recurrent network approach (Gregor et al., 2015) and a deconvolution network approach (Dosovitskiy et al., 2014) have also recently had some success with generating natural images. However, they have not leveraged the generators for supervised tasks.

> 参数化的生成图像模型已经被广泛的探索了。然而，生成来自现实世界的自然图像最近还未获得很大的成功。一些变分抽样的方法来生成图片已经获得了一些成功，但是这些样本经常很模糊。其他一些方法生成图像使用了迭代前向扩散方法，生成对抗网络生成的图片则嘈杂和无法连接。一个拉普拉斯金字塔扩展方法显示了高的图片质量，但是它们依旧有目标摇摇晃晃的问题，因为在链接多个模型的时候引入了噪声。一种循环网络方法和一种反卷积网络方法最近同样获得了一些成功早生成自然图片上。但是，它们依旧没有为监督任务使用生成器。

**2.3 Visualizing the Internals of CNNs**

One constant criticism of using neural networks has been that they are black-box methods, with little understanding of what the networks do in the form of a simple human-consumable algorithm. In the context of CNNs, Zeiler et. al. (Zeiler & Fergus, 2014) showed that by using deconvolutions and filtering the maximal activations, one can find the approximate purpose of each convolution filter in the network. Similarly, using a gradient descent on the inputs lets us inspect the ideal image that activates certain subsets of filters (Mordvintsev et al.).

> 一个关于使用神经网络的普遍批评是认为神经网络是黑盒方法，也就是目前来说人类无法解释和理解神经网络内部的具体原理，在 CNNs 的语境中，Zeiler 展示了通过使用反卷积和过滤最大激活函数，我们可以发现网络中每一个卷积核大概的意图。类似的，在输入中使用梯度下降可以让我们检查某些激活过滤器子集的理想图像。

**3 Approach and Model Architecture**

Historical attempts to scale up GANs using CNNs to model images have been unsuccessful. This motivated the authors of LAPGAN (Denton et al., 2015) to develop an alternative approach to iteratively upscale low resolution generated images which can be modeled more reliably. We also encountered difficulties attempting to scale GANs using CNN architectures commonly used in the supervised literature. However, after extensive model exploration we identified a family of architectures that resulted in stable training across a range of datasets and allowed for training higher resolution and deeper generative models.

> 使用 CNNs 对图像进行建模来扩大 GANs 的历史尝试并不成功。这个动机来自 LAPGAN 的作者试图开发一个替代方法，迭代的提高低分辨率生成图像，可以更可信的建模。我们同样遭遇了困难的尝试，使用 CNN 结构来扩展 GANs 的，用于监督环境。然而，在广泛模型的探索之后，我们发现一个架构类，可以通过一系列的数据集稳定的训练，同时允许训练高分辨率和深度生成模型。

Core to our approach is adopting and modifying three recently demonstrated changes to CNN architectures.

> 我们方法的核心是采用和修改最近展示的对 CNN 架构的三个更改。

The first is the all convolutional net (Springenberg et al., 2014) which replaces deterministic spatial pooling functions (such as maxpooling) with strided convolutions, allowing the network to learn its own spatial downsampling. We use this approach in our generator, allowing it to learn its own spatial upsampling, and discriminator.

> 第一个是全卷积网络，通过大步的卷积替代了确定性的空间池化函数（如最大池化），允许网络学习到它自己的空间下采样。我们使用这个方法在我们的生成器中，允许它学习自己的空间上采样和鉴别器。

Second is the trend towards eliminating fully connected layers on top of convolutional features. The strongest example of this is global average pooling which has been utilized in state of the art image classification models (Mordvintsev et al.). We found global average pooling increased model stability but hurt convergence speed. A middle ground of directly connecting the highest convolutional features to the input and output respectively of the generator and discriminator worked well. The first layer of the GAN, which takes a uniform noise distribution $Z$ as input, could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack. For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output. See Fig. 1 for a visualization of an example model architecture.

> 第二个是在卷积特征之上消除全连接层的趋势。其中最有力的样本是全局平均池化，这种方法已经在 state of the art 图片分离模型中使用了。我们发现全局平均池化提高了模型的稳定性，但是伤害了收敛速度。将最高卷积特征分别直接连接到生成器和鉴别器的输入和输出的中间地带效果很好。GAN 的第一层，使用一个均匀噪声分布 $Z$ 作为输入，可以被视作全连接因为这只是矩阵乘法，但是其结果被改造成4维张量，作为卷积的输入。而鉴别器，最后卷积层被拉伸然后输入到单个 sigmoid 函数中，图1展示了示例模型架构。

![Figure 1](https://img-blog.csdnimg.cn/3773d7a9598e4d6b8d7aae08b3775da0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 1: DCGAN generator used for LSUN scene modeling. A 100 dimensional uniform distribution $Z$ is projected to a small spatial extent convolutional representation with many feature maps. A series of four fractionally-strided convolutions (in some recent papers, these are wrongly called deconvolutions) then convert this high level representation into a 64 × 64 pixel image. Notably, no fully connected or pooling layers are used.

> 输入是一个100维的均匀分布 $Z$，投影到小空间范围内有很多 feature maps 的卷积表示。之后四个部分步幅的卷积（最近的部分论文中，它们被错误的叫做反卷积）转换高层表示到 64 × 64 像素图片。尤其，没有使用全连接或者池化层。

Third is Batch Normalization (Ioffe & Szegedy, 2015) which stabilizes learning by normalizing the input to each unit to have zero mean and unit variance. This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models. This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples to a single point which is a common failure mode observed in GANs. Directly applying batchnorm to all layers however, resulted in sample oscillation and model instability. This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer.

> 第三是批量标准化，通过将每个单元的输入归一化为零均值和单位方差来稳定学习。这帮助处理训练中因为缺乏初始化导致的问题和帮助更深模型中的梯度流动。这证实了获得深度生成器在学习开始时，避免生成器从所有样本中崩溃到单个点的问题，这个问题是 GANs 的普遍问题。直接的应用批量标准化到所有层中。会导致样本震荡和模型不稳定。这可以通过不使用批量标准化到生成器输出层和鉴别器的输入层来避免。

The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output layer which uses the Tanh function. We observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution. Within the discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work well, especially for higher resolution modeling. This is in contrast to the original GAN paper, which used the maxout activation (Goodfellow et al., 2013).

> ReLU 激活函数在生成器中使用，除了使用 Tanh 函数的输出层。我们观察到使用一个有界的激活函数允许模型来学习达到饱和得更快，还能覆盖训练分布的颜色空间。在生成器中，我们发现泄露校正激活函数工作得很好，尤其对于高分辨率的模型。这是和使用了 maxout 激活的原 GAN 的论文相比之下。

Architecture guidelines for stable Deep Convolutional GANs
* Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
* Use batchnorm in both the generator and the discriminator.
* Remove fully connected hidden layers for deeper architectures.
* Use ReLU activation in generator for all layers except for the output, which uses Tanh.
* Use LeakyReLU activation in the discriminator for all layers.

> 稳定的深度卷积 GANs 架构指南
> * 用步幅卷积（鉴别器）和分开步数卷积（生成器）替换所有池化层。
> * 在生成器和鉴别器中都使用批量标准化。
> * 维深度架构移除全连接隐层。
> * 在生成器中为所有层使用 ReLU 激活除了输出层，输出层使用 Tanh 激活。
> * 使用 LeakyReLU 激活函数。

**4 Details of Adversarial Training**

We trained DCGANs on three datasets, Large-scale Scene Understanding (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset. Details on the usage of each of these datasets are given below.

> 我们训练 DCGANs 在三个数据集上，LSUN，Imagenet-1k 和新的脸部数据集。关于这些数据集的使用详情在下面。

No pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1]. All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128. All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02. In the LeakyReLU, the slope of the leak was set to 0.2 in all models. While previous GAN work has used momentum to accelerate training, we used the Adam optimizer (Kingma & Ba, 2014) with tuned hyperparameters. We found the suggested learning rate of 0.001, to be too high, using 0.0002 instead. Additionally, we found leaving the momentum term $\beta_1$ at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.

> 当训练数据时候没有预处理图片到一个 tanh 激活函数的范围 [-1, 1] 之中。所有模型使用一个大小为128的 mini-batch 随机梯度下降（SGD）来训练。所有参数都被初始化为中间点为0的正态分布，其标准差为0.02。在 LeakyReLU 中，而 leak 的斜率设置为0.2在所有模型中。当前面 GAN 的工作呗用作一个动量来加速训练，我们使用了 Adam 优化和调整过的超参数。之后我们发现建议的学习率为0.001，如果太高了一点，可以使用0.0002替代。另外，我们发现如果让动量项（momentum term）$\beta_1$ 保持在建议值0.9作用于训练将导致震荡和不稳定，当减少这个值到0.5的时候将会帮助稳定训练。

**4.1 LSUN**

As visual quality of samples from generative image models has improved, concerns of over-fitting
and memorization of training samples have risen. To demonstrate how our model scales with more data and higher resolution generation, we train a model on the LSUN bedrooms dataset containing a little over 3 million training examples. Recent analysis has shown that there is a direct link between how fast models learn and their generalization performance (Hardt et al., 2015). We show samples from one epoch of training (Fig.2), mimicking online learning, in addition to samples after convergence (Fig.3), as an opportunity to demonstrate that our model is not producing high quality samples via simply overfitting/memorizing training examples. No data augmentation was applied to the images.

> 随着生成图片模型中样本视觉质量的提高，关于过拟合和训练样本记忆的担心开始出现。为了证明我们模型扩展了更多的数据和更高分辨率，我们训练一个模型在 LSUN bedrooms 数据上，包含了至少三百万训练样本。最近的分析显示，一个模型学习的速度和它们的生成性能之间有直接关系。我们展示了其中一个训练 epoch 的数据（Fig.2），模仿在线学习，另外的是一个样本在收敛之后的样子（Fig.3），作为证明我们的模型不会生成高质量样本通过简单的过拟合和记忆训练样本。没有对图像数据进行增强。

![Figure 2](https://img-blog.csdnimg.cn/33df05584f5e452db5e196857e35b115.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 2: Generated bedrooms after one training pass through the dataset. Theoretically, the model could learn to memorize training examples, but this is experimentally unlikely as we train with a small learning rate and minibatch SGD. We are aware of no prior empirical evidence demonstrating memorization with SGD and a small learning rate.

> 在数据中进行一轮训练之后生成的 bedrooms 。理论上，这个模型可以学着记忆训练样本，但是这在实验上不太可能，因为我们使用了较小的学习率和小排量的 SGD。我们意识到没有先前经验证据表明 SGD 和小学习率有记忆性。

![Figure 3](https://img-blog.csdnimg.cn/340fd46e8d794f87b6fa8d2a0e8c24fc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 3: Generated bedrooms after five epochs of training. There appears to be evidence of visual under-fitting via repeated noise textures across multiple samples such as the base boards of some of the beds.

> 在五轮训练之后生成的 bedrooms。这里似乎是欠拟合在视觉上的证据，通过重复噪声纹理在多个样本上，如一些床的基板。

**4.1.1 Deduplication**

To further decrease the likelihood of the generator memorizing input examples (Fig.2) we perform a simple image de-duplication process. We fit a 3072-128-3072 de-noising dropout regularized RELU autoencoder on 32x32 downsampled center-crops of training examples. The resulting code layer activations are then binarized via thresholding the ReLU activation which has been shown to be an effective information preserving technique (Srivastava et al., 2014) and provides a convenient form of semantic-hashing, allowing for linear time de-duplication . Visual inspection of hash collisions showed high precision with an estimated false positive rate of less than 1 in 100. Additionally, the technique detected and removed approximately 275,000 near duplicates, suggesting a high recall.

> 为了进一步降低生成器记忆输入样本的可能性，我们使用了一个简单的图片重复数据删除程序。我们为训练样本的中间裁剪部分使用了一个 3072-128-3072 的去噪声的 dropout 正则化 RELU 的 autoencoder 在32x32下采样下。然后通过对 ReLU 激活进行阈值化，对生成的代码层激活进行二值化，这已被证明是一种有效的信息保存技术，并且提供了一种方便的语义散列形式，允许一个线性时间的去重。对哈希碰撞的目视检查显示出高精度，估计误报率不到百分之一。特别的，这种技术删除和移除了接近 275,000 的重复项，按时一个高的召回率（recall）。

**4.2 Faces**

We scraped images containing human faces from random web image queries of peoples names. The people names were acquired from dbpedia, with a criterion that they were born in the modern era. This dataset has 3M images from 10K people. We run an OpenCV face detector on these images, keeping the detections that are sufficiently high resolution, which gives us approximately 350,000 face boxes. We use these face boxes for training. No data augmentation was applied to the images.

> 从随机查询一个人名字的网络图片中刮取包含人类面部的图片，这些人名字是从 dbpedia 中获得的，标准是他们都出生在现代。这个数据集有 3M 的数据，来自 10K 的人。我们运行 OpenCV 脸部检测器在这些图片上，保持一个高分辨率的检测，这给了我们接近 350,000 个脸部区域。我们使用这些脸部区域来训练。没有数据增强被应用到这些图片上。

**4.3 ImageNet-1K**

We use Imagenet-1k (Deng et al., 2009) as a source of natural images for unsupervised training. We train on 32 × 32 min-resized center crops. No data augmentation was applied to the images.

> 我们使用 Imagenet-1k 这个数据集来作为无监督学习自然图片的来源。我们训练在 32 × 32 的最小调整大小的中间裁剪。没有数据增强被使用到这些图片上。

**5 Empirical Validation of DCGANS Capabilities**

**5.1 Classifying CIFAR-10 Using GANs as a Feature Extractor**

One common technique for evaluating the quality of unsupervised representation learning algorithms is to apply them as a feature extractor on supervised datasets and evaluate the performance of linear models fitted on top of these features.

> 有一个普遍的技术为衡量无监督表示学习算法的质量，应用这个算法作为一个特征提取器在有监督的数据集上，和评估拟合在这些特征之上的线性模型的性能。

On the CIFAR-10 dataset, a very strong baseline performance has been demonstrated from a well tuned single layer feature extraction pipeline utilizing K-means as a feature learning algorithm. When using a very large amount of feature maps (4800) this technique achieves 80.6% accuracy. An unsupervised multi-layered extension of the base algorithm reaches 82.0% accuracy (Coates & Ng, 2011). To evaluate the quality of the representations learned by DCGANs for supervised tasks, we train on Imagenet-1k and then use the discriminator’s convolutional features from all layers, maxpooling each layers representation to produce a 4 × 4 spatial grid. These features are then flattened and concatenated to form a 28672 dimensional vector and a regularized linear L2-SVM classifier is trained on top of them. This achieves 82.8% accuracy, out performing all K-means based approaches. 

> 在 CIFAR-10 数据集上，一个非常强的基础性能已经被证明来自一个好的调整过的通过 K-means 通道的单层特征提取器作为特征学习算法。当我们使用了非常巨大的 feature map 时候，这个技术达到了 80.6% 的准确率。基础算法的无监督多层扩展达到了 82.0% 的准确率。为了评估有监督 DCGANs 表示学习的质量，我们在 Imagenet-1k 上训练如何使用鉴别器所有层的卷积特征，最大池化每一层的表示来生成一个 4 × 4 的空间网络。这些特征之后被拉伸和串联起来形成一个 28672 维的向量，并在他们之上训练一个正则化线性 L2-SVM 分类器。

Notably, the discriminator has many less feature maps (512 in the highest layer) compared to K-means based techniques, but does result in a larger total feature vector size due to the many layers of 4 × 4 spatial locations. The performance of DCGANs is still less than that of Exemplar CNNs (Dosovitskiy et al., 2015), a technique which trains normal discriminative CNNs in an unsupervised fashion to differentiate between specifically chosen, aggressively augmented, exemplar samples from the source dataset. Further improvements could be made by finetuning the discriminator’s representations, but we leave this for future work. Additionally, since our DCGAN was never trained on CIFAR-10 this experiment also demonstrates the domain robustness of the learned features.

> 尤其，鉴别器的特征图要少得多，对比基于 K-means 的技术，但由于许多 4 × 4 空间位置的层，确实会导致更大的总特征向量大小。DCGANs 的性能依旧少于 Exemplar CNNs，这个技术是训练正常的具有鉴别能力的 CNNs 在无监督方式来鉴别特定的选择，积极增强，来自源数据的模范样本。进一步的提高可以使用微调鉴别器的表示，但是我们把这个工作留到以后。特别的，我们的 DCGAN 从来没有在 CIFAR-10 上训练过，这个实验同样证明了已学习得特征的域鲁棒性。

**5.2 Classifying SVHN Digits Using GANs as a Feature Extractor**

On the StreetView House Numbers dataset (SVHN)(Netzer et al., 2011), we use the features of the discriminator of a DCGAN for supervised purposes when labeled data is scarce. Following similar dataset preparation rules as in the CIFAR-10 experiments, we split off a validation set of 10,000 examples from the non-extra set and use it for all hyperparameter and model selection. 1000 uniformly class distributed training examples are randomly selected and used to train a regularized linear L2-SVM classifier on top of the same feature extraction pipeline used for CIFAR-10. This achieves state of the art (for classification using 1000 labels) at 22.48% test error, improving upon another modifcation of CNNs designed to leverage unlabled data (Zhao et al., 2015).

> 在 SVHN 上，我们使用来自 DCGAN 的鉴别器的能用于监督目的的特征，当已标记的数据非常稀缺的时候。遵循与 CIFAR-10 实验类似的数据集准备规则，我们分开一个来自非额外数据的 10,000 个验证数据集样本，然后为所有的超参数和模型选择使用这些样本。1000 个统一分类分布训练样本是随机选择的，然后使用这个样本来训练一个正则线性 L2-SVM 分类器，在用于 CIFAR-10 的相同特征提取管道之上。在 state of the art 上获得 22.48% 的测试错误，改进另一个修改的 CNNs 设计来使用未标记的数据。

Additionally, we validate that the CNN architecture used in DCGAN is not the key contributing factor of the model’s performance by training a purely supervised CNN with the same architecture on the same data and optimizing this model via random search over 64 hyperparameter trials (Bergstra & Bengio, 2012). It achieves a signficantly higher 28.87% validation error.

> 特别的，我们证实在 DCGAN 中使用的 CNN 架构不是对模型性能的关键贡献，通过训练具有相同架构相同数据的纯监督 CNN，之后优化这个模型通过随机搜索超过64个超参数实验。这个方法可以达到一个很高的验证率。

**6 Investigating and Visualizing the Internal of the Networks**

We investigate the trained generators and discriminators in a variety of ways. We do not do any kind of nearest neighbor search on the training set. Nearest neighbors in pixel or feature space are trivially fooled (Theis et al., 2015) by small image transforms. We also do not use log-likelihood metrics to quantitatively assess the model, as it is a poor (Theis et al., 2015) metric.

> 我们以多种方式调查训练过的生成器和鉴别器。我们不会在训练数据集上调用任何的最近邻搜索。像素上或者特征空间上的最近邻会被小图片变换愚弄。我们同样不使用log似然方法来定量评估模型，因为这是一个很差的指标。

**6.1 Walking in the Latent Space**

The first experiment we did was to understand the landscape of the latent space. Walking on the manifold that is learnt can usually tell us about signs of memorization (if there are sharp transitions) and about the way in which the space is hierarchically collapsed. If walking in this latent space results in semantic changes to the image generations (such as objects being added and removed), we can reason that the model has learned relevant and interesting representations. The results are shown in Fig.4.

> 第一个做的实验是为了理解潜在空间的样貌。在以学到的 manifold 上行走是的通常可以告诉我们关于记忆的迹象（如果他们是急剧转变的），和知道哪一个空间是分层折叠的。如果在这个潜在空间上行走导致生成图片语义的改变（诸如对象被添加或者移除），我们可以归结于模型依旧学习到相关和感兴趣的表示。Fig.4 太大只截取了部分内容。

![Figure 4](https://img-blog.csdnimg.cn/dcc2d964c0a44134b6585457f4b09160.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 4: Top rows: Interpolation between a series of 9 random points in $Z$ show that the space learned has smooth transitions, with every image in the space plausibly looking like a bedroom. In the 6th row, you see a room without a window slowly transforming into a room with a giant window. In the 10th row, you see what appears to be a TV slowly being transformed into a window.

> 最顶行：$Z$ 中一系列9个值之间插值显示已学得的空间有平滑的转换，在空间中的所有的图片都似是而非的看着像 bedroom。在第六行中，我们可以看见一个没有窗户的房间缓慢的转换到一个房间有巨大的窗户。在第十行中，我们可以一个有 TV 的房间缓慢转换成一个有窗户的房间。

**6.2 Visualizing the Discriminator Features**

Previous work has demonstrated that supervised training of CNNs on large image datasets results in very powerful learned features (Zeiler & Fergus, 2014). Additionally, supervised CNNs trained on scene classification learn object detectors (Oquab et al., 2014). We demonstrate that an unsupervised DCGAN trained on a large image dataset can also learn a hierarchy of features that are interesting. Using guided backpropagation as proposed by (Springenberg et al., 2014), we show in Fig.5 that the features learnt by the discriminator activate on typical parts of a bedroom, like beds and windows. For comparison, in the same figure, we give a baseline for randomly initialized features that are not activated on anything that is semantically relevant or interesting.

> 以前的工作已经证明在 CNNs 上的在大量图片数据集上有监督训练就会形成一个非常强的以学习特征。特别的，有监督的 CNNs 训练在一个场景分类学习目标鉴别器上。我们证明一个无监督的 DCGAN 训练在一个大量图片数据集上，也可以学习到一个感兴趣的特征的层次结构。使用引导反向传播算法，我们在 Fig.5 上显示已经被鉴别器学得的特征层在 bedroom 特别部分的激活，像床或者窗户。为了对比，在一些图片上，我们为随机初始化特征给出了一个，不会被激活在任何和语义相关或者语义感兴趣的一个基础线。

![Figure 5](https://img-blog.csdnimg.cn/899cd9c944b644aba9b90584c0d6c0d7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 5: On the right, guided backpropagation visualizations of maximal axis-aligned responses for the first 6 learned convolutional features from the last convolution layer in the discriminator. Notice a significant minority of features respond to beds - the central object in the LSUN bedrooms dataset. On the left is a random filter baseline. Comparing to the previous responses there is little to no discrimination and random structure.

> 右边，有引导的反向传播可视化关于最大轴对齐响应，关于首6次已学得的卷积特征来自鉴别器的最后一个卷积层。注意到一个明显的关于床的特征反应 - 在中心目标在 LSUN bedrooms 数据集上。左边则是随机的过滤器基线。对比前一个反应（训练过的过滤器那个），（随机过滤器）这里有几乎没有可鉴别的和随机结构。

**6.3 Manipulating the Generator Representation**

**6.3.1 Forgetting to Draw Certain Objects**

In addition to the representations learnt by a discriminator, there is the question of what representations the generator learns. The quality of samples suggest that the generator learns specific object representations for major scene components such as beds, windows, lamps, doors, and miscellaneous furniture. In order to explore the form that these representations take, we conducted an experiment to attempt to remove windows from the generator completely.

> 除了鉴别器所学习到的表示，这里还有一个问题就是生成器的表示是什么。样本的质量显示生成器学习到了特定的目标表示，在主要场景组合诸如床，窗户等等。为了探索这些表示的形式，我们实施一个实验在生成器中试图完全移除窗户。

On 150 samples, 52 window bounding boxes were drawn manually. On the second highest convolution layer features, logistic regression was fit to predict whether a feature activation was on a window (or not), by using the criterion that activations inside the drawn bounding boxes are positives and random samples from the same images are negatives. Using this simple model, all feature maps with weights greater than zero ( 200 in total) were dropped from all spatial locations. Then, random new samples were generated with and without the feature map removal.

> 在150个样本中，52个窗户框被手动画出。在第二高的卷积层表示中，逻辑回归被用于预测哪一个特征激活在窗户上，通过使用在绘制的边界框内的激活是正数而来自相同图像的随机样本是负数的标准。使用这个简单的模型，所有的 feature maps 的权重大于零（总数为200）被丢弃在所有空间位置上。之后，随机新的样本被生成有或者没有 feature map 删除。

The generated images with and without the window dropout are shown in Fig.6, and interestingly,
the network mostly forgets to draw windows in the bedrooms, replacing them with other objects.

> 生成图片有或者没有窗户丢弃的在 Fig.6 中显示，之后有趣的是网络大多数忘记了画出窗户在 bedroom 中，而是通过其他对象代替。

![Figure 6](https://img-blog.csdnimg.cn/8a252e39506a49cb90ba63acab662770.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 6: Top row: un-modified samples from model. Bottom row: the same samples generated with dropping out ”window” filters. Some windows are removed, others are transformed into objects with similar visual appearance such as doors and mirrors. Although visual quality decreased, overall scene composition stayed similar, suggesting the generator has done a good job disentangling scene representation from object representation. Extended experiments could be done to remove other objects from the image and modify the objects the generator draws.

> 最上面为未更改的来自模型的样本。下面的一行未被丢弃了窗户过滤器的生成样本。一些窗户已经被移除，而其他的一些责备转换成了视觉上相似的对象，如门或者镜子。虽然视觉质量减少了，但是整体场景构成保持相似，显示生成器在通过对象表示来解构场景表示上做得很好。扩展实验可以通过删除其他对象来自图片和更改生成器画出对象来完成。

**6.3.2 Vector Arithmeic on Face Samples**

In the context of evaluating learned representations of words (Mikolov et al., 2013) demonstrated that simple arithmetic operations revealed rich linear structure in representation space. One canonical example demonstrated that the vector(”King”) - vector(”Man”) + vector(”Woman”) resulted in a vector whose nearest neighbor was the vector for Queen. We investigated whether similar structure emerges in the $Z$ representation of our generators. We performed similar arithmetic on the $Z$ vectors of sets of exemplar samples for visual concepts. Experiments working on only single samples per concept were unstable, but averaging the $Z$ vector for three examplars showed consistent and stable generations that semantically obeyed the arithmetic. In addition to the object manipulation shown in (Fig.7), we demonstrate that face pose is also modeled linearly in $Z$ space (Fig. 8).

> 在衡量学习得的词语表示语境下证明简单的算数运算透露富线性结构在表示空间。一个典型样本证明 vector(”King”) - vector(”Man”) + vector(”Woman”) 作用于一个 Queen 向量最近邻的向量。我们调查类似的结构是否出现在我们的生成器 $Z$ 表示中。我们使用相似的结构算法在 $Z$ 向量中，在一系列的视觉概率的模型样本中。实验仅仅在一个概念一个样本上进行是不稳定的，但是平均 $Z$ 向量对于三个实验显示语义上服从算术的一致且稳定。除了图 Fig.7 中显示的对象操作，我们还证明了脸部模型也在 $Z$ 中线性建模。

![Figure 8](https://img-blog.csdnimg.cn/7d3d40a7aac54a0cb0049a8aa54dcb5f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 7: Vector arithmetic for visual concepts. For each column, the $Z$ vectors of samples are
averaged. Arithmetic was then performed on the mean vectors creating a new vector $Y$ . The center sample on the right hand side is produce by feeding $Y$ as input to the generator. To demonstrate the interpolation capabilities of the generator, uniform noise sampled with scale +-0.25 was added to $Y$ to produce the 8 other samples. Applying arithmetic in the input space (bottom two examples) results in noisy overlap due to misalignment.

> 视觉概念的向量算法。对于每一列，样本的 $Z$ 向量是平均的。算法之后在平均向量上来创建一个新的向量 $Y$。右侧的中心样本是通过将 $Y$ 作为输入提供给生成器来生成的。为了演示生成器的插值能力，将比例为 +-0.25 的均匀噪声采样添加到 $Y$ 以生成其他 8 个样本。在输入空间中应用算术（下两个示例）会由于未对齐而导致噪声重叠。

![Figure 8](https://img-blog.csdnimg.cn/a264702ef4ff4b51b3f6ec42dd0ddf9d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaXNpbnN0YW5jZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
Figure 8: A ”turn” vector was created from four averaged samples of faces looking left vs looking right. By adding interpolations along this axis to random samples we were able to reliably transform their pose.

> 一个转向向量是由四个向左看和向右看的脸的平均样本创建的。通过沿该轴向随机样本添加插值，我们能够可靠地转换它们的姿势。

These demonstrations suggest interesting applications can be developed using $Z$ representations learned by our models. It has been previously demonstrated that conditional generative models can learn to convincingly model object attributes like scale, rotation, and position (Dosovitskiy et al., 2014). This is to our knowledge the first demonstration of this occurring in purely unsupervised models. Further exploring and developing the above mentioned vector arithmetic could dramatically reduce the amount of data needed for conditional generative modeling of complex image distributions.

> 这证明了可以使用学习得的 $Z$ 表示来开发感兴趣的应用。前面已经证明有条件的生成模型可以学习成一个令人信服的模型对象诸如规模，旋转和位置属性。这是我们的知识首次证明这发生在完全无监督的模型。未来的探索和发展上面提到的向量算法可以戏剧性的减少复杂图像分布上有条件生成模型的数据数量。

**7 Conclusion and Future Work**

We propose a more stable set of architectures for training generative adversarial networks and we give evidence that adversarial networks learn good representations of images for supervised learning and generative modeling. There are still some forms of model instability remaining - we noticed as models are trained longer they sometimes collapse a subset of filters to a single oscillating mode.

> 我们提出一个更稳定的一系列架构，为生成对抗网络和我们给出证据关于对抗网络学习到一个很好的表示对于图片在有监督学习和生成模型中。这里依旧有一些模型不稳定的形式存在，我们注意到，随着模型的训练时间更长，它们有时会将过滤器的子集折叠为单个振荡模式。

Further work is needed to tackle this from of instability. We think that extending this framework to other domains such as video (for frame prediction) and audio (pre-trained features for speech synthesis) should be very interesting. Further investigations into the properties of the learnt latent space would be interesting as well.

> 未来的工作要处理这个不稳定性的问题。然后要扩展这个架构到其他领域诸如视频或者音频应该很有趣。未来的关于隐空间的特性探索也会很有趣。

# 论文的研究背景
**1. 本论文解决什么问题？（能否通过一个示例来说明拟解决的问题）**

本文解决了 GANs 存在的训练不稳定问题。

**2. 关于该问题，目前的相关工作有哪些？这些相关工作有何优缺点？（综述相关工作）**
相关工作：

未标记数据上的无监督学习目前相关工作

1. 在数据集上先进行聚类数据，之后用聚类出来的簇来优化分类。
2. 还有使用了自动编码器。
3. 还有深度信念网络。

生成自然图片目前相关工作

1. 一种是无参数模型，主要从已有数据集中匹配局部内容，已经被用于纹理合成、高分辨率和画中画内容。
2. 还有就是有参数模型。

CNNs 内部可视化

1. 有 Zeiler 使用反卷积和过滤最大激活来发现每一层试图所做的事情。
2. 还有 Mordvintsev 试图在输入图片中做梯度下降来发现图片激活了那些特定部分。

本文未说明优缺点。

# 论文的主要研究内容
**1. 针对已有工作的不足之处，本文提出了什么方法？（该方法为何有效？）该方法的基本思路是什么？主要创新点在哪？**

提出的方法就是 DCGAN，思路和创新点就是通过研究和可视化了 GANs 和 GANs 中间层学到了什么内容，来优化 GANs 网络的训练。

**2. 阐述本文提出方法的技术细节**

第一个是全卷积网络，通过大步的卷积替代了确定性的空间池化函数（如最大池化），允许网络学习到它自己的空间下采样。我们使用这个方法在我们的生成器中，允许它学习自己的空间上采样和鉴别器。

第二个是在卷积特征之上消除全连接层的趋势。其中最有力的样本是全局平均池化，这种方法已经在 state of the art 图片分离模型中使用了。我们发现全局平均池化提高了模型的稳定性，但是伤害了收敛速度。将最高卷积特征分别直接连接到生成器和鉴别器的输入和输出的中间地带效果很好。GAN 的第一层，使用一个均匀噪声分布 $Z$ 作为输入，可以被视作全连接因为这只是矩阵乘法，但是其结果被改造成4维张量，作为卷积的输入。而鉴别器，最后卷积层被拉伸然后输入到单个 sigmoid 函数中。

第三是批量标准化，通过将每个单元的输入归一化为零均值和单位方差来稳定学习。这帮助处理训练中因为缺乏初始化导致的问题和帮助更深模型中的梯度流动。这证实了获得深度生成器在学习开始时，避免生成器从所有样本中崩溃到单个点的问题，这个问题是 GANs 的普遍问题。直接的应用批量标准化到所有层中。会导致样本震荡和模型不稳定。这可以通过不使用批量标准化到生成器输出层和鉴别器的输入层来避免。

# 论文的实验结果
**1. 阐述本文的实验内容**

实验太多，这里略。

**2. 本文方法的有效性是如何通过实验进行验证的？**

通过添加和移除一些特征表示来观察输出结果发现的确该方法已经发现 GANs 所学到的真实的对象（移除这些对象的表示将会导致输出缺少这些对象），和向量加减法结果来观察最近邻向量是否具有可解释性（向量可以通过交并集来生成符合人类逻辑的输出）等等。

# 论文存在的不足之处
**1. 通过阅读此论文，你能否找到本文工作存在的不足之处？**

无。

**2. 试阐述解决这些不足之处的基本思路？**

无。