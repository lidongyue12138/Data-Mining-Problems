# Data-Mining-Problems

### Learning and Search

**1) Why all learning problems are inverse problems, requiring unbounded exhaustive searches, thus ill-posed?** 

trying to solve inverse problems (from data to model) and infer general rules from few data 

因为数据挖掘处理的数据往往有着维数高，多模型混合生乘，时变等复杂特征，当我们通过⼀个模型去拟合数据时，肯定忽略了数据的⼀些复杂特征，或者说， 不管我们采用哪种模型，都不可能保证完全形容出数据的所有特征。而且在我们选择模型的时候，也不可能从所有的模型中选择最好的，因为穷举的复杂度是极高的。

**2) Why gradient is the key mathematical assumption that we could count on in order to search? What would be the general implications for the existence of at least some continuality or locality?**

**3) What is the generalizability of a mathematical process, from both expressive (smoothness) and inclusive (capacity) point of views?**

**4) What would be some of the solutions for such an ill-posed problem in order to yield at least some reasonable results?**

**5) What are some of the mathematical hurdles that have prevented more generalizable solutions?**

**6) Why variable dependences (interactions) could become an extremely difficult and even an impossible problem? Give philosophical, mathematical, physical, computational, and numerical examples for such a singularity.**  

**7) Why a Euclidian-based measure would be most favored but usually impossible to obtain for a real world issue?**

**8) What are some of the key requirements for a real issue to be formulated as a Euclidian problem?**

**9) What would be the mathematical alternative frameworks to translate a non-Euclidian problem to mathematically appropriate solutions?**

**10) Why in general the complex and high-dimensional data (the so-called big data problem, n<<p) from the same “class” tend to have a low dimensional representation?**      

dimensionality reduction. Regularization. low D solution to high D problem. 

**11) Why we would prefer a low complexity model for a high complex problem?**

**12) What is a loss function? Give three examples (Square, Log, Hinge) and describe their shapes and behaviors.** ⭐️

![微信图片_20190420110309](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420110309.png)

**13) Using these losses to approach the linear boundary of a overlapping problem, inevitably some risks will be incurred; give two different approaches to remedy the risk using the SVM-based hinge loss as an example.** ⭐️

- Use soft-margin
- Use $L_1$ or $L_2$ Norm to reduce dimension
- Use kernel method

**14) Describe biases (under-fitting) and variance (over-fitting) issue in learning, and how can we select and validate an appropriate model?** ⭐️

![微信图片_20190420110834](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420110834.png)

**15) How to control model complexity in the regression of a linear model? Are there supposed to be a unique low-dimensional model for a given high dimensional problem?**

**16) Using the Least Square as the objective function, we try to find the best set of parameters; what is the statistical justification for the Lease Square if the underlying distribution is Gaussian?** ⭐️

![微信图片_20190420111014](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111014.png)

![微信图片_20190420111017](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111017.png)

**17) Could you describe the convexity as to how it would facilitate a search? Using the Least Square-based regression and Likelihood-based estimation as the examples?** ⭐️

Convexity means the global optimum is unique and we can use Gradient-based method easily to find it. 

**18) Gradient Decent has a number of different implementations, including SMO, stochastic methods, as well as a more aggressive Newton method, what are some of the key issues when using any Gradient-based searching algorithm?**​ ⭐️

- The value of hyper-parameters like learning rate (step size). How to jump out of the local minimum.
  The convexity of the problem. 
- parallel computation and speed up.

**19) What are the five key problems whenever we are talking about a learning process (Existence, Uniqueness, Convexity, Complexity, Generalizability)? Why are they so important? ​**⭐️

- Existence shows whether our model can converge.
- Uniqueness shows the difficulty of training. 
- If the problem is convex, we can solve it easily and the global minimum always exists and is unique.
- Complexity shows the cost of training.
- Generalizability shows whether our model can achieve good result in new in test dataset and is robust in solving real world problem 

**20) Give a probabilistic interpretation for logistic regression, how is it related to the MLE-based generative methods from a Bayesian perspective?**​ ⭐️

![微信图片_20190420111505](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111505.png)

**21) What are the mathematical bases for the logics regression being the universal posterior for the data distributed in any kinds of exponential family members?** 

**22) Can you provide a probabilistic comparison for liner and logistic regression?**

**23) Why the log of odd would be something related to entropy and effective information?**

**24) Why often we want to convert a liner to a logistics regression, conceptually and computationally?**

**25) Compare the generative and discriminative methods from a Bayesian point of view?** ⭐️

![微信图片_20190420111654](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111654.png)

**26) What are the most important assumption for something Naïve but still very effective? For instance for classifying different documents?**

**27) What would be the most effective way to obtain a really universal prior? And what would be the most intriguing implications for human intelligence?** 

**28) What are the key advantages of linear models? But why linear model tends not expressive?** ⭐️

![微信图片_20190420111831](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111831.png)

29) What are the key problems with the complex Neural Network with complex integrations of non-linear model? 

**30) What are three alternatives to approach a constrained maximization problem?**⭐️

1. Solving its dual problem (Lagrange Multiplier)
2. Find its equivalent problems (modify objective function)
3. Using kernel tricks 

**31) What is the dual problem? What is strong duality?**⭐️

![微信图片_20190420111954](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111954.png)

![微信图片_20190420111958](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111958.png)

**32) What are the KKT conditions? What is the key implication of them? Including the origin of SVM?**⭐️

![微信图片_20190420112145](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112145.png)

![微信图片_20190420112149](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112149.png)

![微信图片_20190420112300](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112300.png)

**33) What is the idea of soft margin SVM, how it is a nice example of regularization?**⭐️

![微信图片_20190420112335](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112335.png)

**34) What is the idea of kernel? Why not much additional computational complexity?** ⭐️

![微信图片_20190420112425](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112425.png)

**35) What is the general idea behind the kernel? What key computation do we perform? Why is it so general in data modeling?**​ ⭐️

![微信图片_20190420112505](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112505.png)

**36) Why we often want to project a distance “measure” to a different space?**  ⭐️

In many situations, the real data space is usually non-Euclidean. Therefore, we want to project a distance measured in Euclidean space to its real space. (nonlinear $\Rightarrow$ linear) 

**37) What a Turin machine can do? What some of the key computable problems a Turin machine can do?**

**38) What a Turin machine cannot do? What some of the key computable problems a Turin machine cannot do?**

**39) Give your own perspectives on the Hilbert No.10 problem in the context of computational limit.**

**40) Give your own perspectives on the Hilbert No.13 problem in the context of mathematical limit.**

**41) Discuss human intelligence vs. Turin machine as to whether they are mutually complementary, exclusive, or overlapping, or contained into each other one way or another.**  

**42) Explain Bayesian from a recursive point of view, to explain the evolution of human intelligence.** 

**43) What are computational evolutional basis of instinct, attention, inspiration, and imagination?**

**44) Explain the core idea of machine learning to decompose a complex problem into an integration of individual binary problems, its mathematical and computational frame works.**  

**45) What are the limitation of Euclidian (Newtonian) basis, from the space, dimension, measure point of view?**

**46) Why differentials of composite non-linear problems can be very complex and even singular?**  

**47) What is the basis of Turin halting problem? And why temporal resolution (concurrency) is a key for logics and parallelism?** 

**48) What is mathematical heterogeneity and multiple scale?** 

**49) Explain convexity of composite functions? How to reduce local solutions at least in part?**

**50) Why local solution is the key difficulty in data mining, for more generalizable learning?**  

### Probabilistic graphical model

**1) Compare the graphical representation with feature vector-based and kernel-based representations;**⭐️

![微信图片_20190420112632](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112632.png)

**2) Explain why sometime a marginal distribution has to be computed in a graphical model;** ⭐️

![微信图片_20190420112834](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112834.png)

**3) Why class labels might be the key factor to determine if presumptively different data distributions can be indeed discriminated?** 

**4) Why knowledge-based ontology (representation) be a possible solution for many prior-based inference problems?**  005:03:48

**5) Why a graphical model with latent variables can be a much harder problem?**⭐️

![微信图片_20190420113044](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113044.png)

**6) What is the key assumption for graphical model? Using HMM as an example, how much computational complexity has been reduced because of this assumption?**⭐️

![微信图片_20190420113125](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113125.png)

**7) Why does EM not guarantee a global solution? What is a simple proof for that?**⭐️

![微信图片_20190420113206](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113206.png)

**8) Why is K-mean only an approximate and local solution for clustering?**⭐️

![微信图片_20190420113331](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113331.png)

**9) How to interpret the HMM-based inference problem from a Bayesian perspective, using the forward/backward algorithm?**⭐️

![微信图片_20190420113417](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113417.png)

**10) Show how to estimate a given hidden state for a given series of observations using the alpha and beta factors;**  ⭐️

![微信图片_20190420113506](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113506.png)

**11) How a faster inference process would be constructed, given a converging network?**

**12) How can an important node (where inference can be significantly and sensitively affected) be detected using an alpha and a beta process?**

**13) Why often an alpha process (forward) is more important than beta (backward)?**

**14) What are the key differences between an alpha and a beta process from human and machine intelligence point of views?**

**15) How data would contribute to the resolution of an inference process from a structural point of view?**

**16) For a Gaussian graphical model, what is the implication of sparsity for such a graphical model? How is such sparsity achieved computationally?**⭐️

- Most edges have zero weights. i.e. sparse adjacent matrix. (Pruning)
- Applying SVD in dimension reduction. Simplify the computation. 

**17) Explain the objective function of Gaussian graphical model? Thus the meaning of MLE?**

**18) How a row or column-based norm (L1 or L2) can be used to construct hub-based models? And why this might be highly applicable?**

**19) How Gaussian graphical model be used to model a temporally progressive problem? Why still a simple 2d matrix compression problem can still be very demanding computationally?**  006:06:10

**20) Why a spectral or nuclear norm would be better than Frobenius norm to obtain a sparse matrix?**  



### Dimension reduction and feature representation

**1) PCA is an example of dimensional reduction method; give a full derivation of PCA with respect to its eigenvectors; explain SVD and how it is used to solve PCA;**⭐️

![微信图片_20190420113803](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113803.png)

Answer from Newly:

- Interpretation of PCA: 

![微信图片_20190420163253](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420163253.png)

- Interpretation of SVD:

![微信图片_20190420163353](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420163353.png)

![微信图片_20190420163452](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420163452.png)

![微信图片_20190420164654](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420164654.png)

![微信图片_20190420164658](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420164658.png)

![微信图片_20190420164703](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420164703.png)

**2) Compare regular PCA with the low-ranked PCA, what would be advantage using the low-ranked PCA and how it is formulated?**⭐️

![微信图片_20190420113847](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113847.png)

Extend reading: SPCA can restrict the process of linear combination when calculate the eigenvectors. So that the results have a stronger real-world meaning and easier to be explained.
Sparse PCA: https://blog.csdn.net/zhoudi2010/article/details/53489319 

![微信图片_20190420121816](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420121816.png)

SPCA原始文献：[H. Zou (2006) Sparse principal component analysis](http://www.tandfonline.com/doi/abs/10.1198/106186006X113430) 

**3) What is the difference between a singular value and its Eigen value? Explain the resulting singular values of a SVD for how the features were originally distributed;** 006:18:27

Answer from Newly:

![微信图片_20190420171110](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420171110.png)

![微信图片_20190420171114](C:\Users\54942\Desktop\DM_Fina.g)

Question2:

Two scenarios for how the features might be distributed:

- previous feature independent from each other. Original features are informative. PCA is not very effective. 
- features with a lot of redundancy. PCA has to be effective. 

**4) What is the key motivation (and contribution) behind deep learning, in terms of data representation?** ⭐️

Deep learning is a kind of data representation learning. It automatically discovers the representations needed for feature detection or classification from raw data. This replaces manual feature engineering and allows a machine to both learn the features and use them to perform a specific task. And DL does well in complex representation.

Split and combination. Learn complex, abstract feature combination. 

![微信图片_20190420121936](C:/Users/54942/Desktop/DM_Final/Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420121936.png)

- Motivation: 
  1. Decompositional approach: multiple building blocks that learns how original data recombine together.
     2. interpret underlying mechanism, underlying structure, and underlying logics
  2. universal approximation theorem
- Contribution: 
  - automatically discovers the representations needed for feature detection or classification from raw data
  - complex representation for the structure of the data

**5) Compare the advantage and disadvantage of using either sigmoid or ReLu as an activation function?**

Loss: 

- Sigmoid: output probabilistic results. But it has gradient vanishing problems. 
- ReLU: 
  - avoid gradient vanishing problems. Computing gradient is faster.  
  - $x$ becomes arbitrary. Use $b$ (斜率还是截距) to calibrate each activation function. 
  - Combination of ReLU functions is less expressive, because the ReLU function is simple and of low-order. 

**6) Discuss matrix decomposition as a strategy to solve a complex high dimensional problem into a hierarchy of lower dimensional combinations?**

Mathematical justification for deep learning.

(YB writes something on the blackboard...which explains this problem. Something about $n\log n$ overhead for $n^2$ capacity)

Pruning/Dimension Reduction: Sparsity.

**7) Discuss convexity of composite functions, what mathematical strategies might be used to at least reduce local solutions?**     

DL: vertically product of composite functions and horizontally summation of them. This leads to many local solutions.

Strategies to reduce local solutions:

- ReLU: simplify combination.
- Partition neural networks

**8) Why normally we use L2 for the input layers and L1 for the actual modeling? Explain why still sigmoid activations are still used for the output layers?** 

Input layers of L2 and hidden layers of L1:

- L2: smooth inputs. Because some data/ features still are really big. L2 cancel to relative weight and average inputs (not to different from each other)
- L1: for sparsity. Keep pruning connections between the layers. Sparse networks are computationally tolerant.
- Sigmoid (softmax): renormalize and recalibrate the outputs. Get probabilistic outputs (Logistic Loss) and try to evaluate from distance from distribution or entropy point of view. Sometimes we use $tanh()$

**9) What would be the true features of an object modeling problem? Give two examples to highlight the importance of selecting appropriate dimensions for feature representations;**⭐️

Have strong information, low-rank
DL can dig out the underlying relationship within the data and get the "true" feature
Problems: Basis function selection, ways of combination (topology), BP(gradient vanish and explode), memory (RNN, LSTM), Computational complexity, Over/Under-fitting 



head location: reconstruct 2D pic to 3D space.
truth table: only 3 dimensions are critical other boolean variables are useless. (dimension reduction and
feature selection) 

YB's own answer: Two examples

- Posture of human face: high-dimension image and two/ three-dimension representation is good.
- Simulate Boolean algorithm: how many sigmoid functions you need? Three gates that construct 8 bits truth tables. It is three-dimensional (I don't know what this is about)

**10) Why does the feature decomposition in deep learning then a topological recombination could make a better sampling? What would be the potential problems making deep learning not a viable approach?** 006:40:53

DL is able to decompose data into different primitive parts. These parts cam recombine those other parts. This exponentially or factorially increase the data for possible combinations. This increase the sample capacity and can generate sample $\Rightarrow$ GAN

Potential problems: (on slides)

- Local solutions
- Vanishing gradients
- ...

**11) Explain the importance of appropriate feature selection being compatible with model selection in the context of model complexity;**⭐️ 006:44:56

The feature should be compatible with the representation ability of the model. Otherwise, over-fitting or under-fitting will be likely to happen.

Higher dimensional data are more likely to over-fitting. 

The feature selection should match the model selection. 

**12) What would be the ultimate and best representation for a high dimensional and complex problem? How this might be possibly achieved?** 006:46:14

Everything should be modular and re-useful. Everything should be combination of building blocks. Problems should be reorganized by the combination of the structural, logical, and conceptual parts instead of case by case. (Key word: modular combination)

**13) How RNN can be expanded our learning to fully taking advantage of Turing machine? What RNN can whereas CNN cannot do?** 006:47:38

RNN & CNN:

RNNs are more capable of achieving something beyond simulation of functions. RNN begins to simulate some logical process (memorizing, decision, and stopping). RNN learns not only a process function but also Turing process where logics can be simulated (recursions...). CNN cannot do this. 

From Turing point of view: CNN just changes the input and get different output. CNN does not consider how to connect them in a logical, recursive, parallel, and interactive process (No logic and stops for input and output relationship). RNN expands that (Theorem). 

**14) What is the central additional difficulty of RNN compared to CNN?**  006:50:24

Difficulty:

- RNN is a heterogenous and multiscale system. RNN blocks are more individual, and we need to combine them together. RNN itself has local loss. The problem is how to combine local loss (linearly/weighted). there is no global loss. 

**15) In the activation function, there is a constant term “b” to learn, why it is important?**

(Sorry. The audio does not contain this part...)

**16) LSTM integrate short and long term processes, what is the central issue to address to achieve at least some success?** 007:00:09

- Local & global loss

**17) The difference between value-based vs. policy-based gradients?** 

In reinforcement learning: value-based v.s. policy-based （not covered  in test)

**18) Why dynamical programming might not be a good approach to select an optimal strategy?**

（not covered in test)

**19) Explain an expectation-based objective function?**

（not covered  in test)

**20) Explain Haykin’s universal approximation theorem?**

（not covered  in test)

### General problems

**1) In learning, from the two key aspects, data and model, respectively, what are the key issues we normally consider in order to obtain a better model?** ⭐️

- model selection
- feature selection
- model/ feature compatible
- dimensionality reduction
- model robustness 
- sample complexity
- model complexity 

**2) Describe from the classification, to clustering, to HMM, to more complex graphical modeling, what we are trying to do for a more expressive model?**⭐️ 007:03:43

Humans solve complex problems using priori, while machines do so using combinations of basic functions
Using model combination to get a more complex and expressive model, and also fit more complex problem. 

From less expressive to more expressive (increase of model expressiveness): 

- For classification: we use $\{+1, -1\}$ or $ \{0, 1\}$
- Cluster: we look at $\{0, 1\}$ in terms of their combination $\Rightarrow$ Marginal
- HMM: model process as a inference and condition that can be updated

**3) What are the potential risks we could take when trying to perform a logistic regression for classification using a sparsity-based regularization?**⭐️ 007:06:14

L0: NP-Complete problem
L1: 

- overlooking certain parameters due to different order of data. (sequential risk) Because we need to pick which dimension to maintain instead of by itself. L1 Norm could somehow throw away important dimensions. 
- we also risk of underfitting

For linear model:

- sequence risk
- measures is not consistent
- model risk: λ can be either too great or too small, causing under fitting or over fitting problems. 

**4) Give five different structural constrains for optimization with their corresponding scalars;**⭐️ 007:08:45

- L1 norm: $||x||_{L1} = \sum_{r=1}^{n} |x_i|$ 

  more aggressive

- L2 norm: $||x||_{L2} = \sqrt{\sum_{r=1}^{n} x_i^2}$

  more smooth (想象两者的图像)

Matrix Norm
$$
||A||_p = \left(\sum_{i}^{\min\{m,n\}} \sigma_i^p(A) \right)^{1/p}
$$

- Frobenius norm ($p = 2$): $||A||_{F} = \sqrt{\sum_i^{m}\sum_j^n a_{ij}^2}$
- Nuclear norm ($p=1$): the sum of singular value
- Spectrum norm ($p = \infty$): maximum singular value

**5) Give all universal, engineering, and computational principles that we have learned in this course to obtain both conceptually low-complexity model and computationally tractable algorithms?**

Locality, gradient, linearity, convex, low-rank, combination, binary, priori (Bayes), Markov, expectation,
recursion, measure 

YB's audio answer:

- Locality
- Convexity
- Linearity
- Sparsity: reduce complexity
- Low-rank representation: reduce dimension
- Prior (Bayes)
- Markov
- Entropy
- Gaussian

**6) Why data representation is at least equally as important as the actual modeling, the so-called representation learning?** (YB skipped this)

model selection and feature selection are closely associated with one another; data representation needs to be compatible with the model and capture necessary features learning the combination of features and the relationships between features

Note: machine learning tasks such as classification often require input that is mathematically and computationally convenient to process

**7) How does the multiple-layer structure (deep learning) become attractive again?** (YB skipped this)

People realized that structure cannot be imposed on models, so deep learning first learns the structure of the data and finds the relationships. With more layers, the representation of features becomes richer (?)

Deep learning resolves nonlinear thing with multiple linear combination.

Note: the increase in computational resources and the utilization of GPU acceleration, big data

**8) Discuss Turin Completeness and the limit of data mining;** 

Think for the rest of your life. 

Limits:

- Singular issue
- Multiscale
- Local solution

Leads data mining to be empirical and heuristic

**9) Discuss general difficulties of using gradient for composite functions or processes;** (YB skipped this)

**10) What is the trend of machine learning for the next 5-10 years?** (YB skipped this)

deep learning theory (???)
parallel (???)
reinforcement learning (???) 

### Previous Exam

In one problem: one part is substantial and one part is general

**1)     SVM is a linear classifier with a number of possible risks to be incurred, particularly with very high dimensional and overlapping problems. Use a simple and formal mathematics to show and justify (a) how a margin-based liner classifier like SVM can be even more robust than Logistic regression? (b) how to control the overlapping boundary?** 

- Write loss function for SVM, soft margin, lagrangian

Question 1:

- Dual approach: SVM use the inner product to measure similarity

Question2:

- Soft margin: Formulation. More robust.
- Kernel method: Formulation. Nonlinear separation.

**2)     Why a convolution-based deep learning might be a good alternative to address the dilemma of being more selective towards the features of an object, while remaining invariant toward anything else irrelevant to the aspect of interests? Why a linear regression with regulations would result in features which are usually conceptually and structurally not meaningful?**  

Question 1:

- Convolution averages the data. CNN focus not only on the features but also the background. 
- Convolution measures the pattern of the data. (context) 

Question 2:

- Features are combinations of patterns (context). Pattern/ context v.s. the individual feature

**3)     There are a number of nonlinear approaches to learn complex and high dimensional problems, including kernel and neural networks. (a) please discuss the key differences in feature selection between these two alternatives, and their suitability; (b) what are the major difficulties using a complex neural network as a non-linear classifier?** 007:28:18

Question 1:

- DL decompose them into building blocks and learn how to construct them. 
- DL understand the abstraction of the problem and the context.
- DL's risk is mathematical: local solution...
- Kernel emphasize the training and testing of the data. (data driven approach)

Question 2: 

- Composite function & its differentials
- local solution
- gradient vanishing
- singularity

Both problems are exhaustive (inverse and ill-posed).

**4)     For any learning problems, (a) why a gradient-based search is much more favorable than other types of searches? (b) what would be the possible ramifications of having to impose some kinds of sequentiality in both providing data and observing results?** 007:32:44



**5)     Please use linear regression as the example to explain why L1 is more aggressive when trying to obtain sparser solutions compared to L2? Under what conditions L1 might be a good approximation of the truth, which is L0?**   



**6)     What is the key difference between a supervised vs. unsupervised learnings (where we do not have any ideas about the labels of our data)? Why unsupervised learning does not guaranty a global solution? (use mathematical formulas to discuss).**       



**7)     For HMM, (a) please provide a Bayesian perspective about the forwarding message to enhance an inference (using a mathematical form to discuss); how to design a more generalizable HMM which can still converge efficiently?**



**8)     Using a more general graphical model to discuss (a) the depth of a developing prior-distribution as to its contribution for a possible inference; (b) how local likelihoods can be used as the inductions to facilitate the developing inference?**



**9)     Learning from observation is an ill-posed problem, however we still work on it and even try to obtain convex, linear, and possibly generalizable solutions. Please discuss what key strategies in data mining we have developed that might have remedied the ill-posed nature at least in part? Why in general linear models are more robust than other more complex ones?**



**10)   Using logistic regression and likelihood estimation for learning a mixture model (such as the Gaussian Mixture Model), please using Bayesian perspective to discuss the differences and consistencies of the two approaches; why logistic function is a universal posterior for many mixture models?**

# Technical Problems

1. SVM formulations

2. PCA formulation, SVD formulation, and eigenvalue decomposition.

3. What is Sparse PCA? What is low-rank PCA?

   [Sparse Principal Component Analysis](<https://www.tandfonline.com/doi/pdf/10.1198/106186006X113430?needAccess=true>)

   [Sparse PCA through Low-rank Approximations](<http://proceedings.mlr.press/v28/papailiopoulos13.pdf>)

4. 