
# A report in Mixture of Experts
## 1. Overview
&ensp; An MoE model contains a set of expert network $E_1, E_2,..., E_n$ and a gating function $G$. Each expert specializes in different inputs that are decided by the gating function:
$$
\begin{equation}
    MoE(x) = \sum_{i=1}^n g_i(x)\cdot E_i(x).\tag{MoEs}
\end{equation}
$$
where $E_i(x)$ refers to the output of ith expert. The gating function can be implemented as a random selector or a neural network. In the context of attention-based models, the experts are represented by the 2-layers-FFN (feed-forward neural network) in transformer layers. 
&ensp;   It is easy to notice some resemblances between MoE and ensemble learning which also envolves the use of two or more models to process input and then agregate the outputs with convex combination or voting. However, MoEs differs in a way that it introduces a **routing function** called **"gate"** who is responsible for allocating each input to its expert of preference. 
<div style="text-align:center">
<img style="left;" src="https://i0.wp.com/bdtechtalks.com/wp-content/uploads/2020/11/ensemble-machine-learning-models.jpg?resize=696%2C392&ssl=1" width="400"/>
<img style="right;" src="https://deepgram.com/_next/image?url=https%3A%2F%2Fwww.datocms-assets.com%2F96965%2F1695407447-image1.png&w=1920&q=75" width="350"/>
</div>

Current researches mostly use MoEs structure as a replacement to the MLP layers in Transformer models. 
<div style="text-align:right">
<img style="center;" src="https://d3i71xaburhd42.cloudfront.net/ca086f4c09cf8de705830ac2b70951737fab93ca/1-Figure1-1.png" width="800"/>
</div>

## 2. Routing strategies
<div style="text-align:center">
<img style="center"; src="https://d3i71xaburhd42.cloudfront.net/ca086f4c09cf8de705830ac2b70951737fab93ca/3-Figure2-1.png" width="500"/>

 <em> <strong> An example of the top-k token routing scheme over five experts and three input tokens. </strong> </em>
</div>

### 2.1. Densed routing
$$
\begin{equation}
g(x) = Softmax(Ax + b) \tag{Sm-gate}
\end{equation}
$$
### 2.2. Top-K
$$
\begin{equation}
g(x) = Softmax(Topk(Ax + b ), k) \tag{Topk-gate}
\end{equation}
$$
### 2.3. VMoe routing
$$
\begin{equation}
g(x) = Topk(Softmax(Ax + b + \epsilon), k) \tag{Vmoe-gate}
\end{equation}
$$
 
### 2.4. SMoe routing
$$
\begin{equation}
g(x) = Topk(Softmax(Ax\epsilon  + b ), k) \tag{Smoe-gate}
\end{equation}
$$

### 2.5. Expert Choice Routing
For a mini-batch $\mathcal{B}$, denote $\vert \mathcal{B} \vert$ as the cardinallity of the batch,
$$
\begin{equation}
G_{\mathcal{B}}(x) = Topk(Softmax(A X_{\mathcal{B}} + b ), k'), \tag{EC-gate}
\end{equation}
$$
where $k' = \vert \mathcal{B}\vert*k/n$.

<div style="text-align:center">
<img style="right"; src="https://d3i71xaburhd42.cloudfront.net/ca086f4c09cf8de705830ac2b70951737fab93ca/8-Figure6-1.png" width="500"/>

 <em> <strong> Three common classes of routing algorithms. </strong> </em>
</div>

### 2.6. X-Moe routing
$$
\begin{equation}
g(x) = Topk\left(Softmax\left(\dfrac{APx}{\Vert A \Vert_2 \Vert Lx \Vert_2 \tau}\right), k\right) \tag{Xmoe-gate}
\end{equation}
$$
### 2.7. Randomized routing
Zuo et al. (2021) introduced THOR, an algorithm which randomly selects two experts for each input during training and inference and found improvements of 2 BLEU points over standard MoE models.  [*Chen et al. 2023*](https://openreview.net/pdf?id=w1hwFUb_81) proposed a random initialized and fixed router network which act as a droput mechanism to alliviate overfiting on a minority number of experts and mitigate experts representation collapse.

<div style="text-align:center">
<img style="right"; src="https://d3i71xaburhd42.cloudfront.net/ca086f4c09cf8de705830ac2b70951737fab93ca/9-Figure7-1.png" width="700"/>

 <em> <strong> Visualization of six different routing algorithms. Each diagram is of a Transformer sparse expert model with four experts (feed-forward networks) routing two tokens. </strong> </em>
</div>

### 2.8. Routing as assignment
Lewis et al. (2021) and Clark et al. (2022) formulate routing as an assignment problem using linear programming and optimal transport for balanced routing. Liu et al. (2023) proposed an optimal transport formulation with support for k-sparsity constraints. All these approaches are also sparse but more expensive than Top-k style routers.
## 3. Existing drawbacks
### 3.1. Differentiability vs computing budget
Both Top-k (or its variants above) and Softmax routers have their pros and cons. Top-k style routers allow for conditional training, i.e., in the forward pass, for each minibatch of size $ \mathcal{B}$, only $k\mathcal{B}$ (instead of $n\mathcal{B}$) expert evaluations (i.e., $E_i(x)$) are required, and hence in backpropagation, only the gradients of the loss with respect to kB elements need to be computed. With a careful implementation, conditional training can give significant computational savings. However, the discontinuous nature and selection bias in Top-k can lead to challenges during optimization. On the other hand, the softmax router is smooth, hence can be easier to optimize. However, the softmax router can be computationally expensive during both training and inference: the router score for each expert is non-zero; hence, all experts $E_i(x)$ are used per-input $x$. 

### 3.2. Load imbalancing
Load balancing problem aries in the scenario that experts receives a high variance number of input tokens which makes the variaty of experts cease to exist. As a result, only a minority number of experts get updated during training leading to degradation of performance during inference time. [Zuo et al. 2021](https://arxiv.org/abs/2110.04260) pointed out that
>  "*it is possible that $W_g$ collapses such that one row dominates, i.e., all the inputs are routed to one expert.*"

Most routing algorithms handle load balancing by adding Gaussion noise or introducing an auxiliary loss during training to encourage equal amounts of tokens getting sent to the different experts ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538), [Dai et al. 2022](https://arxiv.org/abs/2204.08396), [Mustafa et al. 2022](https://arxiv.org/abs/2206.02770) ). 
### 3.3. Representation collapse and homogenous representation of experts
*Representation collapse* was first pointed out by [Chi et al. 2021](https://arxiv.org/pdf/2204.09179.pdf) 
> The token representation $\mathbf{h}$ tends to be updated toward a linear combination of the expert embeddings. As $N$ is much smaller than the hidden size $d$ in practice, the spanning subspace does not fully utilize
the entire available capacity. Thus, the mechanism renders the Transformer hidden vector $\mathbf{h}$ collapsed to an $N$-dimensional subspace, implying a trend toward representation collapse from $\mathbb{R}^d$ to $\mathbb{R}^N$ where $N << d$ in practice. 

and then [Liu et al. 2023](https://arxiv.org/abs/2310.09762) pointed out the *homogenous represtation* issues of experts. 
> The experts in the MoE architecture have not learned diverse knowledge that is unique to specific inputs.

Current solutions to this issue envolve the normalization (X-Moe by Chi et al. 2021) of embedding vector in a lower dimension and a complicating traing scheme (by Liu et al. 2023).

<span style="color:red">
It is important to clarify the difference between the types of "collapse" in MoEs. Imbalanced loading caused representation collapse in the router parameters W_r where a few rows dominate others. The high embedding dimmension of token with low number of experts leads to the representation collapse in token embedding. An lastly, the low diversity of experts (low specialization) or homogenous representationupA leads to experts redundancy. 
</span>

### 3.4. Gradient mismatch

## 4. Highlights
This part of the report introduce some researches that have great impacts or introduce some helpfull idead for the project.
### 4.1. Stable MoE [*Dai et al. 2022*](https://arxiv.org/abs/2204.08396)
The authors point out that existing learning-to-route MoE methods suffer from the routing fluctuation issue, i.e., the target expert of the same input may change along with training, but only one expert will be activated for the input during inference. The routing fluctuation tends to harm sample efficiency because the same input updates different experts but only one is finally used. 
<div style="text-align:center">
<img style="right;" src="https://d3i71xaburhd42.cloudfront.net/c9550f0d1940ee1adf1549c9a0d699ef896dbefd/2-Figure1-1.png" width="800"/>
<em> Illustration of the routing fluctuation problem.  </em>
</div>

**STABLEMOE** has two training stages. In the first training stage, we follow the learning-to-route paradigm and aim to learn a balanced and cohesive routing strategy. As the routing strategy is being learned, we synchronously distill it into a lightweight router decoupled from the backbone model. In the second training stage, we utilize the distilled router to determine the tokento-expert assignment. The distilled router is frozen in this stage to provide a stable routing strategy. During inference, we also use the frozen distilled router for consistent routing.

<div style="text-align:center">
<img style="right;" src="https://d3i71xaburhd42.cloudfront.net/c9550f0d1940ee1adf1549c9a0d699ef896dbefd/3-Figure3-1.png" width="800"/>
<em>  Illustration of two training stages in STABLEMOE. In training stage 1, we learn a routing strategy and distill it into a lightweight router. Then, we freeze the distilled router for stable routing in training stage 2  </em>
</div>

#### 4.1.1. Training Stage 1: Learn Routing Strategy
The assignment score for between token $t$ and expert $i$ is: 
$$
\begin{align}
s_{t,i} &= E_i^T\mathbf{h}_t^{l-1} \\
a_t &= \argmax_i(s_{t,i}) \\ 
\mathbf{h}^l_t &= \sigma(s_{t, a_t}) \text{FFN}_{a_t}(h^{l-1}_t) + h^{l-1}_t
\end{align}
$$
$\sigma$ is the sigmoid gate (Lewis et al., 2021). Considering the sigmoid gate $\sigma(s_{t,at})$, if $FFN_{a_t}$ is beneficial for token $t$, optimizing the training objective (e.g., minimizing the cross-entropy loss for language modeling) will urge the gate to be greater; otherwise, the gate will tend to be smaller. The gate signal urges similar tokens to be assigned to the same expert that is beneficial to them, thus producing cohesive token-to-expert assignments.
##### Balance loss 
$$
\begin{equation}
\mathcal{L}_{bal} = \alpha\displaystyle\sum_{i=1}^N \left( \dfrac{\vert \mathcal{A}_i \vert - \bar{n}}{\bar{n}} \sum_{t \in \mathcal{A}_i} \sigma(s_{t,i}) \right)
\end{equation}
$$

##### Distilled router
Let $X$ be the input sequence and $\hat{E}$ be the distilled expert centroids, we use word embeddings $D(\cdot)$ to extract the routing features and cross-entropy loss as the distillation loss $\mathcal{L}_{dis}$:
$$
\begin{align}
\hat{\mathbf{h}}_t^{l-1} &= D(X_t),\quad \hat{s}_{t,i} = \hat{E}_i^T \hat{\mathbf{h}}_t^{l-1} \\
\mathcal{L}_{dis} &= -\displaystyle \sum_{t=1}^T \log \left(\dfrac{\exp (\hat{s}_{t, a_t})}{\sum_{i=1}^N\exp(\hat{s}_{t,i})}\right)
\end{align}
$$

##### Training objective
$$
\begin{equation}
 \mathcal{L}_{S1} = \mathcal{L}_{task} + \mathcal{L}_{bal} + \mathcal{L}_{dis}.
\end{equation}
$$

#### 4.1.2. Training Stage 2: Learn with Stable Routing Strategy
Keeping other processes the same as in training stage 1, we calculate the output of the MoE layer as follows:
$$
\begin{align}
\hat{\mathbf{h}}_t^{l-1} &= D(X_t),\quad \hat{s}_{t,i} = \hat{E}_i^T \hat{\mathbf{h}}_t^{l-1}, \\
\hat{a}_{t} &= \argmax_i (\hat{s}_{t,i}), \\
\mathbf{h}^l_t &= \sigma(s_{t, \hat{a}_t}) \text{FFN}_{\hat{a}_t}(\mathbf{h}^{l-1}_t) + \mathbf{h}^{l-1}_t.\\
\mathcal{L}_{S2} &= \mathcal{L}_{task}
\end{align}
$$
#### 4.1.3. Inference
During inference, we also use the frozen distilled router for routing. The fixed routing strategy, which is consistent with training stage 2, makes information learned in MoE layers be utilized more thoroughly and thus leads to better performance.
#### 4.1.4. Results
<div style="text-align:center">
<img style="right;" src="https://d3i71xaburhd42.cloudfront.net/c9550f0d1940ee1adf1549c9a0d699ef896dbefd/5-Table2-1.png" width="800"/>
<em> Perplexity results of language modeling on CC100 combined with Roberta corpus. </em>
</div>

<div style="text-align:center">
<img style="right;" src="https://d3i71xaburhd42.cloudfront.net/c9550f0d1940ee1adf1549c9a0d699ef896dbefd/6-Table3-1.png" width="800"/>
<em> Translation task (X -> En) test BLEU on WMT. </em>
</div>


### 4.2. Random SMoE [*Chen et al. 2023*](https://openreview.net/pdf?id=w1hwFUb_81)
#### Motivattions
> - *Unstable training*. Zoph et al. (2022) pointed out that while techniques like gradient clipping can stabilize SMoE training, they often result in lower quality. The router z-loss (Zoph et al., 2022) is a preferred solution for achieving both improved performance and stability. 
> - *Poor specialization*. One of the intriguing goals of SMoE is to divide-and-conquer the learning task by solving each piece of the task with adaptively selected experts. To encourage specialization and decrease redundancy among experts (Chen et al., 2022b), Dai et al. (2022) pre-defined the expert assignment for different input categories, while Hazimeh et al. (2021) advocated multiple, diverse router policies. 
> - *Representation collapse and load imbalance among experts*. As the primary issues of learning-based SMoEs, various approaches have been proposed to mitigate their negative effects. Shazeer et al. (2017) injected Gaussian noises into gating networks to promote the routing balance. Later, Lepikhin et al. (2020); Fedus et al. (2021) applied an auxiliary loss of load balancing regularizers; Lewis et al. (2021) performed the routing by dealing with a linear assignment problem; Clark et al. (2022) utilized reinforcement learners; Zhou et al. (2022) routed top-k inputs per expert instead of selecting top experts per input sample. Beyond learned routing policies, Roller et al. (2021) and Zuo et al. (2022) designed deterministic hashing and stochastic assignments, respectively, which eliminate the necessity for router networks. Moreover, the current learning-based routing mechanisms in SMoEs tend to push hidden representations clustering around expert centroids (Chi et al., 2022), implying a trend toward representation collapse, which in turn leads to redundant experts, inferior expert specialization, thereby substandard performance.
  
#### Contributions:
> -  We propose a new plug-and-play training framework, **SMoE-Dropout**, to enable scaling transformers in the full capacity setting without collapse. SMoE-Dropout facilitates the randomly and sparsely activated structure of network modules, playing an implicit regularization role similar to dropout. Our new framework leads to enhanced generalization and reduced training costs (e.g., up to 37% running time savings) compared to the vanilla training of large dense transformers at equivalent parameter counts.
 >-  Transformers trained by SMoE-Dropout naturally exhibit a "*self-slimmable*" property that displays smooth and consistent performance boosts when increasing activated experts during inference or fine-tuning 
>

<div style="text-align:left">
<img style="right;" src="https://d3i71xaburhd42.cloudfront.net/1462a0e5b7db47301bb0995db56426e1f4a0ac7d/3-Figure2-1.png" width="800"/>
 <figcaption> <strong> Overview of the proposed SMoE-Dropout</strong>. Left describes the standard transformer layer, consisting of multi-head attention and multi-layer perceptron (MLP) components. Middle Left shows the process of modulization. It splits the original MLP evenly and constructs a series of experts which are smaller MLPs with a reduced hidden dimension. Middle Right presents the overall procedure of SMoE-Dropout. The random router selects the top-k experts given a token embedding and then reweights the features from activated experts. In the end, a summation is conducted to aggregate all features. Right displays the gradually increased number of chosen experts, along with the training procedure  </figcaption>
</div>

#### Random policy routing
SMoE-Dropout considers a randomly initialized and fixed router network to guide token assignment. Different from previous works, the proposal’s assignment is: 
- Implicitly optimized during training, since feature embeddings remain updated for the same input sample; 
- Deterministic during inference thanks to the fixed weights in R. Extensive results in Section 4 verify the superiority of our proposal, compared to existing random policies and the dense baseline with the same model parameters.
> In our paper, we do not combine Droput and SMoE. Based on the finding in ([Zoph et al. 2022](https://www.computer.org/csdl/proceedings-article/ipdpsw/2022/974700b044/1Fu9v1EnITm)), directly integrating Dropout and SMoE results in inferior performance. The reason that we name our method as SMoE-Dropout is they share similar design philosophy - “randomly” activating part of the model and disabling the rest part. (Openreview)

> Specifically, let $\theta^r = \theta^r_0$ and $\theta = \theta_0$ denote the model parameters of the router network and other network components (e.g., experts), respectively. They are randomly initialized as $\theta^r_0$ and $\theta_0$. Throughout the training process, we only update $\theta$ while keeping $\theta^r$ unchanged. This is also why we describe the router as "randomly initialized and fixed". However, with the update of $\theta$, the produced intermediate token embeddings are changed in each iteration. Note that the routing assignment is dependent on both $\theta^r$ (router weights) and the intermediate token embeddings (router inputs), which is updated accordingly due to the optimzied embeddings. Therefore, we say our routing policy is “implicitly optimized” rather than fully random. (Openreview)

#### Progressively enlarged number of activated experts during training
Alternating $k$ in inference time cause performance degradation. This drawback substantially restricts the practical use of SMoEs because diverse real-world scenarios require different resource budgets, necessitating flexible and effective network capacity during inference. The authors propose to gradually increassing the number of activated experts in the training time to create a flexible network that can adapt to any number of experts and computing budget. 
|Schedule | SmoE-Dropout ($k=\dfrac{N}{2}$) | SMoE-Dropout ($k=N$) |
|:--------:|:----------------------:|:--------------:|
| Linear  |$1.1776   $| $1.1486$|
| Exponential | $1.2167$ | $1.1877$|
| Cosine | $1.2142$ | $1.1858$|
 >Test performance of different schedules in SMoE-Dropout. We assess the performance via the Bits-Per-Character (BPC) metric, where a smaller BPC value indicates a better performance.

#### Results
<div style="text-align:center">
<img style="right"; src="https://d3i71xaburhd42.cloudfront.net/1462a0e5b7db47301bb0995db56426e1f4a0ac7d/7-Table1-1.png" width="700"/>

 <em>Testing performance of {Transformer-XL, BERT, RoBERTa} network backbones on {enwik8, BookCorpus, BookCorpus} datasets, respectively. All models are compared under the same number of parameter counts. Training time (s) and inference FLOPs (×1010) are reported. For THOR (Zuo et al., 2022), SMoE, and SMoE-Dropout, evaluations are performed with half (k = N/2 ) or all (k = N) experts activated. </em>
</div>

>  - The performance of SMoE-Dropout is stably improved along with more parameters used, and it outperforms the others after 1.0, 10, and 8 (×107) parameter counts for three backbones.
> - In contrast, learnable SMoE’s and THOR’s BPC are quickly saturated and deteriorated when adopting more experts, which implies the existence of expert redundancy (or representation collapse). The potential reasons for their substandard results are:
>    - The overfitting to fixed # experts utilized during training for learnable SMoE.
>    - The consistency regularization between experts’ predictions for THOR.

<div style="text-align:center">
<img style="right"; src="https://d3i71xaburhd42.cloudfront.net/1462a0e5b7db47301bb0995db56426e1f4a0ac7d/8-Table2-1.png" width="700"/>

 <em> Transfer performance {Accuracy (% ↑), Problem Solving Rate (% ↑)} of {Transformer-XL, BERT, RoBERTa} networks on {SST-2, CSQA, ASDiv-A, MAWPS, SVAMP} datasets. All models are compared under the same number of parameter counts. The same densely fine-tuning is adopted for all approaches, while THOR, SMoE, and SMoE-Dropout are tuned with half (k = N/ 2 ) or all (k = N) experts activated. </em>
</div>

## 5. Future work 
- Solve gradient mismatch between gate loss and other component loss. [Dryen and Hoefler 2022.](https://openreview.net/forum?id=AlkMMzUX95)