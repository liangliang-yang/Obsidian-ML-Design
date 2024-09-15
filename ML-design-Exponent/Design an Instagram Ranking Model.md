
[https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model](https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model)

In this mock interview, Vikram (Meta MLE) designs a machine learning (ML) ranking model for Instagram’s feed.


## Step 1: Define the problem[](https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model#step-1-define-the-problem)

### Identify the core ML problem

The core ML task is twofold:

1. Gather a list of recommended posts.
2. Rank them in descending likelihood of engagement for each user.

### Clarify requirements and tradeoffs

To understand the context of the problem, we will make some reasonable assumptions and check in with the interviewer after summarizing our ideas:

1. In an Instagram (IG) feed, a user will see “Suggested Posts” by various content creators.
2. The primary business objective for IG is to enhance user engagement, measured by total session time or the number of sessions, thereby increasing opportunities for ad impressions.
3. Safeguard metrics like “Reported Posts” and “Blocked Content Creators” ensure user satisfaction and trust.
4. We'll consider a few million post-items for a user and narrow them down to a manageable subset for ranking and display, according to their likelihood of engagement.

The ML system operates at an individual level, whereas the business looks at gross metrics. We can align the ML and team objectives by recommending content that has historically seen high engagement.

Non-functional requirements include:

- **Analytics**: The system will track engagement metrics with a minimum granularity of 5 minutes.
- **System integrity**: We’ll include features like robust logging and reproducibility mechanisms for efficient debugging and diagnostics.
- **Adaptability**: We'll integrate online learning capabilities, allowing the model to update based on real-time user interactions. Additionally, periodic batch learning with recent data will ensure the model remains current.

For ML operations, there will be several **availability** and **scalability** requirements. We can assume a query per second (QPS) of 10,000 users/second requesting the IG feed. This approximates a few hundred million users (assuming a 1% concurrency).

We’ll disregard monetization through ad selection, relevance assessment, and strategic placement since it is peripheral to this use case and we are limited on time.

We’ll pause here for any feedback from the interviewer.


## Step 2: Design the data processing pipeline[](https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model#step-2-design-the-data-processing-pipeline)

Our data processing pipeline will include a comprehensive feature set for users and posts. This approach ensures we maintain a high-quality dataset for accurate feed ranking.

The user features include:

- **Basic user data**: Basic information about the user, such as demographics and account settings. Sensitive personally identifiable information (PII) will be excluded, depending on any regulatory compliance requirements.
- **Activity metrics**: Interactions on the platform, including the number of posts liked, commented on, or shared over specific time frames
- **Social connectivity**: Information on the user’s network (e.g. number of followers, following, and interactions with other users)
- **Content engagement**: Engagement with various content types (e.g. posts, reels, and stories) captured through metrics like view time and interaction types

The post features include:

- **Content characteristics**: If the post content contains text, images, audio, or video, we could consider embeddings derived from dedicated pre-trained text/image/video models as features.
- **Engagement metrics**: Aggregate data on how all users have interacted with the post, (e.g. likes, comments, shares, and view time).
- **Creator metrics**: Information about the content creator, including their overall engagement rates, follower count, and posting frequency.

We’ll compute and store all features in a centralized feature repository, tagged with timestamps, to ensure we can access the most current and historical data. This repository will support efficient lookups and system reproducibility.

For proper data hygiene, we should compute and store user and post features at the time of the impression on the user’s device, not at the time of the user’s actions. Otherwise, we risk information leakage into the feature attributable to user actions.

We’ll create a pipeline with three stages: candidate generation, ranking, and post-processing.

### **1. Candidate generation (also known as retrieval)**

From ~1 million available posts since the user’s last visit, we will sample to gather 100,000 posts based on the user’s preferences (e.g. language, location). From these 100,000 samples, we’ll generate a shorter list of 1,000 candidate post-items.

This step must be computationally swift and not involve deep inferential networks. A typical way to engineer for rapid candidate generation is to learn user and post embedding during training, and then subsequently use approximate nearest neighbor (ANN) techniques to select candidate posts for the user.

### **2. Ranking**

We’ll rank these 1,000 post-items by their likelihood of user engagement. We rank 1,000 posts for each user, so it is not computationally intense. We could use a larger, deeper, and computationally heavier ML model than the previous stage.

### **3. Post-processing**

Finally, we’ll post-process the “ranked list” to remove items that are out of user preferences or inappropriate. We’ll also adjust for fairness, diversity, and freshness of content. The post-processing stage will be based on static rules, such as preference matching, post-origin dates/times, etc.

We now have a list of 1,000 ranked suggested content that will be sent to the user. Since the IG feed supports infinite scrolling, there will be paging, pre-fetching, and caching, which will be handled through a standard system design or user interface app design.

From the MLOps perspective: 10,000 * 100,000 candidate generation queries = 100M candidate generation QPS 1,000 * 1,000 = 1M ranking QPS


## Step 3: Propose a model architecture[](https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model#step-3-propose-a-model-architecture)


To learn an embedding for the candidate generation phase, there are two chief possible approaches:

1. Collaborative filtering with matrix factorization, where we decompose a sparse user-post engagement matrix enhanced with side features of users and post-items
2. Collaborative filtering with a two-tower network, where we create one for users and the other for post-items

### Matrix factorization

Independent of the chosen architecture, the outcome is to learn a lower dimensional embedding (dimension, d) for the user and the post item. The user and the post occupy the same ‘d’ dimensional space.

These embeddings are stored in a repository and fetched subsequently for fast computations of dot products and nearest neighbors. Updates to these embeddings can be made at a pre-decided cadence (e.g. every 4 hours or nightly). Note that the cadence could decay gradually—for instance, frequently in the beginning and less so subsequently—since embeddings for users and post-items change quickly early on and less so over time.

The data diagram below shows the Data Matrix A for collaborative filtering with side features:

![[Pasted image 20240915114243.png]]
The ‘1’ values in the matrix correspond to a user interaction with the post item, e.g. view/watch, like, comment.

This sparse matrix is factorized into:

- U matrix for users/users
- V matrix for post-items

A = m x n, where m = # of users plus # of post-item features. U = m x d V is of dimension n x d.

Therefore, A = U * V.T (V transpose) + L2 loss + gravity loss. The last two terms are weighted regularizers that coax lower parameter weights.

The technique for learning here is known as weighted alternating least squares (ALS) using stochastic gradient descent. The loss function here is a squared loss akin to the Frobenius norm of the difference between A and U * VT.

![[Pasted image 20240915114758.png]]We introduced two methods to compute user and post-item embeddings for collaborative filtering. The main tradeoffs between the two methods include:

- Extensive features for both users and posts are possible with the two-tower network. A similar effort, though algebraically possible, with matrix factorization might require numerical considerations of the size, scale, and range of the numbers in the A-matrix.
- The cold start problem is handled easily in a two-tower approach since an initial embedding for a new user is a forward pass through the user and post-item towers. The matrix factorization approach suffers more severely from the cold start problem.
- Embedding learning through matrix factorization is a rapid and straightforward computation. Two-tower networks are slower because of the depth of the network, non-linearity, other architectures, CNNs, skip connections, etc.

==We can decide to build a two-tower network over matrix factorization, given the benefits of extensive features and a less severe cold start problem.==

###  Two-tower network 
#two-tower-model 

For two-tower networks, we’ll create a deep network with two parallel stacks of subnetworks.

Each of these networks terminates in a ‘d’ dimensional vector. The ‘d’ dimensional vectors are the embeddings of the user or post-item (depending on the stack).

The last layer performs a dot product of the user and post-item embeddings and runs the output through a sigmoid function (which yields values from (0, 1)). The outcome can be interpreted as the likelihood of engagement between the user and the post-item.

To train this network, we’ll need to gather an equal number of positive and negative examples of user-post-item engagements. The labels are ‘1’ for user-post-items that resulted in an engagement and 0 for user-post-items that didn’t.

![[Pasted image 20240915115521.png]]
## Step 4: Train and evaluate the model[](https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model#step-4-train-and-evaluate-the-model)

### Train the model

We’ll choose the two-tower model to train. While preparing the training and validation data, we want to ensure equal representation of positive and negative engagement examples. Typically, the engagement ratio in an Instagram feed is low, so we will allow for negative downsampling and/or positive upsampling.

For the initial training, we gather 500 million positive examples and 500 million negative examples. We shall use 800 million of those examples for training and 200 million for validation.

### Evaluate the model

The task of learning an embedding for each pair of user and post-item is a classification problem. Therefore, the loss function will be binary cross entropy and will include a regularizer term for the network weights. We suggest utilizing Adam Optimizer with a learning rate of 0.1 and a decay of 0.9 over epochs. For rapid progress on training, it’ll be helpful to use smaller batches of 128 training examples at a time.

As the training epochs proceed, we’ll record the training and validation losses. Typically, the training loss will be lower than the validation loss. We can save the model as the training proceeds and stop if we observe the validation losses increasing between epochs. Such a validation loss increase is indicative of over-fitting. Otherwise, we’ll stop at the end of 100 epochs.

Since this is a classification problem, we’ll use the AUC ROC, which indicates the tradeoff between true and false positive rates. We’ll use an empirical AUC threshold of 80%.

Once this model is trained and validated, we can use this model to compute (offline) the embeddings for a user and/or a post by running the forward portion of the appropriate tower. These embeddings will be stored in a database, indexed by user ID and post ID.

## Step 5: Deploy the model[](https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model#step-5-deploy-the-model)

We’ll start by running A/B tests on a smaller portion of the online traffic. If the A/B tests show improvements in topline metrics (user engagement or user DAU) and stability in safeguard metrics, our model is ready to be deployed to all production users. To monitor the deployment, we’ll log the following:

- Features
- Predictions, through probability numbers
- User actions, thresholded to 2 weeks since the item showed up in the feed
- Predictions and actual actions performed by users during specified windows of time to capture early alarms of poor performance

To address the continuous change in the data and environment, we should design an MLOps that allows online learning, in addition to the offline batch learning that occurs at a weekly cadence.

A deployment architecture for serving and monitoring, as well as improvements through recurring offline/batch learning and online learning, is a decision service, as shown in the image below.

![[Pasted image 20240915115651.png]]Recommendation and ranking systems are categorized as contextual decision-makers. These systems have a higher incidence of failure modes and technical debt than traditional supervised prediction problems.

For this problem setting, these incidents occur because:

- The application makes repeated calls to the IG feed’s ranking when faced with a particular context (e.g. a new user or refresh by the same user).
- The reward to the ranking system is delayed by varying amounts, based on the user’s screen/click habits.

Therefore, we should diligently log the following:

- Features (x)
- Suggested actions (a)
- Probabilities (p)
- Rewards (r)

### Senior candidates
The abstractions in the image above (i.e. Explore, Log, Learn, and Deploy) cater to model serving and model re-learning by capturing the following:

- **Explore** captures feature generation (x) and model predictions (a, p).
- **Log** captures gathering rewards (user actions, r), and correlating them with (x, a, p).
- **Learn** corresponds to online or batch learning, where the aggregated (x, a, r, p) are used to learn improved models.
- **Deploy** captures controlled deployment consequent to offline and online evaluations, (e.g. A/B tests).

Monitoring is performed by logging analytical metrics in **Explore** and **Log** abstractions. Alarms and notifications are configured based on accrued metrics within varying time windows (e.g. hours, days, weeks) and granularity (e.g. 5 minutes, 15 minutes). Metrics accrual, analytics, reporting, and operational alarms from data streams is an entire area of study where advanced data structures such as count-min sketches/heaps are used with data partitioners, message queues (e.g. Apache Kafka), and stream processors (e.g. Apache Flink).

Scaling is ubiquitous in these abstractions and is an infrastructure-wide choice. Some ML scaling techniques include:

- Distributed computation
- Parallelized computation
- Model distillation
- Quantization

These techniques can be applied to general ML models, but they are typically discussed with neural networks.

Parallelized computation ideas include model parallelization and data parallelization.

Briefly, **model parallelization** inspects and separates independent parts of the two-tower network into separate sets of servers (e.g. user tower from post-item tower, since they can run independently). Similarly, we can separate parts of each tower that can run independently.

For **data parallelization**, we can imagine multiple copies of the two-tower network. Each instance of a two-tower network processes a separate portion of the training data. Each instance of the two-tower network learns a different set of gradients based on their training samples and prediction errors. These gradients are combined (e.g. averaged) and applied to each server’s parameter values.


## Step 6: Wrap up[](https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model#step-6-wrap-up)

To sum up:

- We constructed an IG feed recommendations pipeline, which includes learning embeddings for users and posts through a two-tower network.
- At serving time, we used the ANN algorithm to select 1,000 posts for a user.
- These posts are ranked, sorted, and then post-processed.
- We created a serving, monitoring, and continuous learning architecture that keeps the models up-to-date. With this solution, we can support monitoring, analytics, and even debugging and traceability of ML predictions.

## Other considerations[](https://www.tryexponent.com/courses/ml-engineer/ml-system-design/design-instagram-feed-ranking-model#other-considerations)

With more time, it’d be beneficial to discuss techniques that support ANN, such as locality-sensitive hashing and KD-trees. These techniques provide two key benefits:

- Allows for data partitioning of a high-dimensional space, such as the space of user and post-item embeddings.
- Supports rapid lookup of neighbors without precisely gathering the nearest ones, which suffices for an Instagram feed.

We could also address ad placement and related frameworks in Instagram feeds. Ads are the key monetization path for social media apps, so they are essential to consider in a wider organization-level design.