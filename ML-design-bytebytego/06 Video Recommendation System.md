
## Clarifying Requirements

![[Pasted image 20240915160528.png]]


Here is a typical interaction between a candidate and an interviewer.

Candidate: Can I assume the business objective of building a video recommendation system is to increase user engagement?

Interview: That’s correct.

Candidate: Does the system recommend similar videos to a video a user is watching right now? Or does it show a personalized list of videos on the user’s homepage?

Interviewer: This is a homepage video recommendation system, which recommends personalized videos to users when they load the homepage.

Candidate: Since YouTube is a global service, can I assume users are located worldwide and videos are in different languages?

Interviewer: That’s a fair assumption.

Candidate: Can I assume we can construct the dataset based on user interactions with video content?

Interviewer: Yes, that sounds good.

Candidate: Can a user group videos together by creating playlists? Playlists can be informative for the ML model during the learning phase.

Interviewer: For the sake of simplicity, let’s assume the playlist feature does not exist.

Candidate: How many videos are available on the platform?

Interviewer: We have about 10 billion videos.

Candidate: How fast should the system recommend videos to a user? Can I assume the recommendation should not take more than 200 milliseconds?

Interviewer: That sounds good.

Let’s summarize the problem statement. We are asked to design a homepage video recommendation system. The business objective is to increase user engagement. Each time a user loads the homepage, the system recommends the most engaging videos. Users are located worldwide, and videos can be in different languages. There are approximately 10 billion videos on the platform, and recommendations should be served quickly.

From <[https://bytebytego.com/courses/machine-learning-system-design-interview/video-recommendation-system](https://bytebytego.com/courses/machine-learning-system-design-interview/video-recommendation-system)>



## Frame the Problem as an ML Task

### Defining the ML objective

The business objective of the system is to increase user engagement. There are several options available for translating business objectives into well-defined ML objectives. We will examine some of them and discuss their trade-offs.

- Maximize the number of user clicks. A video recommendation system can be designed to maximize user clicks. However, this objective has one major drawback. The model may recommend videos that are so-called "clickbait", meaning the title and thumbnail image look compelling, but the video's content may be boring, irrelevant, or even misleading. Clickbait videos reduce user satisfaction and engagement over time.
- Maximize the number of completed videos. The system could also recommend videos users will likely watch to completion. A major problem with this objective is that the model may recommend shorter videos that are quicker to watch.
- Maximize total watch time. This objective produces recommendations that users spend more time watching.
- Maximize the number of relevant videos. This objective produces recommendations that are relevant to users. Engineers or product managers can define relevance based on some rules. Such rules can be based on implicit and explicit user reactions. For example, one definition could state a video is relevant if a user explicitly presses the "like" button or watches at least half of it. Once we define relevance, we can construct a dataset and train a model to predict the relevance score between a user and a video.

In this system, we choose the final objective - Maximize the number of relevant videos as the ML objective because we have more control over what signals to use. In addition, it does not have the shortcomings of the other options described earlier.

![[Pasted image 20240915160715.png]]

## Choosing the right ML category

In this section, we examine three common types of personalized recommendation systems.

- Content-based filtering
- Collaborative filtering
- Hybrid filtering

![[Pasted image 20240915160933.png]]

### Content-based filtering
![[Pasted image 20240915160953.png]]![[Pasted image 20240915161000.png]]
### Collaborative filtering (CF)

![[Pasted image 20240915161102.png]]
![[Pasted image 20240915161113.png]]

### Hybrid filtering
![[Pasted image 20240915161201.png]]
## Data Preparation
![[Pasted image 20240915161216.png]]
![[Pasted image 20240915161223.png]]![[Pasted image 20240915161227.png]]
## Feature engineering
The ML system is required to predict videos that are relevant to users. Let's engineer features to help the system make informed predictions.

### Video features - embedding
![[Pasted image 20240915161312.png]]![[Pasted image 20240915161318.png]]

### User features
![[Pasted image 20240915161336.png]]![[Pasted image 20240915161346.png]]
### User historical interactions
![[Pasted image 20240915161401.png]]![[Pasted image 20240915161405.png]]![[Pasted image 20240915161408.png]]
## Model Development
![[Pasted image 20240915161422.png]]

### Matrix Factorization
![[Pasted image 20240915161439.png]]![[Pasted image 20240915161444.png]]
![[Pasted image 20240915161525.png]]![[Pasted image 20240915161530.png]]
![[Pasted image 20240915161541.png]]
![[Pasted image 20240915161550.png]]
![[Pasted image 20240915161558.png]]![[Pasted image 20240915161615.png]]
![[Pasted image 20240915161621.png]]

![[Pasted image 20240915161653.png]]

### Two-Tower Neural Network 
#two-tower-model 

![[Pasted image 20240915161735.png]]

![[Pasted image 20240915161744.png]]
![[Pasted image 20240915161759.png]]Figure 6.21: Two-tower neural network training workflow

![[Pasted image 20240915161900.png]]
![[Pasted image 20240915161915.png]]

## Evaluation

### Offline metrics
![[Pasted image 20240915162229.png]]

### Online metrics
![[Pasted image 20240915162242.png]]

## Serving
![[Pasted image 20240915162311.png]]

![[Pasted image 20240915162317.png]]

### Candidate generation
![[Pasted image 20240915162346.png]]![[Pasted image 20240915162358.png]]![[Pasted image 20240915162409.png]]
### Scoring
![[Pasted image 20240915162423.png]]
![[Pasted image 20240915162428.png]]

### Re-ranking
![[Pasted image 20240915162447.png]]

### Challenges of video recommendation systems
![[Pasted image 20240915162505.png]]
![[Pasted image 20240915162510.png]]

## Other Talking Points

If there is time left at the end of the interview, here are some additional talking points:

- The exploration-exploitation trade-off in recommendation systems [9].
- Different types of biases may be present in recommendation systems [10].
- Important considerations related to ethics when building recommendation systems [11].
- Consider the effect of seasonality - changes in users' behaviors during different seasons - in a recommendation system [12].
- Optimize the system for multiple objectives, instead of a single objective [13].
- How to benefit from negative feedback such as dislikes [14].
- Leverage the sequence of videos in a user's search history or watch history [2].

# References

1. YouTube recommendation system.  [https://blog.youtube/inside-youtube/on-youtubes-recommendation-system](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system).
2. DNN for YouTube recommendation.  [https://static.googleusercontent.com/media/r](https://static.googleusercontent.com/media/r) esearch.google.com/en//pubs/archive/45530.pdf.
3. CBOW paper.  [https://arxiv.org/pdf/1301.3781.pdf](https://arxiv.org/pdf/1301.3781.pdf).
4. BERT paper.  [https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf).
5. Matrix factorization.  [https://developers.google.com/machine-learning/recommendation/collaborative/matrix](https://developers.google.com/machine-learning/recommendation/collaborative/matrix).
6. Stochastic gradient descent.  [https://en.wikipedia.org/wiki/Stochastic_gradient_descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).
7. WALS optimization.  [https://fairyonice.github.io/Learn-about-collaborative-filtering-and-weighted-alternating-least-square-with-tensorflow.html](https://fairyonice.github.io/Learn-about-collaborative-filtering-and-weighted-alternating-least-square-with-tensorflow.html).
8. Instagram multi-stage recommendation system.  [https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/](https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/).
9. Exploration and exploitation trade-offs.  [https://en.wikipedia.org/wiki/Multi-armed_bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit).
10. Bias in AI and recommendation systems.  [https://www.searchenginejournal.com/biases-search-recommender-systems/339319/#close](https://www.searchenginejournal.com/biases-search-recommender-systems/339319/#close).
11. Ethical concerns in recommendation systems.  [https://link.springer.com/article/10.1007/s00146-020-00950-y](https://link.springer.com/article/10.1007/s00146-020-00950-y).
12. Seasonality in recommendation systems.  [https://www.computer.org/csdl/proceedings-article/big-data/2019/09005954/1hJsfgT0qL6](https://www.computer.org/csdl/proceedings-article/big-data/2019/09005954/1hJsfgT0qL6).
13. A multitask ranking system.  [https://daiwk.github.io/assets/youtube-multitask.pdf](https://daiwk.github.io/assets/youtube-multitask.pdf).
14. Benefit from a negative feedback.  [https://arxiv.org/abs/1607.04228?context=cs](https://arxiv.org/abs/1607.04228?context=cs).

From <[https://bytebytego.com/courses/machine-learning-system-design-interview/video-recommendation-system](https://bytebytego.com/courses/machine-learning-system-design-interview/video-recommendation-system)>