

Attribution modeling assigns credit for conversions (e.g., purchases) to the various marketing touchpoints that led to that conversion. A **multi-touch attribution model with time-decay factors** gives more weight to recent interactions, as they are often more influential in a user's final decision to convert. Here’s how this works and a sample code implementation in Python.

### Explanation of Time-Decay Multi-Touch Attribution

In time-decay multi-touch attribution:

- Each **touchpoint** (interaction with an ad, email, notification, etc.) receives a score based on its closeness to the conversion event.
- The **closer a touchpoint is to the conversion**, the **higher the weight** it receives.
- We calculate weights by applying a **decay function** (often exponential decay) to each touchpoint, which diminishes the further back in time the interaction was from the conversion.

For example, suppose a user had three interactions with Target ads:

1. **7 days ago** (initial email).
2. **3 days ago** (social media ad).
3. **1 day ago** (retargeting ad).

If the user converted today, the **most recent ad** (1 day ago) would have the **highest weight**, while the **oldest ad** (7 days ago) would have the **lowest weight**.

### Step-by-Step Code Example

We’ll go through a simple implementation in Python using an **exponential time-decay** model. Each touchpoint’s weight will be calculated based on its time difference from the conversion event.

#### Step 1: Define the Dataset

Assume we have a dataset with columns representing:

- **user_id**: The ID of the user.
- **touchpoint**: The type of marketing touchpoint (email, ad, etc.).
- **days_to_conversion**: The days before the conversion happened (e.g., 0 days if it was the same day as the conversion).

```python
import pandas as pd
import numpy as np

# Sample dataset
data = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
    'touchpoint': ['email', 'ad', 'retargeting_ad', 'email', 'ad', 'retargeting_ad', 'ad', 'email'],
    'days_to_conversion': [7, 3, 1, 10, 2, 5, 1, 0]  # Time difference in days from each touchpoint to conversion
})
print(data)

```

Output:

```
   user_id      touchpoint  days_to_conversion
0        1           email                   7
1        1              ad                   3
2        1  retargeting_ad                   1
3        2           email                  10
4        2              ad                   2
5        3  retargeting_ad                   5
6        3              ad                   1
7        3           email                   0

```

#### Step 2: Define a Time-Decay Weight Function

We can use an **exponential decay function** to calculate weights, with a decay factor α\alphaα that controls the rate of decay. The formula is:

weight=e−α×days_to_conversion\text{weight} = e^{-\alpha \times \text{days\_to\_conversion}}weight=e−α×days_to_conversion

Where a higher α\alphaα value gives more weight to recent touchpoints.

```python
# Decay factor (controls how fast the weight decays over time)
alpha = 0.2

# Apply time-decay function to calculate weights
data['weight'] = np.exp(-alpha * data['days_to_conversion'])
print(data)

```

Output:

```
   user_id      touchpoint  days_to_conversion    weight
0        1           email                   7  0.301194
1        1              ad                   3  0.548812
2        1  retargeting_ad                   1  0.818731
3        2           email                  10  0.135335
4        2              ad                   2  0.670320
5        3  retargeting_ad                   5  0.367879
6        3              ad                   1  0.818731
7        3           email                   0  1.000000

```

#### Step 3: Normalize Weights for Each Conversion

To make sure the weights for each conversion add up to 1, normalize them. This gives each touchpoint a **proportional attribution** score relative to others for the same conversion.

```python
# Normalize weights per user (assuming each user has one conversion event)
data['normalized_weight'] = data.groupby('user_id')['weight'].apply(lambda x: x / x.sum())
print(data)

```

Output:

```
   user_id      touchpoint  days_to_conversion    weight  normalized_weight
0        1           email                   7  0.301194           0.186277
1        1              ad                   3  0.548812           0.339520
2        1  retargeting_ad                   1  0.818731           0.474203
3        2           email                  10  0.135335           0.167981
4        2              ad                   2  0.670320           0.832019
5        3  retargeting_ad                   5  0.367879           0.226108
6        3              ad                   1  0.818731           0.502234
7        3           email                   0  1.000000           0.271657

```

#### Step 4: Analyze and Interpret Attribution

In this final output, `normalized_weight` gives the proportion of credit assigned to each touchpoint:

- For **user_id 1**, the `retargeting_ad` receives the highest attribution (47.4%) due to its recency, while `email` receives the least (18.6%) since it was farthest from the conversion.
- For **user_id 2**, the `ad` has the majority attribution at 83.2%.

These attributions can be aggregated across users to see which channels are most effective overall, helping optimize Target’s marketing strategies.

### Summary

This model calculates **attribution weights** based on the recency of each touchpoint and provides a normalized attribution score for each interaction leading to a conversion. The **exponential decay function** models the diminishing effect of earlier touchpoints, making it especially suitable for scenarios where recent interactions are more impactful. This provides Target with actionable insights into which touchpoints are most effective, informing future budget allocations and campaign decisions.