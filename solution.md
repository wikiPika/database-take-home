## Solution

### Approach & Analysis

Random walks are rather silly. It's probably better to control the number of loops that happen, if only to morph things into more of a linear search (more predictable),

### Optimization Strategy

##### First implementation: Naive Figure-8

Really dumb, this is just sticking the top node in the middle and wrapping two circles around it.

In hindsight this was not actually "very dumb". It (probabilistically) gurantees a hit while (probabilistically) minimizing probe length to (NODES / 2). This leaves us a score of 203.14, which is pretty good for about two minutes of dumb thinking (thank you GPT for the implementation).

In our subsequent attempts, we'd like

### Implementation Details

##### Naive Figure-8
Although we use the top-1 node as the hub center, we still would like somewhat consistent hits for the rest of the small bunch in the exponential distribution, so we space those evenly around the two circles. Not too carefully though, it's more of a hit of salt or something.

### Results

| Method                             | Success % | Median Path | Score  |
| ---------------------------------- | --------- | ----------- | ------ |
| No Optimization                    | 79.50%    | 476.5       | -      |
| Whatever Terrible Shit You Gave Me | 70.00%    | 463.0       | 119.53 |
| Naive Figure-8                     | 100.00%   | 264.0       | 203.14 |

### Trade-offs & Limitations

[Discuss any trade-offs or limitations of your approach]

### Iteration Journey

[Briefly describe your iteration process - what approaches you tried, what you learned, and how your solution evolved]

---

- Be concise but thorough - aim for 500-1000 words total
- Include specific data and metrics where relevant
- Explain your reasoning, not just what you did
