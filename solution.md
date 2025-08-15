## Solution

### Approach & Analysis

Random walks are rather silly. It's probably better to control the number of loops that happen, if only to morph things into more of a linear search (more predictable),

### Optimization Strategy

##### 1: Naive Figure-8

Really dumb, this is just sticking the top node in the middle and wrapping two circles around it.

In hindsight this was not actually "very dumb". It (probabilistically) gurantees a hit while (probabilistically) minimizing probe length to (NODES / 2). This leaves us a score of 203.14, which is pretty good for about two minutes of dumb thinking (thank you GPT for the implementation).

##### 2: Three Fifths of the Olympic Rings

Since we have a whole three out-edges to work with, we should try making it more of a figure-three-leaf-clover type of thing. You may ask why I didn't do this the first time around... great question, I really can't answer that.

### Implementation Details

##### 1: Naive Figure-8

Although we use the top-1 node as the hub center, we still would like somewhat consistent hits for the rest of the small bunch in the exponential distribution, so we space those evenly around the two circles. Not too carefully though, it's more of a hit of salt or something.

#### 2: Three Fifths of the Olympic Rings

There's no new pizzazz here.

### Results

| Method                               | Success % | Median Path | Score  |
| ------------------------------------ | --------- | ----------- | ------ |
| No Optimization                      | 79.50%    | 476.5       | -      |
| Whatever Terrible Shit You Gave Me   | 70.00%    | 463.0       | 119.53 |
| 1: Naive Figure-8                    | 100.00%   | 264.0       | 203.14 |
| 2: Three Fifths of the Olympic Rings | 100.00%   | 256.25      | 205.07 |

### Trade-offs & Limitations

##### Clove Count

Two cloves vs. three cloves didn't really mean much, probably because the center node isn't really hit thaaaaat frequently.

### Iteration Journey

[Briefly describe your iteration process - what approaches you tried, what you learned, and how your solution evolved]

---

- Be concise but thorough - aim for 500-1000 words total
- Include specific data and metrics where relevant
- Explain your reasoning, not just what you did
