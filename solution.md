## Solution

### Approach & Analysis

Random walks are rather silly. It's probably better to control the number of loops that happen, if only to morph things into more of a linear search (more predictable),

### Optimization Strategy

##### 1: Naive Figure-8

Really dumb, this is just sticking the top node in the middle and wrapping two circles around it.

In hindsight this was not actually "very dumb". It (probabilistically) gurantees a hit while (probabilistically) minimizing probe length to (NODES / 2). This leaves us a score of 203.14, which is pretty good for about two minutes of dumb thinking (thank you GPT for the implementation).

##### 2: Three Fifths of the Olympic Rings

Since we have a whole three out-edges to work with, we should try making it more of a figure-three-leaf-clover type of thing. You may ask why I didn't do this the first time around... great question, I really can't answer that.

##### 3: Ice In My (Xylem) Veins

This one is just an experiment, but what if we add a loopback in the middle of each clove? Turns out it shaves a few percent.

##### 4: Bootleg Chord (single express)

So I learned about a similar sort of thing in Distributed Systems; it'll probably be also good here. We keep a sort of skip-table kind of thing for each node in a circular network which lets you get from anywhere to anywhere else pretty darn quickly while maintaining linear probing as necessary.

As a proof of concept, we'll use $s = \sqrt{500} \sim 22$ as a single skip link. The square root here is an optimal point of the expected hops $\mathbb{E}(d) = \frac{N}{2s} + \frac{s}{2}$.

This one ended up being not much better than clover-ing, probably because the path length ends up being really janky with all the hopping around.

##### 5: de Brujin Rolls in His Grave

To dishonor another researcher, we borrow logarithmic mixing fro de Brujin. Since our target node distribution is exponential, we can be a little handwavey about the last few hundred nodes - we connect the nodes in id sequence and add a skip "express train" between (1, 2, 4, 8, 16, ..., 256, 1).

##### 5b. de Brujin Settles Back Down

There's some tweaking that should be done here, since the exponential distribution parameter is quite large. We add a reverse funnel line - this ends up potentially trapping some queries that target higher node IDs, but the path length becomes much, much shorter. We ought to use this funnel a bit more, to be honest.

##### 6. Your Success Is (Not) Important To Us

It seems that 100% hit rate is a fair price to not-pay if we want insanely short path lengths. This really depends on how sketchy you want your database to be, I suppose.

Since the reward function is biased towards really short path length and the main group of query hits are [0..39] ish, we can justify just routing everyone and their mom towards those queries and neglecting everything else. After all, nobody really goes to 250th Street in New York City anyway (untrue).

We maintain everything in a ring so folks have a chance to hit everything, but otherwise it absolutely slashes path length.

##### 6b. Sweeping the (Parameter) Floor

When in doubt, we run a little parameter sweep.

### Implementation Details

I'll only mention something if it's interesting enoughg to talk about beyond whatever I yap about up there.

##### 1: Naive Figure-8

Although we use the top-1 node as the hub center, we still would like somewhat consistent hits for the rest of the small bunch in the exponential distribution, so we space those evenly around the two circles. Not too carefully though, it's more of a hit of salt or something.

### Results

| Method                                   | Success % | Median Path | Score  |
| ---------------------------------------- | --------- | ----------- | ------ |
| No Optimization                          | 79.50%    | 476.5       | -      |
| Whatever Terrible Shit You Gave Me       | 70.00%    | 463.0       | 119.53 |
| 1: Naive Figure-8                        | 100.00%   | 264.0       | 226.94 |
| 2: Three Fifths of the Olympic Rings     | 100.00%   | 256.25      | 229.09 |
| 3: Ice In My (Xylem) Veins               | 100.00%   | 220.0       | 240.38 |
| 4: Bootleg Chord                         | 100.00%   | 327.25      | 211.98 |
| 5: de Brujin Rolls in His Grave          | 100.00%   | 236.75      | 234.89 |
| 5b: de Brujin Settles Back Down          | 98.0      | 73.5        | 325.50 |
| 6: Your Success Is (Not) Important To Us | 100.0%    | 21.75       | 446.75 |

### Trade-offs & Limitations

Two cloves vs. three cloves didn't really mean much, probably because the center node isn't really hit thaaaaat frequently.

In general I always want near\* 100% hit rate since we have 100,000 tries (10 walks x 10,000 steps); we'd have to seriously mess up for a failure basically.

For the sake of being a database this is probably the most important, but for the sake of your metric I'll also give something that sacks success rate for super low path length. Terrible UX, but at least you'll be disappointed faster (lmao).

### Iteration Journey

I'll use this as more of a general thoughts thing.

The three outdegree thing really sucks! I was going to model this after a slime mold / Tokyo metro map type of thing but Tokyo (and the slime mold I guess) has huge transfer stations (ex. Shinjuku station).

It's a shame I haven't taken a network theory class yet, but it is fun! I _knew_ that freshman seminar I took on swarm organization would come in handy at some point.

---

- Be concise but thorough - aim for 500-1000 words total
- Include specific data and metrics where relevant
- Explain your reasoning, not just what you did
