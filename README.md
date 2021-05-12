# Multi-touch Attribution Models

## METHODOLOGY 
### 1. Heuristic models
### 1.1 Single-touch attribution 
 - **First Touch Attribution (FTA)** appoints all the credits of a conversion to the first channel of a customer journey.
 - **Last Touch Attribution (LTA)** is similar to FTA but appoints the credits to the last channel.
![Single-touch](https://user-images.githubusercontent.com/66676705/117957951-d476a900-b31a-11eb-89fe-f75db352a628.PNG)

### 1.2 Multiple-touch attribution
 - **Linear Touch Attribution (LINEAR)** gives equal weight(s) to all channel(s) in a customer journey that results in conversion.
![Multiple-touch](https://user-images.githubusercontent.com/66676705/117957947-d3de1280-b31a-11eb-8a51-263ae386fe97.PNG)

### 2. Shapley Value method 
Introduced by Shapley (1953), Shapley Value is a credit assignment method in Cooperative Game Theory which allocates payoffs among players based on their contribution to a game. Within a game, the players can cooperate in various combinations to gain rewards. Therefore, the credit of one player is computed based on the expected value of their marginal contribution to the total result of every possible cooperation with other members. In the case of multi-channel attribution, channels are regarded as players in a cooperative game since they work together to attract users and in uence their behaviours. This section will first explain the original Shapley Value approach and later show how it can be applied in the case of channel attribution.

### 2.1 The original model 
Assume that there are k players in a game and let S be a subset of players or a coalition.
The dummy variable x<sub>i</sub> takes value 1 if player i is a member of S and 0 otherwise. The
members of S cooperate and create a total amount of payoffs v(S), which is called the
utility function of S. The value of v(S) depends only on the presence of the players in S
and not on the order of the players entering the coalition. The grand coalition K contains
all k players and v(K) is the total expected value that they all can create together. Given
a cooperative game G = (v, K), the formula of the Shapley Value of player i is as below:
![shapley-orginal](https://user-images.githubusercontent.com/66676705/117959638-84004b00-b31c-11eb-86ca-8f8d5db7955a.PNG) 

where |K| and |S| are the cardinality of the grand coalition K and coalition S. Thus, |K|! and |S|! are the number of all possible permutations of members in K and S respectively, and (|K| - |S| - 1)! is the number of combinations of all players not in S and i. v(S \cup {(x_<sub>i</sub> = 1}) - v(S), named as M(i, S), is the marginal contribution of player i to coalition S. The formula 3.1 shows that Shapley Value \phi<sub>i</sub>(G) is calculated as the weighted average of M(i; S) over all possible coalitions S for each player i.

There are some properties that the Shapley Value method needs to satisfy:
- Efficiency: The Shapley values of all players sum up to the value of the grand coalition so that the total value of the game is distributed among the players.
- Symmetry: This guarantees that players with the same contribution to every coalition should receive equal credits. 
- Dummy Player: A player that does not contribute to any coalition should receive no payoff.

These properties ensure that all players receive their "fair" share of credits for their contribution to the game, and thus it is relatively easy to interpret the results of the model. However, the Shapley Value method has some disadvantages. Firstly, it ignores repeating players and only counts them once, which means that the number of occurrence of the players is irrelevant to the result. Secondly, it does not include the order information, implying that different orders of a coalition yield the same payo. For instance, it does not matter if player A is followed by B and then C or by C and then B. Thirdly, the method depends on all the dierent combinations of the players. Therefore, the number of possible coalitions increase exponentially as the number of players increases, making the running time longer as well.

### 2.2 The adjusted model 
In the case of attribution, the players here are the online channels and the goal of the game is to drive conversions for the website. There are k channels existing in the online marketing game. Let S be a combination of channels and its value v(S) be the conversion rate of all the journeys that share this combination. Another way of defining v(S) is the total conversions of the journeys that have the same channels in S. However, this way would violate the Efficiency property of the Shapley Value method on the following grounds. When the number of participating channels increases, the number of journeys decrease because very few customers would use many different channels. Consequently, the number of conversions becomes smaller while the conversion rate is likely to become higher. The reason is that a customer who comes through many channels has more information and would be more convinced to make some conversions than the one that has less information. As a result, when multiple channels work together, the total number of conversions would be much smaller than that of an individual channel while the conversion rate would be higher than that of each channel.

Moreoever, another adjustment is remodelling the multi-channel game such that each coalition becomes a sub-game that can satisfy the main properties of the Shapley Value method as mention above. Then each coalition S is the grand coalition of its own sub-game, and the value v(S) is the total Shapley values of all the players in S. The reason for this change is the following. There would be quite rare that a customer journey would contain all k channels, and it is very likely that there would be no such journey. Under the second scenario, the original method would force the players to have zero or negative credits because the Efficiency property states that the Shapley values of all players sum up to the value of the grand coalition, which is the case that all the players participate in. This credit assignment is undesirable for the players since they can gain higher payoffs for themselves by cooperating with fewer members. As a result, since the new model considers the combination of all k channels only as a sub-game so that the Shapley values assigned to the players here are just representative of this sub-game, it assists the players to benefit from the payoffs of other potential coalitions. Given these two modifications, the model becomes suitable for the multi-channel attribution game while maintaining the main properties of the original version. However, it still has the weaknesses of the original model which are the unavailability of the number of channel occurrences, the lack of order, and the long computation time. 

Hence, we can apply the original model as described in the previous section to each sub-game, where all utility functions are defined as the conversion rates of different combinations of players, and obtain the Shapley value of each channel in all the sub-games. To get the attribution of a channel, we can sum all the products of its Shapley value and the total number of journeys per sub-game. The below formula explains how to calculate the attribution of channel i in a multi-channel game.       

### 3. Markov Chain models
- **First Order model** only considers the current state and ignores the sequence of the states preceding it. This property is called the Markov property or the memoryless property. 



- **Higher Order models** also consider previous states of the sequence. The number of the previous states depends on the Order. For example, Second Order model would look back 2 steps in the sequence to calculate the probability of moving to the next state. 


### 4. Logistic Regression method 


## DATA SIMULATION
