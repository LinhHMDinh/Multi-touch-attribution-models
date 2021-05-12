# Multi-touch Attribution Models

## METHODOLOGY 
### 1. Heuristic models
 - **First Touch Attribution (FTA)** appoints all the credits of a conversion to the first channel of a customer journey.
![fta](https://user-images.githubusercontent.com/66676705/117965630-4a7f0e00-b323-11eb-9cdb-35d165e143bc.PNG)

 - **Last Touch Attribution (LTA)** is similar to FTA but appoints the credits to the last channel.
![lta](https://user-images.githubusercontent.com/66676705/117965629-4a7f0e00-b323-11eb-8230-9d3a163c9a02.PNG)

<!-- ![Single-touch](https://user-images.githubusercontent.com/66676705/117957951-d476a900-b31a-11eb-89fe-f75db352a628.PNG) -->

 - **Linear Touch Attribution (LINEAR)** gives equal weight(s) to all channel(s) in a customer journey that results in conversion.
![linear](https://user-images.githubusercontent.com/66676705/117965625-49e67780-b323-11eb-98f8-90f1654c62a6.PNG)
 
<!--![Multiple-touch](https://user-images.githubusercontent.com/66676705/117957947-d3de1280-b31a-11eb-8a51-263ae386fe97.PNG) -->

### 2. Shapley Value method 
- Assigns credits among players according to their contribution to a game.
- The Shapley Value of channel 𝑖 in a sub-game 𝑔 is 𝜙<sub>𝑖</sub> (𝑔) 
- The attribution of channel 𝑖 in a multi-channel game 𝐺: 
<img src="https://user-images.githubusercontent.com/66676705/117967286-27555e00-b325-11eb-90ba-c6b9129f3184.PNG" width="250" height="50"> 

There are some properties that the Shapley Value method needs to satisfy:
- Efficiency: The Shapley values of all players sum up to the value of the grand coalition so that the total value of the game is distributed among the players.
- Symmetry: This guarantees that players with the same contribution to every coalition should receive equal credits. 
- Dummy Player: A player that does not contribute to any coalition should receive no payoff.

### 3. Logistic Regression method 
- **The original model**: predicts conversion or no conversion given channels in a journey
- **The regularized model**: a shrinkage term is added to avoid overfitting
- The attribution of a channel is computed using Shapley Value method. 

### 4. Markov Chain models
- **First Order model** only considers the current state and ignores the sequence of the states preceding it. This property is called the Markov property or the memoryless property. 

![markov1](https://user-images.githubusercontent.com/66676705/117965226-cfb5f300-b322-11eb-8ad6-6d2f277bdbaa.PNG)

- **Higher Order models** also consider previous states of the sequence. The number of the previous states depends on the Order. For example, Second Order model would look back 2 steps in the sequence to calculate the probability of moving to the next state.

![markov2](https://user-images.githubusercontent.com/66676705/117965231-cfb5f300-b322-11eb-944d-ec942e473406.PNG)

### 5. Model comparison


### 6. Model evaluation

## DATA SIMULATION
