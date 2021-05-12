# Multi-touch Attribution Models

## DATA SIMULATION
**Session data** 
- Visitor ID (or FullVisitorID as in Google Analytics) is assigned to each new user when they arrive at the website for the first time.
- Visit Start Time is the date and time that a user visits the website  
- NAW (the abbreviation for Naam-Adres-Woonplaats which is Name-Address-City in English) is a high-involvement conversion
- SOFT is a low-involvement conversion
- Channel Group is one of the following:  
> * Organic Search (OS): A visitor search for general keywords in a search engine and clicks on the webpage with no Ad sign. This webpage is ranked based on the relevance of the keywords.
> * Branded Paid Search (BPS): A visitor search for brand-relevant keywords in a search engine and clicks on the webpage with the Ad sign.
> * Generic Paid Search (GPS): A visitor search for general keywords in a search engine and clicks the on the webpage with the Ad sign.  
> * Direct: A visitor accesses the website by typing the website address in the web browser, or by bookmarking the website.
> * Display: By clicking banner advertisements on external websites, a visitor arrives at the advertiser's website
> * Emai: A visitor reaches the website by clicking an email sent to them since they agree to receive emails like newsletters from the advertiser.
> * Social: A visitor comes from social media platforms which are employed by the advertiser to attract users from the platforms like Facebook, Twitter.
> * Referral: A visitor clicks from external websites like price comparing websites.
> * Other: Other marketing forms that don't belong to the above types.


**Journey data**: which each journey reflects all the touch-points that a user comes by before reaching any conversion or a non-conversion
- Visitor ID 
- Conversion:
> * NAW: only high-involvement conversions
> * SOFT: only low-involvement conversions
> * TOTAL: both low- and high-involvement conversions
- Journey: all the touch-points in the journey are presented in chronological order. If the time between 2 touch-points pass the threshold of 30 days or any conversion is reached during one session, a new journey will be created.     

**Here is an example of how session data can be translated to journey data**
Session data: 
![session data](https://user-images.githubusercontent.com/66676705/117973830-245e6b80-b32d-11eb-93b2-dab9964dd89f.PNG)

Journey data with NAW conversion:
![naw - path](https://user-images.githubusercontent.com/66676705/117973806-1c9ec700-b32d-11eb-807e-b60912f5c7b5.PNG)

Journey data with NAW conversion:
![soft - path](https://user-images.githubusercontent.com/66676705/117973807-1d375d80-b32d-11eb-925f-421b2bde8df9.PNG)

Journey data with TOTAL conversion:
![total - path](https://user-images.githubusercontent.com/66676705/117973810-1dcff400-b32d-11eb-8b66-18ab4b2dad1f.PNG)

## METHODOLOGY 
### 1. Heuristic models
 - **First Touch Attribution (FTA)** appoints all the credits of a conversion to the first channel of a customer journey.
![fta](https://user-images.githubusercontent.com/66676705/117965630-4a7f0e00-b323-11eb-9cdb-35d165e143bc.PNG)

 - **Last Touch Attribution (LTA)** is similar to FTA but appoints the credits to the last channel.
![lta](https://user-images.githubusercontent.com/66676705/117965629-4a7f0e00-b323-11eb-8230-9d3a163c9a02.PNG)

<!-- ![Single-touch](https://user-images.githubusercontent.com/66676705/117957951-d476a900-b31a-11eb-89fe-f75db352a628.PNG) -->

 - **Linear Touch Attribution (LINEAR)** gives equal weight(s) to all channel(s) in a customer journey that results in conversion.
 - 
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

![Model comparison](https://user-images.githubusercontent.com/66676705/117968981-1ad20500-b327-11eb-86cd-9ae62c56a6b2.PNG)

### 6. Model evaluation
- **Predictive Accuracy**: The area under the curve (AUC) when each model predicts the conversion event of journeys.
- **Robustness**: The coefficient of variation of AUC per channel: 𝐶𝑉<sub>𝐴𝑈𝐶</sub>=𝜎<sub>𝐴𝑈𝐶</sub>/𝜇<sub>𝐴𝑈𝐶</sub>. Then calculate the average coefficient of variation of channel attribution over all channels 
- Different settings of online conversions:
NAW: high-involvement conversion level
SOFT: low-involvement conversion level
TOTAL: combination of both conversion levels
