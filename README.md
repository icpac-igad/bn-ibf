# How do Bayesian Networks support impact-based forecasting for informed decision-making?


Impact Based Forecasting as Risk Matrices









Mindmap into code
```
import pyAgrum as gum

# Create an empty Bayesian Network
bn = gum.BayesNet("FloodModel")

# Adding nodes (variables) to the Bayesian Network
rainfall = bn.add(gum.LabelizedVariable("rainfall", "Rainfall", 2)) # 2 states: Low, High
river_level = bn.add(gum.LabelizedVariable("river_level", "River water level", 2)) # 2 states: Low, High
flood_barriers = bn.add(gum.LabelizedVariable("flood_barriers", "Quality of flood barriers", 2)) # 2 states: Low, High
sandbags = bn.add(gum.LabelizedVariable("sandbags", "Availability of sandbags", 2)) # 2 states: Low, High
emergency_services = bn.add(gum.LabelizedVariable("emergency_services", "Quality of emergency services", 2)) # 2 states: Low, High
flood = bn.add(gum.LabelizedVariable("flood", "Flood", 2)) # 2 states: Yes, No
people_drown = bn.add(gum.LabelizedVariable("people_drown", "People drown", 2)) # 2 states: Yes, No
houses_ruined = bn.add(gum.LabelizedVariable("houses_ruined", "Houses ruined", 2)) # 2 states: Yes, No

# Adding edges (conditional dependencies) based on the mind map
bn.addArc(rainfall, river_level)
bn.addArc(rainfall, flood)
bn.addArc(river_level, flood)
bn.addArc(flood_barriers, flood)
bn.addArc(sandbags, flood)
bn.addArc(emergency_services, people_drown)
bn.addArc(flood, people_drown)
bn.addArc(flood, houses_ruined)

# You can continue to set conditional probability tables (CPTs) for each node based on real data or expert opinions.
# For example, 
# bn.cpt(rainfall).fillWith([0.7, 0.3]) # 70% Low, 30% High
# ...

# CPT for Rainfall: Let's assume heavy rainfall is forecasted
bn.cpt(rainfall).fillWith([0.2, 0.8]) # 20% Low, 80% High

# CPT for River water level given Rainfall
bn.cpt(river_level)[{'rainfall': 0}] = [0.9, 0.1] # Given low rainfall: 90% Low river level, 10% High
bn.cpt(river_level)[{'rainfall': 1}] = [0.4, 0.6] # Given high rainfall: 40% Low river level, 60% High

# CPT for Quality of flood barriers: Less developed parts have poor quality
bn.cpt(flood_barriers).fillWith([0.6, 0.4]) # 60% Low quality, 40% High quality

# CPT for Availability of sandbags: Limited availability
bn.cpt(sandbags).fillWith([0.7, 0.3]) # 70% Low availability, 30% High availability

# CPT for Quality of emergency services
bn.cpt(emergency_services).fillWith([0.5, 0.5]) # 50% Low quality, 50% High quality

# CPT for Flood based on various factors
# For simplification, let's assume flood happens if any two of the conditions (high rainfall, high river level, low flood barriers, low sandbags) are met
bn.cpt(flood)[:,:,:,:] = [[[0.9, 0.1], [0.8, 0.2]], [[0.8, 0.2], [0.6, 0.4]]] # And similarly for all combinations

# CPT for People drown based on flood and emergency services
# If flood and low-quality emergency services, higher chance of drowning
bn.cpt(people_drown)[:,:] = [[0.1, 0.9], [0.05, 0.95]] # And similarly for other combinations

# CPT for Houses ruined based on flood
bn.cpt(houses_ruined)[:] = [0.3, 0.7] # 30% Not ruined, 70% Ruined given a flood

# Display the Bayesian Network
gnb.showBN(bn)
```




1. Decision Making: Prioritizing Investments in Flood Infrastructure
Suppose Nairobi city officials want to determine the best investment to reduce flood risk. They can do this by comparing the probabilities of flooding under different scenarios.
```
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

# [code to create the BN and set the CPTs from previous examples here]

# Without any investments
no_investment_flood_prob = bn.cpt('flood')[1]

# Improve flood barriers
bn.cpt('flood_barriers')[{ 'rainfall': 1 }] = [0.3, 0.7]  # assume barriers improvement reduces flood risk
barriers_investment_flood_prob = bn.cpt('flood')[1]

# Increase sandbag availability
bn.cpt('sandbags')[{ 'rainfall': 1 }] = [0.4, 0.6]  # assume sandbag availability reduces flood risk
sandbags_investment_flood_prob = bn.cpt('flood')[1]

# Compare
print(f"No Investment: {no_investment_flood_prob}% chance of flood")
print(f"Barriers Investment: {barriers_investment_flood_prob}% chance of flood")
print(f"Sandbags Investment: {sandbags_investment_flood_prob}% chance of flood")
```
2. Causality: Analyzing the Root Causes of Floods
By using interventions (do-calculus) in BN, we can understand the direct causes of flooding.
```
# Setting the value of 'flood' to 1 (occurred)
ie = gum.LazyPropagation(bn)
ie.setEvidence({'flood': 1})
ie.makeInference()

print(f"Given a flood occurred:")
print(f"Probability of heavy rainfall: {ie.posterior('rainfall')[1]}")
print(f"Probability of river level rise: {ie.posterior('river_level')[1]}")
```
3. Anomaly Detection: Identifying Unusual Rainfall Patterns
We can detect anomalies by comparing the observed rainfall data with the predicted probabilities from our BN model.

```
# Generate a series of observations (for the sake of this example, let's assume a sequence)
observations = [0, 0, 1, 1, 1, 1, 0, 0, 1, 0]  # 1 indicates heavy rainfall

# Compute probabilities
predicted_probs = [bn.cpt('rainfall')[1] for _ in observations]

# Detect anomalies (if observed data significantly deviates from predicted)
anomalies = [obs for i, obs in enumerate(observations) if abs(obs - predicted_probs[i]) > 0.5]

print(f"Detected anomalies: {anomalies}")
```


1. Diagnostics: Identifying the Causes of a Flood
Question: Given that a flood has occurred, what's the likelihood that it was caused by heavy rainfall or a rise in the river water level?
```
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

# [code to create the BN and set the CPTs from previous examples here]

# Set evidence that a flood has occurred
ie = gum.LazyPropagation(bn)
ie.setEvidence({'flood': 1})
ie.makeInference()

print(f"Given a flood occurred:")
print(f"Probability of heavy rainfall: {ie.posterior('rainfall')[1]:.2%}")
print(f"Probability of river level rise: {ie.posterior('river_level')[1]:.2%}")
```
2. Predictions: Assessing the Risk of People Drowning during a Flood
Question: If heavy rainfall is forecasted and the quality of emergency services is poor, what's the probability of people drowning?
```
# Set evidence for heavy rainfall and poor emergency services
ie.setEvidence({'rainfall': 1, 'emergency_services': 1})  # assuming 1 represents poor quality
ie.makeInference()

print(f"Given heavy rainfall and poor emergency services:")
print(f"Probability of people drowning: {ie.posterior('people_drown')[1]:.2%}")
```
3. Predictions: Estimating the Risk of Houses Getting Ruined
Question: If there's a high river water level and the availability of sandbags is low, what's the risk to houses?
```
# Set evidence for high river water level and low availability of sandbags
ie.setEvidence({'river_level': 1, 'sandbags': 1})  # assuming 1 represents low availability
ie.makeInference()

print(f"Given high river water level and low availability of sandbags:")
print(f"Probability of houses getting ruined: {ie.posterior('houses_ruined')[1]:.2%}")
```


1. Decision Making under Uncertainty (Decision Networks)
Decision Networks (or influence diagrams) extend Bayesian Networks to include decision nodes and utility nodes, which help in making informed decisions.

Question: Should the city of Nairobi invest in improving flood barriers or increasing the availability of sandbags, considering both the probability of floods and associated costs?
```
import pyAgrum as gum

# [code to create the BN and set the CPTs from previous examples here]

# Create a decision node for the strategy: "improve_barriers" or "more_sandbags"
strategy = gum.LabelizedVariable('strategy', '', 2)
strategy.changeLabel(0, 'improve_barriers')
strategy.changeLabel(1, 'more_sandbags')
strategy_node = bn.add(strategy)

# Create a utility node for the cost and potential damages saved
utility = gum.FunctionalVariable('utility', [bn.variableId('flood'), strategy_node])
bn.add(utility)

# [Fill in the CPT and utility table considering costs and damages]

# [Run inference to decide on the best strategy]
```


2. Intercausal Reasoning
Question: If a flood occurs and we know that the flood barriers were of poor quality, how does that alter the probability of heavy rainfall?
```
ie.setEvidence({'flood': 1, 'flood_barriers': 1})  # assuming 1 means poor quality
ie.makeInference()

print(f"Given a flood and poor flood barriers:")
print(f"Probability of heavy rainfall: {ie.posterior('rainfall')[1]:.2%}")
```

3. Sensitivity Analysis
Question: How sensitive is the flood occurrence to changes in the quality of emergency services?
```
original_prob = ie.posterior('flood')[1]
ie.setEvidence({'emergency_services': 1})  # poor quality
ie.makeInference()
changed_prob = ie.posterior('flood')[1]

print(f"Sensitivity of flood occurrence due to changes in emergency services quality: {changed_prob - original_prob:.2%}")
```


4. Scenario Analysis
Question: How does the probability of houses getting ruined change under different scenarios of rainfall, river level, and sandbag availability?
```
scenarios = [
    {'rainfall': 1, 'river_level': 1, 'sandbags': 0},
    {'rainfall': 1, 'river_level': 0, 'sandbags': 1},
    {'rainfall': 0, 'river_level': 1, 'sandbags': 1}
]

for scenario in scenarios:
    ie.setEvidence(scenario)
    ie.makeInference()
    print(f"Scenario {scenario}:")
    print(f"Probability of houses getting ruined: {ie.posterior('houses_ruined')[1]:.2%}\n")
```


Counterfactual reasoning 
Question: Given that a flood has already occurred (observed evidence), what would the probability of people drowning have been if the quality of emergency services had been good (counterfactual state)?
'''
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

# [code to create the BN and set the CPTs from previous examples here]

# Step 1: Set the observed evidence (a flood has occurred)
ie = gum.LazyPropagation(bn)
ie.setEvidence({'flood': 1})
ie.makeInference()

# Step 2: Compute the counterfactual by intervening on the BN
# Here, we're intervening to set 'emergency_services' to the state representing good quality (assuming it's 0)
counterfactual_bn = bn
counterfactual_bn.cpt('emergency_services')[:] = [1, 0]

# Step 3: Infer the probability of 'people_drown' in the counterfactual scenario
ie_counterfactual = gum.LazyPropagation(counterfactual_bn)
ie_counterfactual.setEvidence({'flood': 1})
ie_counterfactual.makeInference()
prob_people_drown = ie_counterfactual.posterior('people_drown')[1]

print(f"Given that a flood occurred, the counterfactual probability of people drowning (if emergency services had been good) is: {prob_people_drown:.2%}")
'''

Remember, the process involves:

Observing the evidence in the actual world.
Manipulating the BN to reflect the counterfactual world.
Making inference in the counterfactual world.

