# GravitationalWave
My research work at UIUC

1. Generating dataset to train from given dataset
The given dataset is normalized to have total mass 1 and a variety of massratios. 
To create training and testing dataset, we have to generate signals with different total masses and massratios. 
What I do in generate_dataset is to read given dataset and to stretch the signal with times of the total mass.
Then I take the final 1 second of the stretched signal and down sample it to 8192Hz to get training signals and testing signals.
I use different sets of total masses in training and testing so they are totally seperate.

2. 
