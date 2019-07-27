# cuisine

A text classifier:

Predict cuisine based on ingredients

## Background
You've just joined the data team at an online publishing company. One of your verticals is a food publication. A product manager on your team wants to build a feature for this vertical that enables users to query by cuisine, not just by ingredients. Most of your recipes are unlabeled, and it's infeasible to label them by hand. Luckily, you have a small training set of about 10,000 recipes with labeled cuisines.
Design and execute a method to predict the cuisine of a recipe given only its ingredients. 

Data Due Diligence: All-Purpose Flour and Flour are likely the same ingredient, but red onions and yellow onions are incredibly different.
For each major cuisine, what are the driving ingredients that characterize it? What are the features of a cuisine that drive misclassification in your method above?

How could you design this to be robust enough to understand similarities / substitutions between ingredients? 

Your product manager indicates a likelihood that you will only need to write a guideline for an outsourced team to hand label the remaining corpus. How would you go about writing this guide for a few major cuisines?

## Solutions:
First look of the data: Imbalanced dataset, can anticipate that the classifier will work better in the majority class

<img src = images/image_0.png>

Use tSNE and word2vec to visualize the ingredients vs cuisines. I first converted the ingredients in text data to vector data using word2vec. I then use tSNE to reduce the dimensions of the vector space to 2D. Finally, I color the data with the most likely cuisine (in which the ingredients appear most often). We can see the five majority classes have clear clustering. 

<img src = images/image_1.png>

My next task is to generate an instruction for hand labelling for a few major cuisine. As the top 5 cuisine always have clear clustering in our tSNE. I chose a shallow tree model to generate the hand labelling for only these 5 cuisine. I limited the number of leaves and the depth of the tree so that the result of the tree classifier can be used for hand labelling. 

<img src = images/image_2.png>

I then used a random forest classifier to classify all cuisine to see if I can find an AI solution to replace hand labelling. The result is as shown below. One thing I noticed that is although there are almost equal amount of common ingredients between 'brazillian' and 'italian', and 'brazillian' and 'chinese', it is quite likely to classify 'brazillian' as 'italian' but not 'chinese'. Therefore, it is the combination of ingredients that matters.

<img src = images/image_3.jpg>

I decided to use bidirectional RNN model, which shows a great improvement from the random forest.

<img src = images/image_4.png>

Finally, I used data augmentation of the text data to increase the number of examples of the minority classes. As the order of the ingredients doesn't matter, we can shuffle the ingredient text data around to generate new training examples. By doing that, finally, I improve further the results. For example, british improved from 0.28 to 0.56, filipino improved form 0.36 to 0.58.

<img src = images/image_5.png>

<img src = images/image_6.png>


## Try the code

The code for this data chanllenge is in the notebook folder. Github doesn't always load well for large ipynb file, therefore, I also converted the ipynb file to mb file in the folder.
