# Data-Mining

## Project 1  
Classify articles based on their title and their content using the following algorithms:  
1. Knn (our implementation)
2. Multinomial Naive-Bayes
3. Support Vector machines (SVM)
4. Random Forests  
  
*For each algorithm we did a grid search for all the parameters needen so we can achieve the best possible accuracy.*
For part 3 of the project, we had to achieve the best possible accuracy we could using every suitable algorithm for this problem.
To achieve that, we used PorterStemer and strip_punctuation while preproccessing the test set. Also we used OneVsRest with SVM algorithm, to achieve our best score: 97% accuracy. 

## Project 2
Classify bus route data (timestamp and coordinates), to their respective bus route using DTW, LCSS and KNN algorithms.
