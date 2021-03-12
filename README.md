- Sensorless Drive Diagnosis Data Set:
  + 58509 news items
  + 48 attributes, including columns 0-47. The property is of a continuous type, from [-15.796, 4015.4].
  + Column 48 is a column of labels, continuous values from 1-11.
  
  
 - Apply decision tree algorithm across the entire data set, using the hold-out evaluation protocol:
  + Randomly take 4/5 data sets to learn:
    * Number of elements for train: 46807, label value [1.11]
  + 1/5 of the remaining data set is used for testing:
    * Number of test set elements: 11702, label value [1.11]
    
 Accuracy = 98.64% > 75.47% = Accuracy (Bayes) 
 
 
