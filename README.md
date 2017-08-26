# arXiv Decision Tree Classifier

Full Python implementation of a decision tree classifier with pruning. Classifies arXiv abstracts by topic.

Sample usage:

```
python decision_tree.py -h 

python decision_tree.py -i ../Data/train_in.csv -o ../Data/train_out.csv -f 400 -e 1.5 -s 1000

```


A decision tree will be trained on `train_in.csv` and evaluated against `train_out.csv`. It will use 400 words as features and have an entropy threshold of 1.5, it will use 1000 examples to train the tree.

Will produce a file `DTmetrics.txt` containing the performance metrics of the decision tree classifier.
