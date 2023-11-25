# Name-Classifier
A naïve Bayes classifier in Python using a bag-of-character trigrams and incorporating start-and-end of word information to predict the language origin of surnames.

<pre>
<b>$</b> python3 classify.py nbse train.names.txt train.classes.txt test.names.txt > test.nbse.output

<b>$</b> python3 score.py test.nbse.output test.classes.txt\
Accuracy: 0.811\
Macro averaged P: 0.509\
Macro averaged R: 0.365\
Macro averaged F: 0.401

<b>$</b> python3 classify.py lr train.names.txt train.classes.txt test.names.txt > test.lr.output

<b>$</b> python3 score.py test.lr.output test.classes.txt\
Accuracy: 0.784\
Macro averaged P: 0.554\
Macro averaged R: 0.299\
Macro averaged F: 0.347
</pre>
