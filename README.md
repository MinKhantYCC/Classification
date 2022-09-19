# Classification
In this project we will make an algorithm of the supervised machine learning method, a classification (logistics regression).
Logistic Regression can be used to classified an observation into one of two types (binary classification) or one of many types (multinominal classification). It's also similar to Neural Network principal.
In this algorithm, we will use sigmoid function in order to predict our target type. The formula of sigmoid function is as follow

z(x:theta) = theta' * X
y_pred = 1/(1+e^(-z))

where,
    X = input feataures (# of samples * # of features)
    theta  = input weight (# of features+intercept * # of class)
    y_pred = predicted values (# of sample * 1)

To train the model (finding theta values), cost function and gradient descent function are used. Regularization parameter (lambda) can be tuned to prevent overfitting, and its default value is 0.

For model performance evaluation, accuracy_score function from sci-kit learn is used.

You can freely use any part of these code or the whole code. If there is some bug in the code, I sincerely aplogize and request you to notify me, please.

Thanks!
Min Khant