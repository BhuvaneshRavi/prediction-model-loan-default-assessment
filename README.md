# Prediction model for loan default assessment
Prediction model for loan default assessment using Machine Learning classifiers

Objective: 

The goal is to build a model that predicts a probability that a given customer will default on a loan. A customer profile consists of a list of bank transactions preceding the loan request. Each transaction is either a debit (money going out of account, negative values) or credit (money coming into account, positive values).

Dataset:
The model is trained and validated against a bank dataset for loan results on existing customers. 

Data available at Google Drive - https://drive.google.com/file/d/1oPSNCYeCVGJsTX60X-PW088R8S0AMmeT/view?usp=sharing.

Data Preprocessing:
The model is trained the on training data (instances 0 - 9999) and make predictions on the test data (instances 10000-14,999) Test-Train Split

Implementation in Python:
Prediction model is developed using Gradient Boosting classifier. The conceptual idea behind this classifier is to pick an algorithm and make tweaks to it with various regularization schemes, this process improves the learning ability of the model in a gradual and additive fashion. This classifier is particularly effective at classifying complex dataset such as Banking and Financing.

Accuracy:
The results predicted by model on the probability a customers will default on their loan repayment is accurate ~76% of the time (this is validated against a pre-defined validation dataset fed into the model during the train/test phase).

Results/Inference:
The build Machine learning Model is capable of identifying a potential set of customers who maintains a significant account balance and has regular intervals of credit transactions on their account, has high probability of loan repayment. The customers with less account transactions and maintains a very low account balance for a significant time before the loan request date, has a high probability of loan defaulting. The model is also capable of adjusting the prediction parameters of its features based on the future dataset fed into it.

Please refer to the report_on_prediction_inference.pdf for futher explanation. 

Contributions:

