# Credit Card Approval
Predicting if credit card request will approved for the customer given other attributes.

### Motivation
Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). This task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do.

### Libraries Used
  - sklearn
  - pandas
  - numpy
  - seaborn
  - matplotlib

To install the above packages using pip, use command
```$pip install package-name```

### Files Description
- credit-approval_pandas.csv: This is the dataset file for the project
- The initial dataset has been analysed on Tableau as well at https://public.tableau.com/profile/akshit.arora8313#!/
- Data is downloaded from http://archive.ics.uci.edu/ml/datasets/credit+approval
- CreditApproval.ipynb: The jupyter notebook which includes the analysis and modeling
- The project also includes a front-end webiste which can be deloyed using flask app and will provide with "Approved" or "Not Approved" status on entering all the parameters
- For the front-end refer to the folder Credit Card Approval, first run the model.py and then the app.py file

### Usage
  - Run the jupyter notebook CredictCardApproval.ipynb

### Summary
In this project, I have tried to find out the factors that are most important for getting an 
approval for the credit card through the power of Data Analysis and Machine Learning. 
Though we have achieved 86% of accuracy, we also tried to check if we can improve the 
performance further and tried decision tree. 
However, 86% is the best we could get from this data using both the model; 
logistic regression and random forest
