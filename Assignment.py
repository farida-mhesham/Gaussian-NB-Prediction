import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data=pd.read_csv("adult.data", header=None)
data.columns=["age", "type_employer", "fnlwgt", "education",
                "education_num","marital", "occupation", "relationship", "race","sex",
                "capital_gain", "capital_loss", "hr_per_week","country", "income"]
print(data.head)
test=pd.read_csv("adult.test",header=None)
test.columns=["age", "type_employer", "fnlwgt", "education",
                "education_num","marital", "occupation", "relationship", "race","sex",
                "capital_gain", "capital_loss", "hr_per_week","country", "income"]
print(test.head)
x_train=data.drop('income',axis='columns')
x_test=test.drop('income',axis='columns')
y_train = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
y_test = test['income'].apply(lambda x: 1 if x.strip() == '>50K.' else 0)


combined_values=pd.concat([x_train,x_test],axis=0,ignore_index=True)
print(combined_values.head)

categorical_columns=combined_values.select_dtypes(exclude=[np.number]).columns
numerical_columns=combined_values.select_dtypes(include=[np.number]).columns
encoder=OneHotEncoder(drop='first', sparse=False)
encoded_data=encoder.fit_transform(combined_values[categorical_columns])
encoded_df=pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out(categorical_columns))
data_encoded=pd.concat([combined_values[numerical_columns],encoded_df],axis=1)
print(data_encoded.head)

x_train_encoded = data_encoded.iloc[:len(x_train), :].reset_index(drop=True)
x_test_encoded = data_encoded.iloc[len(x_train):, :].reset_index(drop=True)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train_encoded, y_train)
accuracy = model.score(x_test_encoded, y_test)
print("Accuracy:", accuracy)
predictions = model.predict(x_test_encoded)
print("Predictions:", predictions)

tn,fp,fn,tp=confusion_matrix(y_test,predictions).ravel()
sensitivity=tp/(tp+fn)
print("Sensitivity = ",sensitivity)
specificity=tn/(tn+fp)
print("Specificity = ",specificity)
posterior_probability=model.predict_proba(x_test_encoded)
print("posterior probability = ",posterior_probability)

