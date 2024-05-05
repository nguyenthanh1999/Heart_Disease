# **Multiple Disease Prediction**

## **Introduction**
This is my self-learning project. The dataset is collected from Kaggle, containing 11 common features curated by combining 5 popular heart disease datasets already available independently but not combined before.These datasets were collected and combined at one place to help advance research on CAD-related machine learning and data mining algorithms, and hopefully to ultimately advance clinical diagnosis and early treatment.

The features description of the dataset is noted clearly in [documentation.pdf](documentation.pdf). However, for convenience, I've made a few edits.
```python
'''
'chest_pain':{
    'asymptomatic':0
    'non-anginal_pain':1
    'atypical_angina':2
    'typical_angina':3
    }
'''
```
```python
'''
'ST_slope':{
    downsloping:-1
    flat:0
    upsloping:1
    }
'''
```
```python
'''
'sex':{
    'male':0
    'female':1
    }
'''
```
From this dataset, I plan to visualize and conduct analytical searches to determine the correlation of each feature with heart disease. My goal is to identify the most important features for predicting whether someone will have heart disease or not. Subsequently, I aim to build a model capable of predicting the likelihood of heart disease based on CAD status.

*Here is the link to the data source:* [https://www.kaggle.com/datasets/mexwell/heart-disease-dataset/data](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset/data)

## **Methodology**
The dataset is well-prepared, although there are a few instances still presenting issues, such as some with cholesterol serum equal to 0, which is impossible in a medical context, or some very rare instances with extremely serious health problems, such as maximum heart rate less than 80. However, overall, the data creator has done an excellent job in its creation. I just need to make a few adjustments, such as dropping those unrealistic instances and applying standard scaling to the features, to ensure that the data works seamlessly with the library package.

A quick check through the dataset reveals that there are no null values. There appear to be a few duplicated instances, so I removed them and engineered some features that I thought would help in my analysis and model-building.

I utilize the Matplotlib and Seaborn libraries to visualize and analyze the dataset. By examining the plots generated from these libraries, I gain further insights into the data. Finally, I use models from libraries such as Scikit-learn to train, select the best model, and fine-tune it using GridSearchCV for improved predictive performance.

Because I've dropped some unrealistic instances, the data now seems quite balanced, so I don't have to do anything further to balance it.

## **Data Description**
The dataset comprises 1190 instances, each associated with one of two labels representing whether they have heart disease or not. Each instance contains 11 features. As mentioned before, I've dropped some unrealistic instances and engineered 2 more features to enhance my work. Additionally, I've dropped duplicate instances, resulting in a dataset with only 746 usable instances and a total of 13 features, including parameters such as "age", "max_heart_rate", "cholesterol", "oldpeak", and others.

From this dataset, I can visualize and conduct analytical searches to determine the correlation of each feature with heart disease and identify the most important features for predicting whether someone will have heart disease or not. For example, individuals with an age higher than 54 may have a higher chance of having heart disease...

## **Results**
To start, I use a countplot to visualize the distribution of the label column in the dataset.

    Labels Countplot
<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\1.Count_plots_before_cleaning.png" alt="Label countplot before cleaning">
</div>

The countplot of labels clearly illustrates all the label names and the number of instances for each label. From this visualization, we can observe the imbalance in the dataset. Let's use a histplot and boxplot to examine the distributions of each feature to see if there are any issues with the dataset.

    Histplot Distributions

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\2.Distributions_before_cleaning.png" alt="Distributions before cleaning">
</div>

    Boxplot Distributions
 
<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\3.Box_plots_before_cleaning.png" alt="Box plots before cleaning">
</div>

In general, the distribution of features seems quite standard. However, there are some features with unusual data, most notably 'cholesterol', which has a remarkable number of zero values. According to what I have learned, zero serum cholesterol is unrealistic because cholesterol is a lipid molecule essential for various physiological processes, including the formation of cell membranes and the synthesis of hormones. Therefore, cholesterol serum equal 0 is unrealistic.

This could be due to a small mistake during data collection, but we cannot determine which value is correct to fix it. Therefore, I have decided to remove them from the dataset.

And here is what I got:

    Labels Countplot

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\4.Count_plots_after_cleaning.png" alt="Label countplot after cleaning">
</div>

    Histplot distributions

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\5.Distributions_after_cleaning.png" alt="Distributions after cleaning">
</div>

    Boxplot distributions
 
<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\6.Box_plots_after_cleaning.png" alt="Box plots after cleaning">
</div>

After removing approximately 172 instances with 'cholesterol' feature values equal to 0, I observed that 390 instances were labeled as 0 and 356 instances were labeled as 1, remaining in the dataset. The output labels appear to be fairly balanced, with nearly all feature distributions being standard.

Additionally, some very rare and unusual instances with extremely serious health problems signal, such as maximum heart rate less than 70 or resting blood pressure values equal to 0, have also been removed alongside those instances where 'cholesterol' feature values equal to 0. The distribution of all features is now more reliable.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\7.Correlation_matrix.png" alt="Correlation_matrix">
</div>

From the correlation matrix, we can observe that features such as 'age', 'exercise_angina', and 'oldpeak' exhibit strong positive correlations with the output 'heart_disease'. This suggests that higher values of 'age', 'exercise_angina', and 'oldpeak' correspond to a higher likelihood of the output being 'heart_disease' = 1. Conversely, features such as 'sex', 'chest_pain', 'max_heart_rate', and 'ST_slope' display strong negative correlations with the output 'heart_disease'. This indicates that lower values of 'sex', 'chest_pain', 'max_heart_rate', and 'ST_slope' are associated with a higher likelihood of the output being 'heart_disease' = 1. Other features have a slightly weaker correlation compared to the aforementioned ones.

For example, when examining the distribution of the 'age' feature

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\8.a.Distributions_by_labels_for_age.png" alt="Distributions_by_labels_for_age">
</div>

The distribution of the 'age' feature differs between cases where the output is labeled as 0 and 1, particularly around the age of 54. When 'heart_disease' = 0, there are numerous instances distributed in the age range lower than 54. Conversely, when 'heart_disease' = 1, the distribution is more focused on the age range higher than 54.

From that, I have divided the 'age' feature into two ranges: lower than 54 and higher than 54, and engineered a new feature based on these two ranges.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\8.b.Pie_plots_for_age_range.png" alt="Pie_plots_for_age_range">
</div>

We can see the difference more clearly through these pie plots. When 'heart_disease' value equals 0, the distribution of these instances focuses mostly on the lower range, below 54, represented by the value of 1. Conversely, when 'heart_disease' value equals 1, the distribution of these instances focuses mostly on the higher range, above 54, represented by the value of 2.

Or in case of 'max_heart_rate' feature

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\15.a.Distributions_by_labels_for_max_heart_rate.png" alt="Distributions_by_labels_for_max_heart_rate">
</div>

We can observe a completely opposite pattern in the distribution of the 'max_heart_rate' feature.

It is similar to the 'age' feature in that its distribution is clearly divided into two ranges for the two different 'heart_disease' labels. However, it is opposite to the 'age' feature in that when 'heart_disease' = 0, most instances focus on a range higher than 140. Conversely, when 'heart_disease' = 1, a significant number of instances focus on a range lower than 140.

So I engineered a new feature by dividing the 'max_heart_rate' feature into two ranges: one for values lower than 140 presented by 1 and another for values higher than 140 presented by 2.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\15.b.Pie_plots_for_max_heart_rate_range.png" alt="Pie_plots_for_max_heart_rate_range">
</div>

The pie plots show us a clearer difference in distribution between both cases of 'heart_disease' from the two ranges of maximum heart rate values, which I named 'max_heart_rate_range'.

For features with weak correlations, for example 'cholesterol' feature.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\12.Distributions_by_labels_for_cholesterol.png" alt="Distributions_by_labels_for_cholesterol">
</div>

we can observe similar distributions in both cases where 'heart_disease' = 0 and 'heart_disease' = 1. The main difference is the proportion of each feature value in each case.

*For more details, please refer to* [report.docx](report.docx).


**Model selecting** 
---
After completing the analytical phase, I perform preprocessing steps such as rescale all the features using StandardScaler to enhance the model's learning ability for prediction on the dataset. Subsequently, I utilize models from libraries such as Scikit-learn to train, select the best model, and fine-tune it using GridSearchCV for improved predictive performance.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\19.Models_result.png" alt="Images\19.Models_result.png">
</div>

From the results, it is evident that the Random Forest model consistently achieves the highest scores in all evaluation metrics, followed by the SVM model. Conversely, the Logistic Regression model consistently produces the lowest scores across all evaluation metrics.

A Random Forest model achieving a training accuracy of 1.0 (or 100%) suggests a perfect fit to the training data. While this might seem impressive at first glance, it could also indicate potential issues, such as overfitting.

In disease prediction, prioritizing recall (sensitivity) to maximize the detection of true positive cases is typically crucial, followed by maintaining a balance between precision and recall (as indicated by the F1-score) to ensure accurate and reliable predictions. Given that the SVM model demonstrates the highest scores after Random Forest model in both metrics, it is the preferred choice for this dataset.

Finally, with GridSearchCV and the chosen hyperparameters, I tuned my model to achieve the following results:

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\20.a.SVM_tuned_result(1).png" alt="SVM_tuned_result(1).png">
</div>

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\20.b.SVM_tuned_result(2).png" alt="SVM_tuned_result(2).png">
</div>


