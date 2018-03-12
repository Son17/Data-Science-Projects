
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import csv
from sklearn.decomposition import PCA
from scipy import stats
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV


# In[2]:

file=[]
with open('C:\Users\sonja tilly.ABERDEEN\\dataset_challenge.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        file.append(row)


# In[3]:

data = np.array(file)


# In[4]:

variables = data[1:, :-1]

target = data[1:, -1].astype(int)   


# In[5]:

# replace 'na' with zero

def replace_zero(vars):

    for var in vars:
        var[var == 'NA'] = 0
    return (vars)

variables = replace_zero(variables)


# In[33]:


class Dataset(object):
    '''A dataset that we load into the system'''

    def __init__(self, features, target):
        ""
        self.features = features
        self.target = target
 
    '''create summary statistics'''
    def summary_stats(self):
        summary_stats=[]
        for var in self.features:
            sum_stats = stats.describe(var.astype(float))
            summary_stats.append(sum_stats)
        return summary_stats
    
    '''Plot variable distributions'''
    def plot_distribution(self,feature_number):
        fig = plt.figure()
        sns.set_style("white")
        sns.distplot(self.features[feature_number].astype(float), kde=False, rug=True)
        plt.title('Distribution Graph for Feature: ' + str(feature_number))
        plt.tight_layout()
    
    '''perform principal component analysis'''
    def PCA_trans(self,nb_components):
        return(PCA(n_components=nb_components).fit_transform(self.features))
    
    '''make scatter plot'''
    def scatter_plot(self, var1, var2):
        var_redux = df.PCA_trans(2)
        redux_zipped = zip(var_redux, df.target)
        colors=[]
        for i in redux_zipped:
            if i[:][:][-1] ==1:
                colors.append('magenta')
            else:
                colors.append('seagreen')
        return (plt.scatter(var1, var2, c=colors))
    
    '''calculate correlation between features and a variable'''
    def corr(self, a_variable):
        results=[]
        for i in range(0,self.features.shape[1]):
            transp=self.features.transpose()[i]
            correlation=np.corrcoef(transp.astype('float'), a_variable)[0][1]
            results.append(correlation)
        return (results)
    
    '''extract kurtosis from summary statistics'''
    def kurtosis(self):
        kurtosis = []
        summary_stats = self.summary_stats()
        for sum in summary_stats:
            kurt = sum[:][:][-1]
            kurtosis.append(kurt)
        return (kurtosis)

    '''make regression plot'''
    def reg_plot(self, feature_number, a_feature, TrueorFalse):
        var_p= self.features[:,feature_number]
        sns.regplot(var_p.astype('float'), a_feature, logistic=TrueorFalse)
    
    '''create dictionary from a list with the length of the features'''
    def zipped(self, a_list):
        idx = range(len(self.features))
        zipped = dict(zip(idx, a_list))
        return (zipped)
    
    '''find a value large than a given number in a dictionary'''
    def find_values(self, a_list, number):
        zipped = dict(df.zipped(a_list))
        for k, v in zipped.items():
            if v > number:
                print(k, v)
        return
    
    '''find the minimum and maximum values and corresponding keys in a dictionary'''
    def min_max(self, zipped):
        minim = min(zipped, key=lambda k: zipped[k])
        maxim = max(zipped, key=lambda k: zipped[k])
        print("Minimum:", minim, zipped[minim], "Maximum:", maxim, zipped[maxim])
                 
    '''create a SVM classifier, cross-validate and return best parameters and score'''
    def svm(self):
        params = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, params, cv=5, scoring='f1_micro')
        clf.fit(self.features.astype('float'), self.target) 
        best_params = clf.best_params_
        best_score = clf.best_score_
        return ("best parameters:", best_params, "best score:", '{0:.4f}'.format(best_score))


# Question 1

# In[7]:

df = Dataset(variables,target)


# In[8]:

summary_stats = df.summary_stats()


# In[9]:

kurtosis = df.kurtosis()


# In[10]:

df.find_values(kurtosis, 3)


# In[11]:

features_to_plot=[79,103,231,263]

for i in features_to_plot:
    df.plot_distribution(i)


# The above variables have been chosen for their leptokurtic distributions, with kurtosis values above 3, which is the value for a univariate normal distribution. This means that the above distributions have fatter tails, or outliers. 
# The histogram with rugplot has been chosen as it allows to plot a univariate distribution of observations and draw small vertical lines to show each observation in a distribution.

# Question 2

# In[12]:

var_redux = df.PCA_trans(2)


# In[13]:

df.scatter_plot(var_redux[:,0], var_redux[:,1])


# Question 3a

# In[14]:

correlations = df.corr(target)


# In[15]:

zipped2 = dict(df.zipped(correlations))


# In[31]:

df.min_max(zipped2)


# In[17]:

df.reg_plot(196, target, True)


# Variable 196 was selected for visualisation as it exhibited the strongest relationship with the target variable based on correlation. The relationship is moderately positive with a correlation coefficient of +0.34.
# 
# The correlation coefficient was chosen to examine the relationship between the variables and the target variable as it represents the linear interdependence of two variables, describing the strength of the relationship and whether the relationship is negative or positive. A criticism is that the choice is overly simplistic as variables may have a non-linear relationship not captured by the metric. 
# The regression plot was chosen as it permits to plot data and a linear regression model fit. Since the target is binary, we are dealing with a classification task and the plot was adjusted for this setting logistic to True (statsmodels estimates a logistic regression model).
# 
# The chart illustrates a mildly positive relationship between the variable and the target.

# Question 3b

# In[18]:

pc1 = var_redux[:,0]


# In[19]:

correlation2 = df.corr(pc1)


# In[20]:

zipped3 = dict(df.zipped(correlation2))


# In[32]:

df.min_max(zipped3)


# In[22]:

df.reg_plot(6, pc1, False)


# Variable 6 was selected for visualisation as it exhibited the strongest relationship with the the first principal component based on correlation.
# 
# As in question 3a, the correlation coefficient was chosen to examine the relationship between the variables and the target variable as it represents the linear interdependence of two variables, describing the strength of the relationship and whether the relationship is negative or positive. A criticism is that the choice is overly simplistic as variables may have a non-linear relationship not captured by the metric. The regression plot was chosen as it permits to plot data and a linear regression model fit. 
# 
# The chart illustrates a fairly strong negative relationship between the variable and the first principal component.
# 

# Question 4

# In[23]:

df.svm()


# A support vector machine classifier was chosen as it is effective when the number of features is greater than the number of samples, which is the case in this dataset. SVMs tend to work well with clear margin of separation and they are effective in high dimensional spaces.
# SVMs do not perform well when a lot of 'noise' is present in the dataset.
# 
# Model performance was evaluated with cross-validation using the F1 score, as it is an appropriate evaluation metric for a classification task.
# This metric can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.  F1_micro calculates metrics globally by counting the total true positives, false negatives and false positives.
# 
# The model's best score is an F1 score of 0.72. Further performance improvements could be achieved by feature selection (e.g. through dropping variables with a very low correlation to the target) and by addressing collinearity between features. 

# In[ ]:



