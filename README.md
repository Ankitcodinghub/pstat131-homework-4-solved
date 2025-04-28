# pstat131-homework-4-solved
**TO GET THIS SOLUTION VISIT:** [PSTAT131 Homework 4 Solved](https://www.ankitcodinghub.com/product/pstat-131-homework-4-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;119132&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;PSTAT131 Homework 4 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
For this assignment, we will continue working with part of a Kaggle data set that was the subject of a machine learning competition and is often used for practicing ML models. The goal is classification; specifically, to predict which passengers would survive the Titanic shipwreck.

Load the data from data/titanic.csv into R and familiarize yourself with the variables it contains using the codebook (data/titanic_codebook.txt).

Make sure you load the tidyverse and tidymodels!

Remember that you’ll need to set a seed at the beginning of the document to reproduce your results.

Create a recipe for this dataset identical to the recipe you used in Homework 3.

library(tidyverse) library(tidymodels) library(ISLR) # For the Smarket data set library(ISLR2) # For the Bikeshare data set library(discrim) library(poissonreg) library(corrr) library(klaR) # for naive bayes library(forcats) library(corrplot) library(pROC) library(recipes) library(rsample) library(parsnip) library(workflows) tidymodels_prefer()

titanic &lt;- read.csv(“titanic.csv”) %&gt;% mutate(survived = factor(survived, levels = c(“Yes”, “No”)),

pclass = factor(pclass))

Question 1

Split the data, stratifying on the outcome variable, survived. You should choose the proportions to split the data into. Verify that the training and testing data sets have the appropriate number of observations.

titanic_split &lt;- initial_split(titanic, prop = 0.80, strata = survived) titanic_train &lt;- training(titanic_split) titanic_test &lt;- testing(titanic_split) dim(titanic_train)

## [1] 712 12

dim(titanic_test)

## [1] 179 12

Question 2

Fold the training data. Use k-fold cross-validation, with k = 10.

titanic_folds &lt;- vfold_cv(titanic_train, v = 10)

Question 3

In your own words, explain what we are doing in Question 2. What is k-fold cross-validation? Why should we use it, rather than simply fitting and testing models on the entire training set? If we did use the entire training set, what resampling method would that be?

1. In question 2, we use k-fold to divide the training data into 10 groups.

2. k-fold cross-validation is an approach that involves randomly dividing the set of observations into k groups, or folds, of approximately equal size. The first fold is treated as a validation set, and the method is fit on the remaining k-1 folds.

3. We can avoid an over-fitting by using k-fold cross-validation.

4. Bootstrap.

Question 4

Set up workflows for 3 models:

1. A logistic regression with the glm engine;

2. A linear discriminant analysis with the MASS engine; 3. A quadratic discriminant analysis with the MASS engine.

How many models, total, across all folds, will you be fitting to the data? To answer, think about how many folds there are, and how many models you’ll fit to each fold.

# set up recipe titanic_recipe &lt;- titanic_train %&gt;%

recipe(survived ~ pclass + sex + age + sib_sp + parch + fare) %&gt;% step_impute_linear(age) %&gt;% step_dummy(all_nominal_predictors()) %&gt;% step_interact(terms = ~ starts_with(“sex”):fare + age:fare)

• A logistic regression with the glm engine:

log_reg &lt;- logistic_reg() %&gt;%

set_mode(“classification”) %&gt;% set_engine(“glm”)

log_wkflow &lt;- workflow() %&gt;%

add_recipe(titanic_recipe) %&gt;% add_model(log_reg)

• A linear discriminant analysis with the MASS engine:

lda_mod &lt;- discrim_linear() %&gt;%

set_mode(“classification”) %&gt;% set_engine(“MASS”)

lda_wkflow &lt;- workflow() %&gt;% add_model(lda_mod) %&gt;% add_recipe(titanic_recipe)

• A quadratic discriminant analysis with the MASS engine:

qda_mod &lt;- discrim_quad() %&gt;%

set_mode(“classification”) %&gt;% set_engine(“MASS”)

qda_wkflow &lt;- workflow() %&gt;% add_model(qda_mod) %&gt;% add_recipe(titanic_recipe)

Since there are 3 different types of model and 10 folds for each type, we will fit 30 models in total to the data.

Question 5

Fit each of the models created in Question 4 to the folded data.

log_fit &lt;- fit_resamples(log_wkflow, titanic_folds) lda_fit &lt;- fit_resamples(lda_wkflow, titanic_folds) qda_fit &lt;- fit_resamples(qda_wkflow, titanic_folds)

Question 6

Use collect_metrics() to print the mean and standard errors of the performance metric accuracy across all folds for each of the four models.

Decide which of the 3 fitted models has performed the best. Explain why. (Note: You should consider both the mean accuracy and its standard error.)

collect_metrics(log_fit)

## # A tibble: 2 x 6

## .metric .estimator mean n std_err .config

## &lt;chr&gt; &lt;chr&gt; &lt;dbl&gt; &lt;int&gt; &lt;dbl&gt; &lt;chr&gt;

## 1 accuracy binary 0.808 10 0.0190 Preprocessor1_Model1

## 2 roc_auc binary 0.849 10 0.0164 Preprocessor1_Model1

collect_metrics(lda_fit)

## # A tibble: 2 x 6

## .metric .estimator mean n std_err .config

## &lt;chr&gt; &lt;chr&gt; &lt;dbl&gt; &lt;int&gt; &lt;dbl&gt; &lt;chr&gt;

## 1 accuracy binary 0.796 10 0.0225 Preprocessor1_Model1

## 2 roc_auc binary 0.851 10 0.0168 Preprocessor1_Model1

collect_metrics(qda_fit)

## # A tibble: 2 x 6

## .metric .estimator mean n std_err .config

## &lt;chr&gt; &lt;chr&gt; &lt;dbl&gt; &lt;int&gt; &lt;dbl&gt; &lt;chr&gt;

## 1 accuracy binary 0.768 10 0.0158 Preprocessor1_Model1

## 2 roc_auc binary 0.839 10 0.0175 Preprocessor1_Model1

The logistic model performs the best since it have highest mean and relatively low ROC.

Question 7

Now that you’ve chosen a model, fit your chosen model to the entire training dataset (not to the folds).

log_fit_7 &lt;- fit(log_wkflow, titanic_train)

Question 8

Finally, with your fitted model, use predict(), bind_cols(), and accuracy() to assess your model’s performance on the testing data!

Compare your model’s testing accuracy to its average accuracy across folds. Describe what you see.

prediction &lt;- predict(log_fit_7, new_data = titanic_test, type = “class”) %&gt;%

bind_cols(titanic_test %&gt;% select(survived)) %&gt;% accuracy(truth = survived, estimate = .pred_class)

prediction

## # A tibble: 1 x 3

## .metric .estimator .estimate

## &lt;chr&gt; &lt;chr&gt; &lt;dbl&gt;

## 1 accuracy binary 0.804

collect_metrics(log_fit)

## # A tibble: 2 x 6

## .metric .estimator mean n std_err .config

## &lt;chr&gt; &lt;chr&gt; &lt;dbl&gt; &lt;int&gt; &lt;dbl&gt; &lt;chr&gt;

## 1 accuracy binary 0.808 10 0.0190 Preprocessor1_Model1

## 2 roc_auc binary 0.849 10 0.0164 Preprocessor1_Model1

The testing accuracy of the model is 0.804.

The Average accuracy across folds is 0.808.

We can see that the model’s testing accuracy is slightly smaller than the model’s average accuracy across folds.
