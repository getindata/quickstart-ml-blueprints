#   Propensity model on Google Analytics 4 data with classification algorithms

## Overview

This GID ML Framework use case is supposed to serve as a basic example of a typical straightforward application of a ML model to some customer or user data that is gathered on a constant basis and collected as daily snapshot of user activity. This particular showcase features predicting the probability of a customer to add a product to a shopping cart during a session on the site and it is based on [Google Analytics](https://analytics.google.com/analytics/web/provision/#/provision) data. From the perspective of modeling approach this example can be easily translated to other problems, especially in **customer analytics area**, that involve estimating propensity to take some action based on the behavior, e.g.:

- churn prediction in telco,
- propensity-to-buy models in e-commerce,
- probability of default in banking,
- and many more.

This blueprint also shows example how to cover if your solution:
- data extraction with parametrized SQL queries and Kedro data catalog
- missing data imputation
- feature encoding
- automated hyperparameter optimization
- probability calibration
- selected model explanations with SHAP values

## Business context

Google Analytics is a popular web service used by companies to get insights about the traffic on their websites. It offers dashboarding capabilities out-of-the box which can be helpful to quickly get some insights, but much more can be done when we access the underlying data directly and apply advanced analytics, headed by machine learning, on our own.

The example presents using a sample data from the newest iteration of the tool (Google Analytics 4) to predict **the probability of adding an item to the shopping cart during an online user session based on the data gathered during that session**.

Full business potential of such model could be revealed in the online inference setup in which the batch-trained model would be served to compute predictions in real time. This way, some other mechanisms could developed on top of propensity prediction that would automatically take some actions (e.g. presenting some additional incentives to the user) during the ongoing session. The basic example that we provide includes only batch scoring on sessions data that were already collected in daily snapshots, so it doesn't support real time actions, however it is still valuable in many ways:

- it shows batch training and scoring workflows that can be easily translated to different business problems and datasets, especially in customer analytics area
- even when your ultimate goal is to deploy the model for online inference, you will still need to implement pipelines in batch version to be able to evaluate your model
- batch scoring is also the basis for explaining the model using eXplainable AI technique
- thanks to keeping solution modular transforming inference pipeline into online inference version is pretty straightforward if data engineering mechanisms are ready for this type of deployment

We plan to include an online inference demo on data streams as an extension of this use case in the future.

## Data

## Nodes and Pipelines
