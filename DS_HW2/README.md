# DS_HW2

| show_id | type  | title                | director        | cast | country       | date_added         | release_year | rating | duration | listed_in     | description                                                                                                                                              |
| ------- | ----- | -------------------- | --------------- | ---- | ------------- | ------------------ | ------------ | ------ | -------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| s1      | Movie | Dick Johnson Is Dead | Kirsten Johnson |      | United States | September 25, 2021 | 2020         | PG-13  | 90 min   | Documentaries | (a long message)|

## why i can't use regression on DS_HW1 data :

Regression is used to predict continuous numeric values.  
This dataset mostly contains text and categorical features like title, director, and country.  
There is no strong continuous target variable for prediction.  
Columns such as description and cast cannot be directly used in regression models.  
Many features also have high-cardinality values, which makes modeling difficult.  
The dataset is better suited for classification, clustering, or recommendation tasks.  
Regression would only work after heavy preprocessing and feature engineering.

## New dataset

So i've used data/Teen_Mental_Health_Dataset.csv dataset 
you can see it's details with df.head()

# questions :
the questions are answered in the pdf file 


