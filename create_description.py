import math
import pandas as pd

def create_description(input_frame):
    description=[]
    for i in range(len(input_frame)):
        row=input_frame.iloc[i]
        if row["anatom_site_general_challenge"]!=0:
            site=row["anatom_site_general_challenge"]
        else:
            site=None
        if int(row["age_approx"])!=0:
            age=int(row["age_approx"])
        else:
            age=None
        if age is not None and site is not None:
            data="The patient is a " + row['sex'] + " approximately " + str(int(row['age_approx'])) + " years old." + " They have a lession on their " + row["anatom_site_general_challenge"]+"."
        if age is None and site is not None:
            data="The patient is a " + row['sex'] + "." + " They have a lession on their " + row["anatom_site_general_challenge"]+"."
        if site is None and age is not None:
            data="The patient is a " + row['sex'] + " approximately " + str(int(row['age_approx'])) + " years old."    
        description.append(data) 
    return description

test = pd.read_csv('/scratch/ssc10020/IndependentStudy/SLIP/dataset/ISIC/test_data_old.csv')
test["age_approx"] = test["age_approx"].fillna(0)
test["anatom_site_general_challenge"] = test["anatom_site_general_challenge"].fillna(0)
print('Before adding description:')
print(test.head())
test['description'] = create_description(test)
print('After adding description:')
print(test.head())
test.to_csv('/scratch/ssc10020/IndependentStudy/SLIP/dataset/ISIC/test_data.csv')