import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import joblib
import re
from nltk.corpus import stopwords 
import string 
from nltk.stem import WordNetLemmatizer


################################################################################################
class TextPreprocessing() :
    def _init_(self, df : pd.DataFrame = pd.DataFrame) :
        self.df = df
        
        
    def Clean(self , df) :
        self.df = df
        df_copy = df.copy(deep = True)
        text_cols = list(df_copy.select_dtypes(include = ['object']).columns)
        for col in text_cols :
            for idx, text in enumerate(df_copy[col]) :
                te = []
                word = re.sub(r'(@|#)\w+' , '' , text)
                word = re.sub("[,.]", "", word)
                word = re.sub(r'https?://\S+' , '' , word)
                word = re.sub(r'(\?|!)+' , '' , word)
                word = re.sub(r"\(|\)", "", word)
                word = re.sub(r'(^\s+)' , '' , word)
                word = re.sub(r'(\s+$)' , '' , word)
                word = re.sub(r'\d+' , '' , word)
                word = word.split()
                for i in word :
                    if (i not in s_words) & (i not in punc) :
                        i = i.lower()
                        i = lmt.lemmatize(i , 'v')
                        te.append(i)
                df_copy.at[idx , col] = te
        return df_copy
    
    def Vactorize (self, df , target_name) :
        self.df = df
        self.target_name = target_name
        df_cleaned = df.copy(deep = True)
        text_cols = list(df_cleaned.select_dtypes(include = ['object']).columns)
        pos_word = {}
        neg_word = {}

        pos_df = df_cleaned[df_cleaned[target_name] == 1].reset_index(drop=True)
        neg_df = df_cleaned[df_cleaned[target_name] == 0].reset_index(drop=True)
        
        for col in text_cols :

            pos_word[col] = [word for sublist in pos_df[col] for word in sublist]
            neg_word[col] = [word for sublist in neg_df[col] for word in sublist]



        pos_freq = {}
        neg_freq = {}
        for key in pos_word.keys() :
            positive_dict = {}
            for word in pos_word[key] :
                positive_dict[word] = positive_dict.get(word , 0) + 1

            pos_freq[key] = positive_dict


        for key in neg_word.keys() :
            negative_dict = {}
            for word in neg_word[key] :
                negative_dict[word] = negative_dict.get(word , 0) + 1

            neg_freq[key] = negative_dict
            
        return pos_freq , neg_freq

        
        
        
    def Vactorization (self , df , target_name) :
        self.df = df
        self.target_name = target_name
        df_cleaned = df.copy(deep = True)
        text_cols = list(df_cleaned.select_dtypes(include = ['object']).columns)
        pos_freq , neg_freq = TextPreprocessing().Vactorize(df_cleaned , target_name)
        for col in text_cols :
            df_cleaned['{}_pos'.format(col)] = 0
            df_cleaned['{}_neg'.format(col)] = 0
            for idx, List in enumerate(df_cleaned[col]) :
                pos_frequent = 0
                neg_frequent = 0
                for word in List :
                    pos_frequent += pos_freq[col].get(word , 0)
                    neg_frequent += neg_freq[col].get(word , 0)



                df_cleaned.at[idx ,'{}_pos'.format(col)] = pos_frequent
                df_cleaned.at[idx ,'{}_neg'.format(col)] = neg_frequent
            df_cleaned.drop([col] , axis = 1 , inplace = True)
        return df_cleaned
    
    
    def fit_transform(self , df , target_name) :
        self.df = df
        self.target_name = target_name
        
        df_cleaned = TextPreprocessing().Clean(df)
        df_vact = TextPreprocessing().Vactorization(df_cleaned , target_name)
        
        return df_vact
    
    
    def Naive_Bayes(self , df, target_name) :
        self.df = df
        self.target_name = target_name
        df_naive = TextPreprocessing().Clean(df)
        pos_freq , neg_freq = TextPreprocessing().Vactorize(df_naive , target_name)
        text_cols = list(df_naive.select_dtypes(include = ['object']).columns)
        
        v_n_pos , v_n_neg = {} , {}

        for key in pos_freq.keys() :
            v_n_pos[key] = len(pos_freq[key])
            n = 0
            for word in pos_freq[key] :
                n += pos_freq[key].get(word , 0)
            v_n_pos[key]+=n
        for key in neg_freq.keys() :
            v_n_neg[key] = len(neg_freq[key])
            n = 0
            for word in neg_freq[key] :
                n += neg_freq[key].get(word , 0)
            v_n_neg[key]+=n
            
            
        prob_pos_dict = {}
        for key in pos_freq.keys():
            positive_dict = {}
            for word in pos_freq[key] :
                positive_dict[word] = (pos_freq[key].get(word , 0) + 1) / (v_n_pos[key])
            prob_pos_dict[key] = positive_dict




        prob_neg_dict = {}
        for key in neg_freq.keys():
            negative_dict = {}
            for word in neg_freq[key] :
                negative_dict[word] = (neg_freq[key].get(word , 0) + 1) / (v_n_neg[key])
                prob_neg_dict[key] = negative_dict
            
            
            
        for col in text_cols :
            df_naive['{}_probs'.format(col)] = 0
            for idx, List in enumerate(df_naive[col]) :
                score = 0
                for word in List :
                    try :
                        b = np.log((prob_pos_dict[col].get(word , 0)) / (prob_neg_dict[col].get(word , 0)))
                        if b == -float('inf') :
                            pass
                        else :
                            score +=b
                    except :
                        pass

                df_naive['{}_probs'.format(col)][idx] = score
            df_naive.drop([col] , axis = 1 , inplace = True)
            
            
        return {'probs_pos':prob_pos_dict ,'probs_neg':prob_neg_dict } , df_naive   
############################################################################################################################
pre = TextPreprocessing()


# Models
loaded_model = joblib.load(open("RF_model_CCS", 'rb'))
encoder = joblib.load(open("encoder", 'rb'))

# Title
st.title("Job Posting Fake Detection")

# Form Design
with st.form(key="form1"):
    left_column,right_column,midle_column = st.columns(3)

    with left_column:
        job_title=st.text_input(label="Job Title")
        salary=st.text_input(label="Salary")
        job_requirements=st.text_input(label="job requirements")
        function=st.text_input(label="Function")
        optionsset3=['Unspecified', "Bachelor's Degree", 'High School or equivalent',
       'Professional', "Master's Degree", 'Vocational',
       'Associate Degree', 'Some College Coursework Completed',
       'Vocational - Degree', 'Certification',
       'Some High School Coursework', 'Vocational - HS Diploma',
       'Doctorate']
        education_required = st.selectbox('Education required:', optionsset3)

    with midle_column:
        department=st.text_input(label="Department")
        company_information=st.text_input(label="company information")
        benefits=st.text_input(label="Benefits")
        questions = st.checkbox('Has Questions')
        telecom = st.checkbox('telecommuting')
        logo = st.checkbox('Logo exist')

    with right_column:
        location=st.text_input(label="office location")
        job_description    =st.text_input(label="job description    ")
        industry=st.text_input(label="industry")
        # Create a combo box
        optionsset1= ['Temporary', 'Part-time','Full-time','Contract','Other']
        employment_type = st.selectbox('Employment type:', optionsset1)

        optionsset2= [ 'Not Applicable','Internship', 'Entry level','Mid-Senior level','Associate', 'Executive', 'Director']
        experience_required = st.selectbox('Experience required:', optionsset2)
  
    submit=st.form_submit_button(label="Check")



# Text
text = "job title: " + job_title + " - the location of the company is " + location + " - the depatment you will work at is " + department + " - the salary offered is " + salary + " - and here is the company information: " + company_information + " - the job description: " + job_description + " - and here are the requirments of the job: " + job_requirements + " - the benfits you will gain: " + benefits + " - industry: " + industry + " - you funtion: " + function

# education_required
# education_required_num = encoder.transform(education_required)

if (employment_type == "Temporary"):
    employment_type_num = 1
elif(employment_type =="Part-time"):
    employment_type_num = 2
elif(employment_type =="Full-time"):
    employment_type_num = 3
elif(employment_type =="Contract"):
    employment_type_num = 4
elif(employment_type =="Other"):
    employment_type_num = 0
    
    
# experience_required
if (experience_required == "Not Applicable"):
    experience_required_num = 0
elif (experience_required == "Internship"):
    experience_required_num = 1
elif (experience_required == "Entry level"):
    experience_required_num = 2
elif (experience_required == "Mid-Senior level"):
    experience_required_num = 3
elif (experience_required == "Associate"):
    experience_required_num = 4
elif (experience_required == "Executive"):
    experience_required_num = 5
elif (experience_required == "Director"):
    experience_required_num = 6

# Question  
if questions :
    questions_num =1
else:
    questions_num =0

# telecom
if telecom :
    telecom_num =1
else:
    telecom_num =0

# Logo
if logo :
    logo_num =1
else:
    logo_num =0

    


if(submit):
    test_df = pd.read_csv('test.csv')
    new_row = pd.DataFrame({'telecommuting':telecom_num , 'company logo exist?':logo_num , 'has_questions':questions_num , 'employment_type':employment_type_num , 'experience required':experience_required_num , 'education required': 1,"fake?":1,'Job Info.':text }, index=[0])
    test_df = pd.concat([new_row,test_df.loc[:]]).reset_index(drop=True)

    df_cleaned = pre.Vactorization(test_df , target_name = 'fake?')


    X = df_cleaned.drop(['fake?'] , axis = 1)
    x = X.iloc[0].values.reshape(1,-1)
    y_pred = loaded_model.predict(x)
    output="...."
    if y_pred[0]==1:
        output='Fake'
    elif y_pred[0]==0:
        output='Real'
    st.subheader(output, anchor=None)


