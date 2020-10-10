#!/usr/bin/env python
# coding: utf-8

# In[23]:

import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

filename = "gs_drop_10_10_20.joblib"
best_model = joblib.load(filename)


def run():

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            input_df = data.replace([-99, -98, -1, -2, -3, -4, -5, -6, -7, -8, -9], np.nan)
            filter_input_df = input_df["InterviewType_07"] == 1
            input_df = input_df[filter_input_df]
            input_df = input_df.reset_index(drop = True)
            predictors =  input_df
            predictors =  input_df
            predictors = predictors[[
            'Quarter', 'Gender', 'RaceWhite', 'RaceBlack', 'Agegroup',
            'OverallHealth', 'CapableManagingHealthCareNeeds', 'HandlingDailyLife',
            'ControlLife', 'DealWithCrisis', 'GetsAlongWithFamily',
            'SocialSituations', 'FunctioningHousing', 'Symptoms', 'Nervous',
            'Hopeless', 'Restless', 'Depressed', 'EverythingEffort', 'Worthless',
            'PsychologicalEmotionalProblems', 'LifeQuality',
            'EnoughEnergyForEverydayLife', 'PerformDailyActivitiesSatisfaction',
            'HealthSatisfaction', 'RelationshipSatisfaction', 'SelfSatisfaction',
            'Tobacco_Use', 'Alcohol_Use', 'Cannabis_Use', 'ViolenceTrauma',
            'Housing', 'Education', 'Employment', 'EnoughMoneyForNeeds',
            'Friendships', 'EnjoyPeople', 'BelongInCommunity', 'SupportFromFamily',
            'SupportiveFamilyFriends', 'GenerallyAccomplishGoal', 'EverServed',  'ActiveDuty_Else', 'NightsHomeless']]
            aiety = np.where(input_df[["DiagnosisOne"]] == 62, 1, 0)
            aiety = pd.DataFrame(data = aiety, columns = ["aiety"])
            aiety = aiety.reset_index(drop = True)
            mdd_s = np.where(input_df[["DiagnosisOne"]] == 62, 1, 0)
            mdd_s = pd.DataFrame(data = mdd_s, columns = ["mdd_s"])
            mdd_s = mdd_s.reset_index(drop = True)
            drug_use = input_df["Cocaine_Use"] +input_df["Meth_Use"] + input_df["StreetOpioids_Use"] +  input_df["RxOpioids_Use"] + input_df["Stimulants_Use"]  + input_df["Inhalants_Use"] +  input_df["Sedatives_Use"] + input_df["Hallucinogens_Use"] + input_df["Other_Use"]
            drug_use = pd.DataFrame(data = drug_use, columns = ["drug_use"])
            drug_use = drug_use.reset_index(drop = True)
            er_hos_use_base = input_df["NightsDetox"] + input_df["NightsHospitalMHC"] + input_df["TimesER"]
            er_hos_use_base = pd.DataFrame(data = er_hos_use_base,columns = ["er_hos_use_base"])
            er_hos_use_base = er_hos_use_base.reset_index(drop = True)
            x = np.array([1])
            telehealth = np.repeat(x, [len(input_df)], axis=0)
            ## Index 
            telehealth = pd.DataFrame(data = telehealth, columns = ["telehealth"])
            telehealth = telehealth.reset_index(drop = True)
            x = np.array([1])
            grant = np.repeat(x, [len(input_df)], axis=0)
            ## Index 
            grant = pd.DataFrame(data = grant, columns = ["grant"])
            grant = grant.reset_index(drop = True)
            jail_arrest_base = input_df[["NumTimesArrested"]] + input_df[["NightsJail"]]
            jail_arrest_base = pd.DataFrame(data = jail_arrest_base, columns = ["jail_arrest_base"])
            jail_arrest_base = jail_arrest_base.reset_index(drop = True)
            def if_else(row):

                if row['DiagnosisOne'] == 59:
                    val= 1
                else:

                    val = 0

                return val
            mdd_r =  input_df.apply(if_else, axis=1)
            mdd_r = pd.DataFrame(data = mdd_r, columns = ["mdd_r"])
            mdd_r = mdd_r.reset_index(drop = True)

            def if_else(row):

                if row['SexualIdentity'] > 1:
                    val= 1
                else:

                    val = 0

                return val

            another_s_ident =  input_df.apply(if_else, axis=1)
            another_s_ident = pd.DataFrame(data = another_s_ident, columns = ["another_s_ident"])
            another_s_ident = another_s_ident.reset_index(drop = True)
            EverServed = predictors[["EverServed"]]
            ActiveDuty_Else = predictors[["ActiveDuty_Else"]]
            NightsHomeless = predictors[["NightsHomeless"]]
            predictors = predictors.drop(columns = ["EverServed", "ActiveDuty_Else", "NightsHomeless"])
            frames =  [predictors, telehealth, grant, EverServed, ActiveDuty_Else, NightsHomeless, aiety, mdd_r, mdd_s, another_s_ident, drug_use , er_hos_use_base,  jail_arrest_base]
            predictors_all = pd.concat(frames, axis = 1)
            predictors_all = predictors_all.to_numpy()
            prob_drop =  best_model.predict_proba(predictors_all)
            prob_drop =   pd.DataFrame(prob_drop[:,1], columns = ["prob_drop"])
            def if_else(row):

                if row['prob_drop'] > 0.29731086:

                    val = "very high risk"

                elif row['prob_drop'] > 0.29731086 / 2:

                    val = "high risk "
    
                elif row['prob_drop'] > 0.29731086 / 3:
        
                    val = "medium risk"
    
                else:

                    val = "low risk"

                return val

            prob_drop['risk_level'] = prob_drop.apply(if_else, axis=1)
            pred_dat = input_df.reset_index(drop = True)
            pred_dat_filter = pred_dat["InterviewType_07"] == 1
            pred_dat = pred_dat[pred_dat_filter]
            pred_dat = pred_dat.reset_index(drop = True)
            ConsumerID =  pred_dat.iloc[:,0]
            drop_out_risk_level = prob_drop.iloc[:,1]
            drop_out_risk_level = drop_out_risk_level.reset_index(drop = True)
            pred_dat = input_df.reset_index(drop = True)
            pred_dat_filter = pred_dat["InterviewType_07"] == 1
            pred_dat = pred_dat[pred_dat_filter]
            frames = [ConsumerID, drop_out_risk_level]
            pred_dat = pd.concat(frames, axis = 1)
            st.write(pred_dat)
st.set_option('deprecation.showfileUploaderEncoding', False)
if __name__ == '__main__':
    run()
