import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
!pip install missingno
import missingno as msno
import gc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import math
import pickle
import os
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.utils import resample
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
import os
import os.path
import sqlite3
import flask

from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
from sqlalchemy import create_engine
from hcdr_model import initial_function_definition


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    grab_col_names(dataframe)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns




class initial_function_definition:
     def reduce_memory_usage(df):
  
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
        for col in df.columns:
            col_type = df[col].dtype
        
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
        return df
    
    def application_data(app_data):        
            app_data = app_data[app_data['CODE_GENDER'] != 'XNA']
            app_data['NAME_FAMILY_STATUS'].fillna("Not_Available", inplace=True)
            app_data['NAME_HOUSING_TYPE'].fillna("Not_Available", inplace=True)
            app_data['FLAG_MOBIL'].fillna("Not_Available", inplace=True)
            app_data['FLAG_EMP_PHONE'].fillna("Not_Available", inplace=True)
            app_data['FLAG_CONT_MOBILE'].fillna("Not_Available", inplace=True)
            app_data['FLAG_EMAIL'].fillna("Not_Available", inplace=True)
            app_data['OCCUPATION_TYPE'].fillna("Not_Available", inplace=True)
            app_data['DAYS_BIRTH'] = app_data['DAYS_BIRTH'].abs()/-365
            app_data['DAYS_EMPLOYED'].replace(365243,np.nan, inplace=True)
            app_data['INCOME_PER_PERSON'] = app_data["AMT_INCOME_TOTAL"] / app_data["CNT_FAM_MEMBERS']
            app_data["INCOME_PER_CHILD"] = app_data["AMT_INCOME_TOTAL"] / app_data["CNT_CHILDREN"]
            app_data["INCOME_PER_CHILD"] = app_data["AMT_INCOME_TOTAL"] / (1 + app_data["CNT_CHILDREN"])
            app_data['MaasVUrunFiyat'] = app_data['AMT_INCOME_TOTAL']/ app_data['AMT_GOODS_PRICE']
            app_data['MaasVsKrediTutari'] =app_data['AMT_INCOME_TOTAL']/ app_data['AMT_CREDIT']
            app_data['KrediTutariVsUrunFiyati'] =app_data['AMT_CREDIT'] - app_data['AMT_GOODS_PRICE']
            app_data['MusteriSkor'] = app_data['REGION_RATING_CLIENT']*app_data['REGION_RATING_CLIENT_W_CITY']
            app_data["EMPLOYED_BIRTH_DAYS"] = app_data["DAYS_EMPLOYED"] / app_data["DAYS_BIRTH"]
            app_data['NEW_SOURCES_PROD'] = app_data['EXT_SOURCE_1'] * app_data['EXT_SOURCE_2'] * app_data['EXT_SOURCE_3']
            app_data['NEW_EXT_SOURCES_MEAN'] = app_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
            app_data['NEW_SCORES_STD'] = app_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
            app_data["DAYS_LAST_PHONE_CHANGE"] = app_data["DAYS_LAST_PHONE_CHANGE"]/-1
            for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
                app_data[bin_feature], uniques = pd.factorize(app_data[bin_feature])
            app_data['Industry'] = app_data['ORGANIZATION_TYPE'].apply(lambda x: x if 'Industry' in x else 'NotIndustry')
            app_data['Business Entity'] = app_data['ORGANIZATION_TYPE'].apply(lambda x: x if 'Business Entity' in x else 'NotBsiness')
            app_data['Trade'] = app_data['ORGANIZATION_TYPE'].apply(lambda x: x if 'Trade' in x else 'NotTrade') 
            app_data['Transport'] = app_data['ORGANIZATION_TYPE'].apply(lambda x: x if 'Transport' in x else 'NotTransport')
            app_data["GOOD_EXIT_MEAN_DIVIDE"] = app_data["AMT_GOODS_PRICE"] / app_data["NEW_EXT_SOURCES_MEAN"]
            app_data["GOOD_EXIT_MEAN_CROSS"] = app_data["AMT_GOODS_PRICE"] * app_data["NEW_EXT_SOURCES_MEAN"]
            app_data["EXIT_BIRTH_DIVIDE"] = app_data["DAYS_BIRTH"] / app_data["EXT_SOURCE_3"]
            app_data["EXIT_BIRTH_CROS"] = app_data["DAYS_BIRTH"] * app_data["EXT_SOURCE_3"]
            app_data["GOOD_EXIT_MEAN_DIVIDE_BIRTH_CROS"] = app_data["GOOD_EXIT_MEAN_DIVIDE"] / app_data["EXT_SOURCE_3"]
            app_data["DAYS_EXT_DIVIDE"] = app_data["DAYS_BIRTH"] / app_data["NEW_EXT_SOURCES_MEAN"]
            app_data["DAYS_EXT_CROSS"] = app_data["DAYS_BIRTH"] * app_data["NEW_EXT_SOURCES_MEAN"]
            app_data["DAYS_SOURCE_2_DIVIDE"] = app_data["DAYS_ID_PUBLISH"] / app_data["EXT_SOURCE_2"]
            app_data["DAYS_SOURCE_2_CROSS"] = app_data["DAYS_ID_PUBLISH"] * app_data["EXT_SOURCE_2"]
            app_data["G_E_DIVIDE"] = app_data["GOOD_EXIT_MEAN_DIVIDE"] / app_data["EXIT_BIRTH_DIVIDE"]
            app_data["D_D_CROSS"] = app_data["DAYS_EXT_CROSS"] * app_data["DAYS_SOURCE_2_DIVIDE"]
            app_data["D_D_DIVIDE"] = app_data["DAYS_EXT_DIVIDE"] / app_data["DAYS_SOURCE_2_DIVIDE"]
            app_data["1"] = app_data["DAYS_EXT_DIVIDE"] / app_data["GOOD_EXIT_MEAN_DIVIDE"]
            app_data, cat_cols = one_hot_encoder(app_data, nan_as_category=False)
            gc.collect()
            return app_data
    def pos_cash_balance(pos):
        pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
        pos_agg = pos.groupby("SK_ID_CURR").agg({'MONTHS_BALANCE': ['max', 'mean', 'size'],
                                                  'SK_DPD': ['max', 'mean'],
                                                   'SK_DPD_DEF': ['max', 'mean']})
        pos_agg.columns = pd.Index(['POS' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

        # gerekli değişkenlerin olduğu dataframe pos_agg oldu. pos dataframe'ini çalışmadan çık
        del pos
        gc.collect()  
        return pos_agg
    def installments_payments(inst):
        cat_cols, num_cols, cat_but_car = grab_col_names(inst)
        inst, cat_cols = one_hot_encoder(inst, nan_as_category= True)
        inst["PAYMENT_RATE"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
        # taksit ödemesi - ödenen ücret
        inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

        # V_G vadesi geçen gün sayısı, V_O Son ödemesine kalan gün sayısı
        inst['V_G'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
        inst['V_O'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']

        # Negatif değer yerine optimize etmiş gibi sıfır koysak?
        inst['V_G'] = inst['V_G'].apply(lambda x: x if x > 0 else 0)
        inst['V_O'] = inst['V_O'].apply(lambda x: x if x > 0 else 0)

        #Her bir taksitte ödenen miktar
        inst["PER_PAYMENT_NUMBER"] = inst["AMT_PAYMENT"] / inst["NUM_INSTALMENT_NUMBER"]

        ### Taksit sayısıyla, vadesinden önce ödeme oranı arasında negatif bir korelasyon var.
        # Taksit sayısı arttıkça, vadesinden önce ödeme düşüyor.

            # Aynı işlemleri inst için de yaptım ve inst dataframe silebiliriz.
        inst_agg = inst.groupby("SK_ID_CURR").agg({'PER_PAYMENT_NUMBER': ['max', 'mean', 'size'],
                                                        'V_O': ['max', 'mean'],
                                                         'V_G': ['max', 'mean'],
                                                         'PAYMENT_DIFF': ['max', 'mean'],
                                                         'PAYMENT_RATE': ['max', 'mean'],
                                           })
        inst_agg.columns = pd.Index(['INST' + e[0] + "_" + e[1].upper() for e in inst_agg.columns.tolist()])

        del inst
        gc.collect()
        return inst_agg
    def credit_card_balance(cred):
        cred, cat_cols = one_hot_encoder(cred, nan_as_category= True)
        cred.drop(['SK_ID_PREV'], axis= 1, inplace = True)
        cred_agg = cred.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
        cred_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cred_agg.columns.tolist()])
        cred_agg['CC_COUNT'] = cred.groupby('SK_ID_CURR').size()
        del cred
        gc.collect()
        return cred_agg

                                                                                    
                                                                                    
apptrain = initial_function_definition.reduce_memory_usage(pd.read_csv("../input/home-credit-default-risk/application_train.csv"))
apptest = initial_function_definition.reduce_memory_usage(pd.read_csv("../input/home-credit-default-risk/application_test.csv"))
cred = initial_function_definition.reduce_memory_usage(pd.read_csv("../input/home-credit-default-risk/credit_card_balance.csv"))
pos = initial_function_definition.reduce_memory_usage(pd.read_csv("../input/home-credit-default-risk/POS_CASH_balance.csv"))
inst = initial_function_definition.reduce_memory_usage(pd.read_csv("../input/home-credit-default-risk/installments_payments.csv"))
                                                                                    

df_train = pd.read_pickle('')
df_test = test_data[df_train.columns]
df_test['SK_ID_CURR'] = test_data['SK_ID_CURR']
df_test['TARGET'] = np.nan
                                                                                    
                                                           
                                                                                  
                                                                                    
