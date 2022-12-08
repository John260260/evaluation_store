import pandas as pd
import numpy as np
import boto3, pickle, time, datetime, warnings, json

# Streamlit
# import streamlit.components.v1 as components
import streamlit as st

# ElasticSearch
from elasticsearch import Elasticsearch

# Evidently
import evidently
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ProbClassificationPerformanceTab

warnings.filterwarnings("ignore")

es = Elasticsearch(st.secrets["link"], http_auth=(st.secrets["els_user"], st.secrets["els_pass"]))
file = open("hydra.json")
hydra = json.load(file)
file.close()

performance_widgets= hydra["performance_widgets"]#'Current: Precision-Recall Curve','Reference: Precision-Recall Table', 'Current: Precision-Recall Table','Classification Quality By Feature']
target_index_name = hydra["target_index_name"]
indicator_index_name = hydra["indicator_index_name"]
rev_class_dict = hydra["rev_class_dict"]
class_dict = hydra["class_dict"]

rev_class_dict_ = {v: k for k, v in class_dict.items()}
class_dict_= {v: k for k, v in rev_class_dict.items()}

def get_features(prob_mode:int):
    numerical_features = hydra[f"mode{prob_mode}_numerical_features"]
    categorical_features = hydra[f"mode{prob_mode}_categorical_features"]
    features = hydra[f"mode{prob_mode}_features"]
    target_name = hydra["target_name"]
    predicted_classes = hydra["predicted_classes"]
    parameters = features + [target_name] + predicted_classes
    version = hydra[f"mode{prob_mode}_version"]
    return numerical_features, categorical_features, target_name, predicted_classes, features, parameters, version

def get_pricing_data(start_date, end_date,quota:str,prob_mode:int, classes:list=None):
    pricing_query = [
           {"term": {'@log_name': "datascience.pricing"}},
           {"terms": {"type":["model_indicators"]}},
           {"terms": {"prob_mode":[str(prob_mode)]}},
           {"range":
                {
                    "doj": {
                        "gte": start_date,
                        "lte": end_date
                      }
                }
           },
           {"range":
                {
                    "pnr" : {
                        "gt" : "10"
                    }
               }
           }
        ]
    if "ALL_CLASS" not in classes:
        # print(classes, [class_dict[class_] for class_ in classes])
        pricing_query.append({"terms": {"class":[class_dict[class_] for class_ in classes]}})
    result_ = es.search(index=indicator_index_name, query={"bool": { "must": pricing_query }}, size= 10000)
    result = pd.DataFrame([res['_source'] for res in result_['hits']['hits']])
    result['input_data'] = result['input_data'].apply(lambda x : eval(x))
    result['indicators'] = result['indicators'].apply(lambda x : eval(x))
    input_data = pd.DataFrame(list(result['input_data'].values))
    common_columns = list(set(result.columns).intersection(set(input_data.columns)))
    result = pd.concat([result,input_data.drop(labels=common_columns, axis=1)], axis=1)
    indicator_data = pd.DataFrame(list(result['indicators'].values))
    common_columns = list(set(result.columns).intersection(set(indicator_data.columns)))
    result = pd.concat([result,indicator_data.drop(labels=common_columns, axis=1)], axis=1)
    result = result.sort_values(by=["pnr","dtd","@timestamp"], ascending=False).drop_duplicates(subset=["pnr","dtd"]).sort_index().reset_index(drop=True)
    if 'ALL_QUOTA' not in quota:
        result = result[result['quota']==quota].reset_index(drop=True)
    return result

def get_pnr_chart_status(start_date, end_date, pnrs:list, ins_type:int):
    target_query = [
               {"term": {'@meta.appName': "RailofyBackend"}},
               {"term": {'@logLevel': "info"}},
               {"terms": {"indexedKeys.module":["Pricing V2"]}},
               {"terms": {"indexedKeys.pnr":pnrs}},
               {"range":
                    {
                        "@timestamp": {
                            "gte": start_date,
                            "lte": end_date
                          }
                    }
               }
    ]

    # st.write(str(target_query))
    es = Elasticsearch(st.secrets["link"], http_auth=(st.secrets["els_user"], st.secrets["els_pass"]))
    result_ = es.search(index=target_index_name, query = {"bool": { "must": target_query }},size= 10000)
    # st.write(str({"bool": { "must": target_query }}))
    result2 = pd.DataFrame([{"pnr":res['_source']['indexedKeys']['pnr'],
                "doj":res['_source']['indexedKeys']['doj'],
                "reject_reason":res['_source']['data']['reject_reason'],
               "reject_message":res['_source']['data']['reject_message'],
                "ins_type":res['_source']['data']['ins_type'],
                "dtd":res['_source']['data']['dtd'],
               "cnf_prob":res['_source']['data']['cnf_prob'],
               "rac_prob":res['_source']['data']['rac_prob'],
               "current_status" : res['_source']['data']['pnr_details'][0]['current_status'],
               "timestamp": res['_source']['@timestamp']} for res in result_['hits']['hits']])
    result2 = result2[result2['ins_type']==ins_type].reset_index(drop=True)
    result2 = result2.sort_values(by=['pnr','timestamp'], ascending=False).drop_duplicates(subset=['pnr']).sort_index().reset_index(drop=True)
    result2['pnr'] = result2['pnr'].astype(str)
    return result2

def fetch_es_data(start_date, end_date, quota:str, prob_mode:int, model_type:str, classes:list , chunksize=50):
    data = get_pricing_data(start_date=start_date, end_date=end_date, quota=quota, prob_mode=prob_mode, classes=classes)
    pnrs = list(data["pnr"].unique())
    chart_status = pd.DataFrame()
    if model_type == "CNF":
        ins_type = 1
    if model_type == "RAC":
        ins_type = 0

    if len(pnrs) > chunksize:
        for index in range(0, len(pnrs), chunksize):
            chart_status = chart_status.append(
                get_pnr_chart_status(start_date=start_date, end_date=end_date, pnrs=pnrs[index:index + chunksize],
                                     ins_type=ins_type))
    else:
        chart_status = get_pnr_chart_status(start_date=start_date, end_date=end_date, pnrs=pnrs, ins_type=ins_type)
    data = data.merge(chart_status, on=['pnr'], suffixes=("", "_final"), how="left")
    data = data[data['timestamp'].notna()].reset_index(drop=True)
    data = data.rename(columns={'current_status_final': 'chartStatus'})
    data['dtd'] = data['dtd'].astype(int)
    if model_type.upper() == "RAC":
        data['target'] = np.where(data['chartStatus'].isin(['RAC', 'CNF']), 'class1', 'class0')
        #         data['predicted'] = np.where(data['rac_prob']>=50, 'class1', 'class0')
        data['class1'] = data['rac_prob'] / 100
        data['class0'] = 1 - data['class1']
    if model_type.upper() == "CNF":
        data['target'] = np.where(data['chartStatus'].isin(['CNF']), 'class1', 'class0')
        #         data['predicted'] = np.where(data['cnf_prob']>=50, 'class1', 'class0')
        data['class1'] = data['cnf_prob'] / 100
        data['class0'] = 1 - data['class1']
    return data

def correct_chartstatus(data:pd.DataFrame):
    chart_status = hydra["chart_status"]
    data = data.replace(chart_status)
    return data

def get_current_data(start_date, end_date, quota:str, prob_mode: int, model_type : str, classes:list):
    data = fetch_es_data(start_date=start_date, end_date=end_date, quota=quota, prob_mode=prob_mode, model_type=model_type,
                         classes=classes)
    data = correct_chartstatus(data)
    data = data[(data['chartStatus'] == 'WL') | (data['chartStatus'] == 'RAC') | (data['chartStatus'] == 'CNF')]
    return data

def fetch_reference_data(prob_mode: int,version: str,quota: str,model_type: str,classes: list):
    s3 = boto3.client('s3',aws_access_key_id=st.secrets["aws_access_key_id"],aws_secret_access_key=st.secrets["aws_secret_access_key"],
                      region_name= st.secrets["region_name"])
    train_file = f"data/mode{prob_mode}{version}{quota}data/{model_type.lower()}_model_train_set.csv"
    test_file = f"data/mode{prob_mode}{version}{quota}data/{model_type.lower()}_model_test_set.csv"
    aws_bucket = hydra["aws_bucket"]
    train_data = s3.get_object(Bucket=aws_bucket, Key=train_file)
    test_data = s3.get_object(Bucket=aws_bucket, Key=test_file)
    train_data_df = pd.read_csv(train_data['Body'])
    test_data_df = pd.read_csv(test_data['Body'])
    del train_data, test_data
    data = pd.concat([train_data_df, test_data_df], ignore_index=True)
    del train_data_df, test_data_df
    if "ALL_CLASS" not in classes:
        data['class'] = data['class'].map(rev_class_dict_)
        data = data[data['class'].isin(classes)]
        data['class'] = data['class'].map(class_dict_)
    # st.write(f"Current Sub-Selection of Classes are {classes}")
    data['quota']=data['quota'].str[:2]
    return data

def get_model(model_type:str, prob_mode:int, version:str, quota:str):
    s3 = boto3.client('s3',aws_access_key_id=st.secrets["aws_access_key_id"],aws_secret_access_key=st.secrets["aws_secret_access_key"],
                      region_name= st.secrets["region_name"])
    model_path = f"pnr_only_pricing_daily/models/{model_type.lower()}_prob_model_mode{prob_mode}{version}-1_{quota.lower()}.pkl"
    print(model_path)
    model_data = s3.get_object(Bucket=hydra["aws_bucket"], Key=model_path)
    model_body = model_data['Body'].read()
    model = pickle.loads(model_body)
    return model

def get_prediction(model, data:pd.DataFrame, parameters : list ,features :list):
    print(features)
    # print(parameters)
    data[['class0', 'class1']] = model.predict_proba(data[features])
    for col in data.columns:
        if "target" in col:
            data = data.rename(columns={col: 'target'})
            break
    performance_data = data[parameters].dropna()
    drift_data = data[features].dropna()
    return performance_data, drift_data

def type_conversions(df_1, df_2, df_3, categorical_features:list, numerical_features:list, predicted_classes:list):
    df_1[categorical_features] = df_1[categorical_features].astype('int32')
    df_1[numerical_features] = df_1[numerical_features].astype(float)
    df_1[predicted_classes] = df_1[predicted_classes].astype(float)
    df_2[categorical_features] = df_2[categorical_features].astype('int32')
    df_2[numerical_features] = df_2[numerical_features].astype(float)
    df_3[categorical_features] = df_3[categorical_features].astype('int32')
    df_3[numerical_features] = df_3[numerical_features].astype(float)
    df_3[predicted_classes] = df_3[predicted_classes].astype(float)
    return df_1, df_2, df_3

def s3_put_object(html : str,filename:str):
    s3 = boto3.client('s3',aws_access_key_id=st.secrets["aws_access_key_id"],aws_secret_access_key=st.secrets["aws_secret_access_key"],
                      region_name= st.secrets["region_name"])
    s3.put_object(Body=html, Bucket='railofy-datascience', Key=filename, ContentType='text/html')

def PerformanceDashBoard(categorical_features: list, numerical_features: list,
                         prediction_classes: list, target_name: str,
                         reference_data : pd.DataFrame(), current_data : pd.DataFrame(), save=True,
                         file_name="Performance_Analysis.html"):
    column_mapping = ColumnMapping()
    column_mapping.target = target_name
    column_mapping.prediction = prediction_classes
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features
    prob_perform_tab = Dashboard(tabs=[ProbClassificationPerformanceTab(verbose_level=0, include_widgets=performance_widgets)])
    prob_perform_tab.calculate(reference_data=reference_data, current_data=current_data,
                               column_mapping=column_mapping)
    if save:
        html = prob_perform_tab.html()
        s3_put_object(html=html,filename=file_name)
        print(f"HTML Saved to : {file_name}")
        return file_name
    else:
        return prob_perform_tab


def DriftAnalysisDashBoard(categorical_features: list, numerical_features: list,
                           reference_data: pd.DataFrame(), current_data: pd.DataFrame(), save=True,
                           file_name="drift_analysis.html"):
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features
    data_drift_dashboard = Dashboard(tabs=[DataDriftTab(verbose_level=0)])
    data_drift_dashboard.calculate(reference_data=reference_data,
                                   current_data=current_data,
                                   column_mapping=column_mapping)
    if save:
        html = data_drift_dashboard.html()
        s3_put_object(html=html,filename=file_name)
        print(f"HTML Saved to : {file_name}")
        return file_name
    else:
        return data_drift_dashboard

def main(start_date : str,end_date : str, quota : str ,model_type:str,prob_mode : int, classes:list, performance_dashboard_name:str, drift_dashboard_name:str):
    numerical_features, categorical_features, target_name, predicted_classes, features, parameters, version = get_features(prob_mode=prob_mode)
    current_data = get_current_data(quota= quota, model_type=model_type, prob_mode=prob_mode,start_date=start_date,end_date=end_date, classes=classes)
    current_data = current_data[parameters]
    if "ALL_QUOTA" in quota:
        ref_data = pd.DataFrame()
        if model_type=="CNF":
            quotas=["GN","PQ","RL"]
        elif model_type=="RAC":
            quotas=["GN"]
        for quota_ in quotas:
            ref_data = pd.concat([ref_data,fetch_reference_data(prob_mode=prob_mode,version=version,quota= quota_,
                                                                model_type=model_type, classes=classes)])
    else:
        ref_data = fetch_reference_data(prob_mode=prob_mode,version=version,quota= quota, model_type=model_type,
                                        classes=classes)
    if "ALL_QUOTA" in quota:
        performance_ref_data, drift_ref_data = pd.DataFrame(), pd.DataFrame()
        if model_type=="CNF":
            quotas=["GN","PQ","RL"]
        else:
            quotas=["GN"]
        for quota_ in quotas:
            model = get_model(model_type=model_type, prob_mode=prob_mode, version=version, quota=quota_)

            performance_ref_data_temp, drift_ref_data_temp = get_prediction(data=ref_data[ref_data['quota']==quota_],
                                                                            parameters = parameters,
                                                                            features = features,
                                                                            model = model)
            performance_ref_data = pd.concat([performance_ref_data, performance_ref_data_temp])
            drift_ref_data = pd.concat([drift_ref_data, drift_ref_data_temp])
    else:
        model = get_model(model_type=model_type, prob_mode=prob_mode, version=version, quota=quota)
        performance_ref_data, drift_ref_data = get_prediction(model=model, parameters=parameters, features=features, data=ref_data)
    performance_ref_data, drift_ref_data, current_data = type_conversions(df_1=performance_ref_data, df_2=drift_ref_data, df_3=current_data,
                                                                          categorical_features=categorical_features, numerical_features=numerical_features,
                                                                          predicted_classes=predicted_classes)
    # performance_dashboard_name = f"evidently/performance_dashboard_mode{prob_mode}{version}{quota}.html"
    # drift_dashboard_name = f"evidently/drift_analysis_dashboard_mode{prob_mode}{version}{quota}.html"
    performance_dashboard = PerformanceDashBoard(categorical_features=categorical_features, numerical_features=numerical_features,
        prediction_classes=predicted_classes,target_name=target_name,current_data=current_data[parameters],
        reference_data=performance_ref_data[parameters], file_name=performance_dashboard_name)
    drift_analysis_dashboard= DriftAnalysisDashBoard(categorical_features=categorical_features,numerical_features=numerical_features,
        current_data=current_data[features],reference_data=drift_ref_data[features],file_name=drift_dashboard_name)
    # return performance_dashboard_name,drift_dashboard_name

if __name__ == "__main__":
    start_date = '2022-11-19'
    end_date = '2022-11-28'
    quota = "GN"
    model = "RAC"
    prob_mode = 3
    classes = ["1A"]
    performance_dashboard_name = f"performance_evaluation/performance_dashboard_mode{prob_mode}_{model}_{quota}_{start_date}-{end_date}_{classes}.html"
    drift_dashboard_name = f"performance_evaluation/drift_analysis_dashboard_mode{prob_mode}_{model}_{quota}_{start_date}-{end_date}_{classes}.html"
    main(start_date='2022-11-19', end_date='2022-11-28', quota="GN", model_type="RAC", prob_mode=3, classes=["1A"],performance_dashboard_name= performance_dashboard_name,drift_dashboard_name= drift_dashboard_name)