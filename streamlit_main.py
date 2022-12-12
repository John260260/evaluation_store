import os.path

import boto3
import streamlit as st
import streamlit.components.v1 as components
import time,datetime,json
import warnings
warnings.filterwarnings("ignore")
from performance_drift_analysis import main
file = open("hydra.json")
hydra = json.load(file)
file.close()

s3 = boto3.client('s3',aws_access_key_id=st.secrets["aws_access_key_id"],aws_secret_access_key=st.secrets["aws_secret_access_key"],
                      region_name= st.secrets["region_name"])

## page config for streamlit
st.set_page_config(page_title="Performance Evaluation",page_icon="railofy.jpg",layout="wide")
hide_menu_style=""" 
<style>
# MainMenu {visibility:hidden; }
footer {visibility:hidden; }
</style>
"""
st.markdown(hide_menu_style,unsafe_allow_html=True)

##variables
mode= [1,2,3]
Quota_stauts= ["GN","PQ","RL","ALL_QUOTA"]
model_status= ["CNF","RAC"]
# dtd_available= ("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15")
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
prior_4days = today - datetime.timedelta(days=4)
class_available=["ALL_CLASS","1A","2A","2S","3A","SL"]

form_1=st.sidebar.form(key="Options")
form_1.header("Params")
prob_mode= form_1.radio("**A] Select the Prob-Mode:**", mode, index=0)
quota = form_1.radio("**B]  Quotas:**", Quota_stauts, index=0)
model = form_1.radio("**C]  Model:**", model_status, index=1)
start_date= form_1.date_input("**D] Select the start-date range of doj:**",(prior_4days))
end_date= form_1.date_input("**E] Select the end-date range of doj for analysis:**",(yesterday))
# end_date= start_date+ datetime.timedelta(days=3)
form_1.write('<style>div.row-widget.stRadio >div{flex-direction:row;}</style>', unsafe_allow_html=True)
# no_of_days = form_1.number_input("select no_of_days from 0-15", min_value=0.0, max_value=15.0, value=10.0, step=1.0)
classes= form_1.radio("**F] Classes:**", class_available)
form_1.markdown("""<style>div.stButton > button:first-child {background-color: rgb(238, 202, 202);}</style>""", unsafe_allow_html=True)
bl,sub_button=form_1.columns([1.3,1])
submitted=sub_button.form_submit_button("Submit")

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

if submitted:
    # try:
    start_time = time.time()
    head, img = st.columns([1,2])
    head.header(f"Mode-{prob_mode}")
    img.image("railofy.jpg", width=100)

    performance_dashboard_name = f"performance_evaluation/performance_dashboard_mode{prob_mode}_{model}_{quota}_{start_date}-{end_date}_{classes}.html"
    drift_dashboard_name = f"performance_evaluation/drift_analysis_dashboard_mode{prob_mode}_{model}_{quota}_{start_date}-{end_date}_{classes}.html"
    try:
        try:
        # test="https://railofy-datascience.s3.ap-south-1.amazonaws.com/performance_evaluation/test.html"
            performance_expander = st.expander("See the performance dashboard")
            with performance_expander:
                HtmlFile = s3.get_object(Bucket="railofy-datascience", Key=performance_dashboard_name)
                source_code = HtmlFile['Body'].read()
                components.html(source_code, height=1000, scrolling=True)

            dashboard_expander = st.expander("See the drift dashboard")
            with dashboard_expander:
                HtmlFile = s3.get_object(Bucket="railofy-datascience", Key=drift_dashboard_name)
                source_code = HtmlFile['Body'].read()
                components.html(source_code, height=1000, scrolling=True)
            # performance_dashboard_img=s3.get_object(Bucket='railofy-datascience', Key=performance_dashboard_name)
            # drift_dashboard_img=s3.get_object(Bucket='railofy-datascience', Key=drift_dashboard_name)

        except:
            main(quota=quota, model_type = model,start_date= start_date,end_date= end_date, prob_mode=prob_mode,classes=[classes],
                 performance_dashboard_name= performance_dashboard_name,drift_dashboard_name= drift_dashboard_name )
            performance_expander = st.expander("See the performance dashboard")
            with performance_expander:
                HtmlFile = s3.get_object(Bucket="railofy-datascience", Key=performance_dashboard_name)
                source_code = HtmlFile['Body'].read()
                components.html(source_code, height=1000, scrolling=True)

            dashboard_expander = st.expander("See the drift dashboard")
            with dashboard_expander:
                HtmlFile = s3.get_object(Bucket="railofy-datascience", Key=drift_dashboard_name)
                source_code = HtmlFile['Body'].read()
                components.html(source_code, height=1000, scrolling=True)
    except Exception as e:
        st.write(f" Selection of RAC Model along with PQ & RL Quota is Unavailable")
        st.write(e)
    time_taken = time.time() - start_time
    overall_time=convert(time_taken)
    st.write("Overall time taken: ", overall_time)
