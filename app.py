import streamlit as st
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

import zipfile
import uuid
import io
import os

st.markdown('''
            <style> .font {
            font-size: 45px;
            color: #008000;
            text-align: left;}
            </style>
            ''', unsafe_allow_html=True)

st.sidebar.markdown(
    '<p class="font">CSV-file Code Predictor</p>', unsafe_allow_html=True)

with st.sidebar.expander('**How it Works**', expanded=True):
    st.write('''
            
            ''')

st.markdown('<p class="font">Upload Excel Files</p>', unsafe_allow_html=True)

file_upload = st.file_uploader('Upload File(s)', type=['csv'], accept_multiple_files=True,
                               label_visibility='hidden')

root_dir_path = os.getcwd()
root_dir_path = os.path.abspath(os.path.split(root_dir_path)[0])
model_save_path = rf'{root_dir_path}/models/models/'
predictor = TabularPredictor.load(model_save_path)
label='kpb_activity_code'

if file_upload is not None:
    file_list = []

    for file in file_list:

        if file.type not in ['application/vnd.ms-excel',
                                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                st.error(
                    f'This file type is invalid: {file.name} must be an .xlsx or .xls file to proceed.')
                
        else:
             
            df = pd.read_csv(file)

            if df['KPB AKTIVITETSKOD'] is not None:
                df.drop(columns=['KPB AKTIVITETSKOD'])
            else:
                data = TabularDataset(df)
                single_predicted_code = predictor.predict(data = data, model = predictor.model_best)

                df[label] = single_predicted_code
                csv = df.to_csv(index=False)
                df_predicted = pd.read_csv(csv)
                file_name_no_ext = os.path.splitext(file.name)[0]

                if df.shape != df_predicted.shape:
                    st.error(f'''
                            The CSV file generated from {file.name} is not identical to the original file. 
                            Inspect the Excel file for merged/formatted cells, formulas, special characters, etc
                            and try again.
                            ''')
                else:
                    file_list.append((csv, file_name_no_ext + '.csv'))
                    st.download_button(
                        label=f'Download \'{file_name_no_ext}\' as CSV',
                        data=csv,
                        file_name=file_name_no_ext + '.csv',
                        mime='text/csv',
                        key=str(uuid.uuid4()))
                    
    if len(file_list) > 1:
        zipped_csvs = io.BytesIO()
        with zipfile.ZipFile(zipped_csvs, mode='w') as zip_file:
            for file_data, file_name in file_list:
                zip_file.writestr(file_name, file_data)
        st.download_button(
            label='Download All Files as Zip',
            data=zipped_csvs.getvalue(),
            file_name='Combined_CSVs.zip',
            mime='application/zip',
            key=str(uuid.uuid4()))