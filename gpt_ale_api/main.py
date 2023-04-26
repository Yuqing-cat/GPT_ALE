import json
import os
import threading
import time

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from fastapi import APIRouter, FastAPI,BackgroundTasks
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from schemas import Config
import utils


DEFAULT_AZURE_CLIENT_ID = "" 
DEFAULT_AZURE_TENANT_ID = "" 

config = Config(
    clientId = os.environ.get("AZURE_CLIENT_ID", DEFAULT_AZURE_CLIENT_ID),
    tenantId = os.environ.get("AZURE_TENANT_ID", DEFAULT_AZURE_TENANT_ID),
    azureOAuthEnable = True
)
 
default_connection_string = ""
storage_connection_string = os.environ.get("CONNECTION_STR", default_connection_string)

default_blob_container = "demo"
default_data_folder = "annotated/archive"

class Annotation(BaseModel):
    id:str
    label:str
    ann_by:str

# common parameters
# TODO: decouple cloud file path definition from main.py
# Cloud path Definitions and local path definitions
job_metadata_file_cloud_path = os.path.join('annotated/archive/jobs.json')
category_file_cloud_path = os.path.join('unannotated/mapping.json')
annotations_file_cloud_path = os.path.join('annotations.json') 
unannotated_file_cloud_path = 'unannotated/data.pkl'
metric_file_cloud_path = 'metrics.json'

job_metadata_file_local_path = utils.get_local_path_from_cloud_path(job_metadata_file_cloud_path)
category_file_local_path = utils.get_local_path_from_cloud_path(category_file_cloud_path)
annotations_file_local_path = utils.get_local_path_from_cloud_path(annotations_file_cloud_path)
unannotated_file_local_path = utils.get_local_path_from_cloud_path(unannotated_file_cloud_path)
metric_file_local_path = utils.get_local_path_from_cloud_path(metric_file_cloud_path)

make_sure_this_finishes = False
categories_buffer = {}

def download_file_from_blob_loop():
    '''
    This function runs every 2 min
    '''
    while True:
        try:
            retrieve_latest_data_from_blob()
            time.sleep(60*2)
        except Exception as ex:
            print(ex)
            continue

def retrieve_latest_data_from_blob():
    '''
    This function fetches files from blob, and buffer to local folder for faster processing.
    '''
    if make_sure_this_finishes:
        return
    # At the beginning, check cloud-side status. 
    # If a file appears, but not tracked by the json, we need to update the json.
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    my_container = blob_service_client.get_container_client(default_blob_container)
    blobs = my_container.list_blobs(name_starts_with=default_data_folder)
    job_metadata_collection = []
    json_metadata_blob = my_container.get_blob_client(job_metadata_file_cloud_path)

    # apart from archived folder, download the data.pkl from ununnotated folder.
    # since this folder could be overwritten, read every loop and override local copy.
    download_single_file_from_blob(blob_service_client,unannotated_file_cloud_path,unannotated_file_local_path,force_download=True)
    
    if json_metadata_blob.exists():
        # If the json file is not present (first run), download it.
        # otherwise, keep it empty
        bytes = my_container.get_blob_client(job_metadata_file_cloud_path).download_blob().readall()
        if len(bytes)==0:
            json_metadata_blob.delete_blob()
            os.remove(job_metadata_file_local_path)
            return
        os.makedirs(os.path.dirname(job_metadata_file_local_path), exist_ok=True)
        with open(job_metadata_file_local_path, "wb+") as file:
            file.write(bytes)
        job_metadata_collection=json.load(open(job_metadata_file_local_path,'rb'))
    
    # find the blob that is not in json, run analyze logic, then add it. 
    # we do not update existing blocks inside the json.
    update_archived_jobs(blob_service_client, blobs, job_metadata_collection, json_metadata_blob)

def update_archived_jobs(blob_service_client, blobs, job_metadata_collection, json_metadata_blob):
    # optimization here (yihui): since acc/precision /.. are almost impossible to be the same as last job.
    # Then if that happens, DO NOT re-use the last job status, empty is empty.
    # corner case here, value - empty - empty - new value. The second "empty" should not be replaced by "value"

    job_names_from_json = [x['jobid'] for x in job_metadata_collection]
    for blob in blobs:
        if blob.name.endswith('pkl'):
            blob_file_name = blob.name
            local_file_path = os.path.join('./download',blob_file_name)
            job_id = blob_file_name.split('/')[-1].split('data_')[-1].split('.pkl')[0]
                    
                    # download the file, if it is already downloaded, skip.
            download_single_file_from_blob(blob_service_client,blob_file_name,local_file_path)

            is_blob_in_json = any([x for x in job_names_from_json if x in blob_file_name])
            metadata = analyze_completed_run(blob_file_name,job_id)
            if os.path.isfile(metric_file_local_path):
                os.remove(metric_file_local_path)
            download_single_file_from_blob(blob_service_client,metric_file_cloud_path,metric_file_local_path)
            metric_object = json.load(open(metric_file_local_path,'r')) if os.path.isfile(metric_file_local_path) else None

            if not is_blob_in_json:  
                # this is the place to handle new job, assign metric, heat map to the job.
                metadata['creation_time'] = str(blob.creation_time)
                metadata['can_annotate']=True
                if metric_object:
                    last_not_empty_jobs = [x for x in job_metadata_collection if 'acc' in x]
                    last_not_empty_job = None if len(last_not_empty_jobs)==0 else last_not_empty_jobs[-1]
                    metadata['acc'] = str(round(metric_object['global']['accuracy_score'][-1][1]*100,2))
                    metadata['precision'] = str(round(metric_object['global']['precision_score'][-1][1]*100,2))
                    metadata['recall'] = str(round(metric_object['global']['recall_score'][-1][1]*100,2))
                    metadata['f1'] = str(round(metric_object['global']['f1_score'][-1][1]*100,2))
                    analyze_heat_map_and_confusion_mat(metadata, metric_object)
                job_metadata_collection.append(metadata)                
            
            # save the json object, both locally and on cloud.
            # local save
            else:
                target_metadata_to_update = [x for x in job_metadata_collection if x['jobid']==metadata['jobid']][0]
                if target_metadata_to_update['can_annotate']==True:
                    label_to_target,_ = get_target_label()
                    target_metadata_to_update['number_of_category']=len(label_to_target.keys())
                    if metric_object:
                        analyze_heat_map_and_confusion_mat(target_metadata_to_update, metric_object)
    with open (job_metadata_file_local_path,'w+') as jsonf:
        json.dump(job_metadata_collection,jsonf)

            # cloud save
    if(json_metadata_blob.exists()):
        json_metadata_blob.delete_blob()
    with open(job_metadata_file_local_path, "rb") as stream:
        json_metadata_blob.upload_blob(stream, blob_type="BlockBlob")

def analyze_heat_map_and_confusion_mat(metadata, metric_object):
    metadata['heat_map'] ={}
    for category in metric_object['confusion_matrix']:
        metadata['heat_map'].setdefault(category,{})
        for i,row in enumerate(metric_object['confusion_matrix'][category]):
            for j,val in enumerate(row):
                metadata['heat_map'][category].setdefault((i,j),0)
                metadata['heat_map'][category][(i,j)]=val
        metadata['heat_map'][category] = json.dumps([(k[0],k[1],v) for k,v in metadata['heat_map'][category].items()])

def analyze_completed_run(file_cloud_name,job_id):
    # run for every archived job.
    job_metadata = {}
    label_to_target,_ = get_target_label()
    job_metadata.setdefault('jobid',job_id)
    job_metadata.setdefault('number_of_category',len(label_to_target.keys()))
    # by default, all jobs are open for annotation, until it's annotated.
    job_metadata.setdefault('can_annotate',True)
    return job_metadata

def get_target_label():
    global categories_buffer
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    download_single_file_from_blob(blob_service_client,category_file_cloud_path,category_file_local_path,True)
    download_single_file_from_blob(blob_service_client,annotations_file_cloud_path,annotations_file_local_path,True)

    label_to_target =json.load(open(category_file_local_path,'r'))
    if categories_buffer:
        label_to_target = label_to_target | categories_buffer
    target_to_label = {v:k for k,v in label_to_target.items()}
    return label_to_target,target_to_label

def download_single_file_from_blob(blob_service_client,cloud_file_path,local_file_path,force_download = False):
    if (not os.path.isfile(local_file_path)) or force_download:
        my_container = blob_service_client.get_container_client('demo')
        blob = my_container.get_blob_client(cloud_file_path)
        if blob.exists():
            bytes = blob.download_blob().readall()

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            with open(local_file_path, "wb+") as file:
                file.write(bytes)
            if cloud_file_path.endswith('.pkl') and not force_download:
                print('[Pickle Cache]downloaded {}'.format(cloud_file_path))

def upload_annotated_file_to_blob_new(df):
    blob_service_client =  BlobServiceClient.from_connection_string(storage_connection_string)
    target_file_name = os.path.join("annotated/new/data.pkl")
    my_container = blob_service_client.get_container_client('demo')
    
    os.makedirs(os.path.join('./download',"annotated", "new"), exist_ok=True)
    df.to_pickle(os.path.join('./download',target_file_name))
    
    upload_single_file(target_file_name, my_container)
    return "uploaded,waiting for Job to pick up"

def upload_single_file(target_file_name, my_container):
    blob_client = my_container.get_blob_client(target_file_name)

    if(blob_client.exists()):
        blob_client.delete_blob()
        time.sleep(5)
    with open(os.path.join('./download',target_file_name), "rb") as stream:
        blob_client.upload_blob(stream, blob_type="BlockBlob")
app = FastAPI()
router = APIRouter()

# Enables CORS
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

bg_thread = threading.Thread(name='background', target=download_file_from_blob_loop)
bg_thread.start()

@router.get("/all_points",tags=["SME"])
def get_all_background_points_for_annotation():

    df = pd.read_pickle(unannotated_file_local_path)
    df = df[df['tsne']!=0]
    gray_points = []
    for row in df.itertuples():
        gray_points.append([int(x*10000) for x in json.loads(row.tsne)])
    return gray_points

@router.get("/all_points_plus",tags=["SME"])
def get_all_background_points_for_annotation_with_more_info():
    unannotated_df = pd.read_pickle(unannotated_file_local_path)
    unannotated_df = unannotated_df[unannotated_df['tsne']!=0]
    job_metadata = json.load(open(job_metadata_file_local_path,'r'))
    additional_info = {}
    additional_info['transformations'] = []

    if len(job_metadata)>1:
        last_successful_run = job_metadata[-2]['jobid']
        latest_archive_df = pd.read_pickle("./download/annotated/archive/data_{0}.pkl".format(last_successful_run))
        latest_archive_df = latest_archive_df[latest_archive_df['ann']>-1]

        for row in latest_archive_df.itertuples():
            if row.Index in unannotated_df.index and 'tsne' in latest_archive_df and 'tsne' in unannotated_df:
                new_loc = unannotated_df.loc[row.Index].tsne
                if new_loc != row.tsne:
                    transformation_object={}
                    transformation_object.setdefault("index",row.Index)
                    transformation_object.setdefault("old_loc",[int(x*10000) for x in json.loads(row.tsne)])
                    transformation_object.setdefault("new_loc",[int(x*10000) for x in json.loads(new_loc)])
                    additional_info['transformations'].append(transformation_object)

    additional_info['gray_points']=[]
    for row in unannotated_df.itertuples():
        object = {}
        id = row.Index
        object['id']=id
        object['text'] = row.text
        object['title'] = row.title
        object['label'] = row.label
        object['point'] = [int(x*10000) for x in json.loads(row.tsne)]
        object['score'] = row.score
        object['gpt'] = row.gpt3
        object['ann_by'] = row.ann_by

        additional_info['gray_points'].append(object)
    return additional_info

@router.get("/annotation_target",tags=["SME"])
def get_metadata_and_points_for_annotation():
    df = pd.read_pickle(unannotated_file_local_path)
    df = df[df['tsne']!=0]
    df_ann = df.loc[df['ann'] == -1]
    color_points=[]
    for row in df_ann.itertuples():
        object = {}
        id = row.Index
        object['id']=id
        object['text'] = row.text
        object['title'] = row.title
        object['label'] = row.label
        object['point'] = [int(x*10000) for x in json.loads(row.tsne)]
        object['score'] = row.score
        object['gpt'] = row.gpt3
        object['ann_by'] = row.ann_by


        color_points.append(object)
    color_points = sorted(color_points,key=lambda x:-x['score'])
    return color_points[:40]

@router.post('/update_annotation',tags=['SME'])
def update_annotation_and_continue_hyperloop(json_input:list[Annotation],run_id):
    global make_sure_this_finishes,categories_buffer 

    blob_service_client =  BlobServiceClient.from_connection_string(storage_connection_string)
    my_container = blob_service_client.get_container_client('demo')

    make_sure_this_finishes = True
    df = pd.read_pickle(unannotated_file_local_path)    
    input_dict = {int(x.id):x for x in json_input}  
    label_to_target,_ = get_target_label()
    for row in df.itertuples():
        # in theory, for incoming annotations, we only need to update "ann"
        # considering mapping.json is somehow inaccurate, we use the client-side annotation dict as supplement.
        # so that we could combine the client-side categories with server-side mapping.json.
        if row.Index in input_dict:
            df.loc[row.Index,'label'] = input_dict[row.Index].label
            df.loc[row.Index,'ann_by'] = input_dict[row.Index].ann_by
            if input_dict[row.Index].label in label_to_target:
                df.loc[row.Index,'ann'] = label_to_target[input_dict[row.Index].label]
            else:
                max_cat_ind = max(label_to_target.values())
                label_to_target.setdefault(input_dict[row.Index].label,max_cat_ind+1)
                df.loc[row.Index,'ann'] = max_cat_ind+1

    categories_buffer = categories_buffer | label_to_target
                
    job_metadata_file_local_path = os.path.join('./download/annotated/archive/jobs.json')
    job_metadata = json.load(open(job_metadata_file_local_path,'r'))
    for meta in job_metadata:
        if meta['jobid'] == run_id:
            meta['can_annotate']=False
    json.dump(job_metadata,open(job_metadata_file_local_path,'w+'))
    upload_single_file(job_metadata_file_local_path.split('./download/')[1], my_container)
    make_sure_this_finishes=False
    return upload_annotated_file_to_blob_new(df)


@router.get("/all_runs",tags=["Runs"])
async def get_all_archived_job_runs(background_tasks:BackgroundTasks):
    background_tasks.add_task(retrieve_latest_data_from_blob)
    job_metadata = json.load(open(job_metadata_file_local_path,'r'))
    # ensure the latest one is able to annotate
    job_metadata = sorted(job_metadata,key=lambda x: x['jobid'])
    blob_service_client =  BlobServiceClient.from_connection_string(storage_connection_string)
    my_container = blob_service_client.get_container_client('demo')



    lock_file_blob = my_container.get_blob_client("lock.txt") 
    new_data_file_blob = my_container.get_blob_client("annotated/new/data.pkl")
    if lock_file_blob.exists() and not any([x for x in job_metadata if x['can_annotate']]):
        job_metadata.append( {"jobid": "One Job Is Running","can_annotate": False})
    elif new_data_file_blob.exists():
        job_metadata.append( {"jobid": "One Job Queued","can_annotate": False})
    # upload_single_file(job_metadata_file_local_path.split('./download/')[1], my_container)
    return job_metadata

@router.get("/all_unfinished_runs",tags=["Runs"])
async def get_all_archived_job_runs():
    result = []
    blob_service_client =  BlobServiceClient.from_connection_string(storage_connection_string)
    my_container = blob_service_client.get_container_client('demo')
    lock_file_blob = my_container.get_blob_client("lock.txt") 
    new_data_file_blob = my_container.get_blob_client("annotated/new/data.pkl")
    progress_blob = my_container.get_blob_client("progress.log")

    if lock_file_blob.exists():
        progress=0
        job_metadata = json.load(open(job_metadata_file_local_path,'r'))
        if any([x for x in job_metadata if x['can_annotate']==True]):
            if progress_blob.exists():
                with open('progress.log','w+') as progressf:
                    progressf.write('0')
                upload_single_file('progress.log',my_container)
                
                return [{"jobid": "Job Running","progress": 100}]
            return []

        if progress_blob.exists():
            download_single_file_from_blob(blob_service_client,"progress.log","./download/progress.log",force_download=True)
            progress_content = json.load(open("./download/progress.log",'r'))
            progress = float(progress_content)
            return [{"jobid": "Job Running","progress": progress}]
    elif new_data_file_blob.exists():
        # known issue here: if data.pkl was used in previous job, and generated by next job, this logic here will have some error.
        return [{"jobid": "Job Queued","progress":0}]
    else:
        return []

@router.get("/job_point_cloud",tags=["Runs"])
def get_point_cloud_of_prediction(run_id):
    full_name = "annotated/archive/"+"data_"+run_id+'.pkl'
    full_path = os.path.join('./download',full_name)
    df = pd.read_pickle(full_path)
    point_cloud = {}
    if 'tsne' in df:
        df = df[df['tsne']!=0]
        point_cloud = {}
        for row in df.itertuples():
            point_cloud.setdefault('category',{})
            points = [int(x*10000) for x in json.loads(row.tsne)]
            point_cloud['category'].setdefault(row.label,[])
            point_cloud['category'][row.label].append(points)
    return point_cloud

@router.get("/line_charts",tags=["Runs"])
def get_data_for_all_line_charts():
    metric_object = json.load(open(metric_file_local_path,'r')) if os.path.isfile(metric_file_local_path) else None
    if metric_object:
        target_object = {k:{} for k in metric_object.keys()}
        for chart_name in metric_object.keys():
            target_object[chart_name] = metric_object[chart_name]
            for line_name in target_object[chart_name].keys():
                vals = target_object[chart_name][line_name]
                for i in range(len(vals)):
                    vals[i][0]=i
        del target_object['confusion_matrix']
        return target_object
            

@router.get("/run_status",tags=["Runs"])
def get_details_for_a_job(run_id):
    annotations = json.load(open(annotations_file_local_path,'r'))
    job_metadata = json.load(open(job_metadata_file_local_path,'r'))
    job_target = [x for x in job_metadata if run_id == x['jobid']][0]
    job_can_annotate = job_target['can_annotate']
    if job_can_annotate:
        df = pd.read_pickle(unannotated_file_local_path)
    else:
        full_name = "annotated/archive/"+"data_"+run_id+'.pkl'
        full_path = os.path.join('./download',full_name)
        df = pd.read_pickle(full_path)
    
    _,target_to_label = get_target_label()
    job_detail = {}
    job_detail['can_annotate'] = job_can_annotate
    if 'heat_map' in job_target:
        job_detail.setdefault('heat_map',job_target['heat_map']['none'])
        job_detail.setdefault('all_heat_maps',job_target['heat_map'])
    
    if job_can_annotate:
        job_detail.setdefault('more_info',get_all_background_points_for_annotation_with_more_info())

    # target to label
    job_detail['target_to_label'] = target_to_label

    # pie chart
    df_ann = df[df['ann']>-1]
    job_detail['pie_chart'] = json.dumps([(x[0],x[1].shape[0]) for x in df_ann.groupby('ann')])
    job_detail['annotations'] = annotations
    return job_detail


@router.get("/config", response_model=Config)
def get_config():
    return config

app.include_router(router=router)
