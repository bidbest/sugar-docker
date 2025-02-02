import os
import time
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
import subprocess
import shutil

# Google Drive Folder IDs
INPUT_FOLDER_ID = "1T5DGnBa5muJTQ21-Z4L__vHOsl-vzJ7S"
OUTPUT_FOLDER_ID = "1TQWkOOzFj4dsgZuwdu_32ZJvM7OASyXI"
DONE_FOLDER_ID = "1CjC_Id-gRXSMwLwYi0d0csXQWIlMokEk"
CREDENTIALS_FILE = "/sugar/datasets_gs/gs-server-447602-380ce272a782.json"

# Local paths
LOCAL_FOLDER = "/sugar/datasets_gs/"

# Authenticate Google Drive
def authenticate_drive():
    scopes = ["https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
    return build("drive", "v3", credentials=credentials)

def check_n_frames(new_video_id):
    file_name = new_video_id['name']
    n = 100
    
    if '_n_' in file_name:
        n_tmp = file_name.split('_n_')[1]
        n = int(n_tmp.split('.')[0])

    print(f"Using {n} frames")
    return n

# Download a file from Google Drive
def download_file(drive_service, new_video_id, ws_path):
    file_id = new_video_id['id']
    file_name = new_video_id['name']
    local_path = os.path.join(ws_path, file_name)

    if os.path.isfile(local_path):
        print(f"{file_name} already downloaded")
        return

    request = drive_service.files().get_media(fileId=file_id)
    print(f"Doanloading {file_name} .....")
    with open(local_path, "wb") as f:
        f.write(request.execute())

    print(f"{file_name} Downloaded!")

# Upload a file to Google Drive
def upload_to_drive(drive_service, ws_path, output_folder_id, renamed_file_name, retries=5):
    renamed_file_name = renamed_file_name.split('.')[0] + '.ply'
    file_metadata = {"name": renamed_file_name, "parents": [output_folder_id]}
    local_file_path = os.path.join(ws_path, "model/point_cloud/iteration_7000/point_cloud.ply")
    if not os.path.isfile(local_file_path):
        print("Something went wrong, exit!")
        sys.exit(1)

    print("Uploading 3dgs ....") 
    media = MediaFileUpload(local_file_path, mimetype="application/octet-stream")
    
    for attempt in range(retries):
        try:
            drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
            print("3dgs uploaded!")
            return
            
        except Exception as e:
            print(f"Upload failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)  # Wait before retrying
            else:
                raise


# Check for new videos in Google Drive
def fetch_new_videos(drive_service):
    query = f"'{INPUT_FOLDER_ID}' in parents and mimeType contains 'video/'"
    response = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = response.get("files", [])
    
    return files

# Move video Once done
def move_file(drive_service, file_id, new_folder_id):
    # Retrieve the existing parents of the file
    file = drive_service.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents", []))
    # Move the file to the new folder
    drive_service.files().update(
        fileId=file_id,
        addParents=new_folder_id,
        removeParents=previous_parents,
        fields="id, parents"
    ).execute()


def create_local_ws(new_video_id):
    file_name = new_video_id['name']
    folder_name = file_name.split('.')[0]
    ws_path = os.path.join(LOCAL_FOLDER, folder_name)
    print(f"Creating local path:\n{ws_path}")
    os.makedirs(ws_path, exist_ok=True)

    return ws_path


def create_gs(ws_path, n_frames=100):
    command = f"conda run --live-stream -n sugar python do_all.py -s {ws_path} -n {n_frames}"
    subprocess.run(command, shell=True, check=True)


# Main workflow
def main():
    drive_service = authenticate_drive()

    while True:
        print("Checking for new videos...")
        new_videos = fetch_new_videos(drive_service)
        
        if len(new_videos) < 1:
            print("No new videos. Sleeping for 10 seconds...")
            time.sleep(10)
            continue
        
        new_video_id = new_videos[0]
        print(f"Found new video: { new_video_id['name']}")
        ws_path = create_local_ws(new_video_id)
        download_file(drive_service, new_video_id, ws_path)
        n_frames = check_n_frames(new_video_id)
        create_gs(ws_path, n_frames)
        upload_to_drive(drive_service, ws_path, OUTPUT_FOLDER_ID, new_video_id['name'])
        move_file(drive_service, new_video_id['id'], DONE_FOLDER_ID)

if __name__ == "__main__":
    main()
