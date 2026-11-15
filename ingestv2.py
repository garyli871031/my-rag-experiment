import io
import os
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Read-only access is safest for ingestion pipelines
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Map native Google Workspace files to downloadable formats
GOOGLE_EXPORT_MAP = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "text/csv",
        ".csv",
    ),
    "application/vnd.google-apps.presentation": (
        "application/pdf",
        ".pdf",
    ),
}


def get_drive_service(credentials_path: str = "credentials.json", token_path: str = "token.json"):
    """Authenticates via OAuth2 Desktop flow and caches token.json for headless runs."""
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def sync_drive_folder_recursive(
    service,
    folder_id: str,
    local_dest: str | Path = "./docs/gdrive_sync",
) -> None:
    """Recursively traverses a Google Drive folder and downloads/exports all files locally."""
    dest_path = Path(local_dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=100,
                pageToken=page_token,
            )
            .execute()
        )

        for item in response.get("files", []):
            item_id = item["id"]
            item_name = item["name"]
            mime_type = item["mimeType"]

            # 1. Recurse into subfolders
            if mime_type == "application/vnd.google-apps.folder":
                print(f"[DIR]  Entering Drive folder: {item_name}")
                sync_drive_folder_recursive(service, item_id, dest_path / item_name)
                continue

            # 2. Export native Google Docs/Sheets/Slides
            if mime_type in GOOGLE_EXPORT_MAP:
                export_mime, ext = GOOGLE_EXPORT_MAP[mime_type]
                out_file = dest_path / f"{item_name}{ext}"
                request = service.files().export_media(fileId=item_id, mimeType=export_mime)
            # 3. Download standard binary files (PDFs, images, DOCX, etc.)
            elif not mime_type.startswith("application/vnd.google-apps"):
                out_file = dest_path / item_name
                request = service.files().get_media(fileId=item_id)
            else:
                print(f"[SKIP] Unsupported Google app type: {item_name} ({mime_type})")
                continue

            print(f"[FILE] Downloading: {out_file}")
            with io.FileIO(out_file, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

        page_token = response.get("nextPageToken")
        if not page_token:
            break