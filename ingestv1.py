import argparse
from local_reader import iter_local_files
#from gdrive_reader import get_drive_service, sync_drive_folder_recursive


def process_single_file(file_info: dict):
    """
    Replace this stub with your existing unstructured/LangChain
    chunking + embedding + pgvector insert logic.
    """
    print(
        f"Ingesting: {file_info['relative_path']} "
        f"(hash: {file_info['file_hash'][:12]}...)"
    )
    # Example:
    # if is_hash_in_pgvector(file_info["file_hash"]): return
    # elements = partition(filename=file_info["path"], ...)
    # embed_and_upsert_to_postgres(elements, metadata=file_info)


def main():
    parser = argparse.ArgumentParser(description="Offline RAG Document Ingestion Pipeline")
    parser.add_argument("--docs-dir", default="./docs", help="Local root folder to scan recursively")
    parser.add_argument("--drive-folder-id", default=None, help="Optional Google Drive folder ID to sync first")
    args = parser.parse_args()

    # Step 1: Optional online sync from Google Drive into local staging folder
    if args.drive_folder_id:
        print(f"Syncing Google Drive folder '{args.drive_folder_id}'...")
        service = get_drive_service()
        sync_drive_folder_recursive(
            service=service,
            folder_id=args.drive_folder_id,
            local_dest=f"{args.docs_dir}/gdrive_sync",
        )

    # Step 2: Offline recursive traversal & ingestion into pgvector
    count = 0
    for file_info in iter_local_files(args.docs_dir):
        process_single_file(file_info)
        count += 1

    print(f"Completed offline ingestion scan across {count} files.")


if __name__ == "__main__":
    main()