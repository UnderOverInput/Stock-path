#!/usr/bin/env python3
"""
Deploy Stock Prediction API to Hugging Face Spaces
"""
import os
import sys
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from pathlib import Path

def deploy_to_huggingface(token: str, space_name: str, username: str):
    """
    Deploy the FastAPI application to Hugging Face Spaces

    Args:
        token: HF access token with write permissions
        space_name: Name for your space (e.g., "stock-prediction-api")
        username: Your HF username
    """
    api = HfApi(token=token)

    # Create the full repo ID
    repo_id = f"{username}/{space_name}"

    print(f"🚀 Deploying to Hugging Face Spaces: {repo_id}")

    try:
        # Create the space
        print("Creating Space...")
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True  # Don't fail if space already exists
        )
        print(f"✅ Space created: https://huggingface.co/spaces/{repo_id}")

        # Upload files
        print("\n📤 Uploading files...")
        files_to_upload = [
            "README.md",
            "requirements.txt",
            "Dockerfile",
            "app.py"
        ]

        for file in files_to_upload:
            if Path(file).exists():
                print(f"  Uploading {file}...")
                upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=repo_id,
                    repo_type="space",
                    token=token
                )
            else:
                print(f"  ⚠️  {file} not found, skipping...")

        print(f"\n✨ Deployment complete!")
        print(f"🌐 Your API will be available at: https://{username}-{space_name}.hf.space")
        print(f"📊 Space URL: https://huggingface.co/spaces/{repo_id}")
        print(f"📝 API Docs: https://{username}-{space_name}.hf.space/docs")
        print(f"\n⏰ Note: First build may take 5-10 minutes")

        return repo_id

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    print("🤗 Hugging Face Spaces Deployment Tool")
    print("=" * 50)

    # Get token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\n🔑 Hugging Face Token Required")
        print("Get your token from: https://huggingface.co/settings/tokens")
        token = input("Paste your HF token (with write permissions): ").strip()

    if not token:
        print("❌ Token is required!")
        sys.exit(1)

    # Get username
    username = input("\nEnter your HF username: ").strip()
    if not username:
        print("❌ Username is required!")
        sys.exit(1)

    # Get space name
    default_name = "stock-prediction-api"
    space_name = input(f"\nEnter space name (default: {default_name}): ").strip() or default_name

    print(f"\n📋 Configuration:")
    print(f"  Username: {username}")
    print(f"  Space: {space_name}")
    print(f"  Repo: {username}/{space_name}")

    confirm = input("\nProceed with deployment? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Deployment cancelled.")
        sys.exit(0)

    # Deploy
    deploy_to_huggingface(token, space_name, username)


if __name__ == "__main__":
    main()
