"""Quick script to list all files in the checkpoint volume."""
import modal

app = modal.App("list-checkpoints")
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=False)

@app.function(volumes={"/checkpoints": checkpoint_volume})
def list_files():
    import os
    print("Files in /checkpoints:")
    found = []
    for root, dirs, files in os.walk("/checkpoints"):
        for f in files:
            full = os.path.join(root, f)
            size = os.path.getsize(full)
            print(f"  {full}  ({size/1024/1024:.1f} MB)")
            found.append(full)
    if not found:
        print("  (empty)")
    return found

@app.local_entrypoint()
def main():
    files = list_files.remote()
    print(f"\nTotal files found: {len(files)}")
