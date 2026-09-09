# modal_inspect_volumes.py
"""Quick script to inspect Modal volume contents."""
import modal

app = modal.App("inspect-volumes")

data_volume = modal.Volume.from_name("cataract-data", create_if_missing=False)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=False)


@app.function(
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=300,
)
def inspect_volumes():
    import os
    from pathlib import Path
    from collections import defaultdict

    def walk(path, max_depth=3, prefix=""):
        """Walk a directory tree up to max_depth."""
        entries = []
        try:
            for entry in sorted(Path(path).iterdir()):
                if entry.is_dir():
                    n_files = sum(1 for _ in entry.rglob("*") if _.is_file())
                    entries.append(f"{prefix}📁 {entry.name}/  ({n_files} files)")
                    if max_depth > 0:
                        entries.extend(walk(entry, max_depth - 1, prefix + "  "))
                else:
                    size_mb = entry.stat().st_size / (1024 * 1024)
                    entries.append(f"{prefix}📄 {entry.name}  ({size_mb:.1f} MB)")
        except PermissionError:
            entries.append(f"{prefix}❌ Permission denied")
        except Exception as e:
            entries.append(f"{prefix}❌ {e}")
        return entries

    print("=" * 70)
    print("  cataract-data volume (/data)")
    print("=" * 70)
    for line in walk("/data", max_depth=3):
        print(line)

    print("\n" + "=" * 70)
    print("  cataract-checkpoints volume (/checkpoints)")
    print("=" * 70)
    for line in walk("/checkpoints", max_depth=3):
        print(line)

    # Count images by type
    print("\n" + "=" * 70)
    print("  IMAGE COUNT SUMMARY")
    print("=" * 70)
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for root_name, root_path in [("data", "/data"), ("checkpoints", "/checkpoints")]:
        counts = defaultdict(int)
        total = 0
        for p in Path(root_path).rglob("*"):
            if p.is_file() and p.suffix.lower() in img_exts:
                counts[p.suffix.lower()] += 1
                total += 1
        if total > 0:
            print(f"\n  {root_name}: {total} images")
            for ext, count in sorted(counts.items()):
                print(f"    {ext}: {count}")
        else:
            print(f"\n  {root_name}: 0 images")

    # Check for CSV/label files
    print("\n" + "=" * 70)
    print("  LABEL / CSV FILES")
    print("=" * 70)
    for root_name, root_path in [("data", "/data"), ("checkpoints", "/checkpoints")]:
        for p in Path(root_path).rglob("*"):
            if p.suffix.lower() in ('.csv', '.xlsx', '.xls', '.json'):
                size_kb = p.stat().st_size / 1024
                print(f"  {root_name}: {p.relative_to(root_path)}  ({size_kb:.1f} KB)")

    # Check for model checkpoints
    print("\n" + "=" * 70)
    print("  MODEL CHECKPOINTS")
    print("=" * 70)
    for p in Path("/checkpoints").rglob("*.pth"):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.relative_to('/checkpoints')}  ({size_mb:.1f} MB)")
    for p in Path("/checkpoints").rglob("*.safetensors"):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.relative_to('/checkpoints')}  ({size_mb:.1f} MB)")
    for p in Path("/checkpoints").rglob("*.pkl"):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.relative_to('/checkpoints')}  ({size_mb:.1f} MB)")

    return "done"


@app.local_entrypoint()
def main():
    print("🔍 Inspecting Modal volumes...")
    inspect_volumes.remote()