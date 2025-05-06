from pathlib import Path

from safetensors.torch import save_model
from .pretrained import load_pretrained

def main():
    model = load_pretrained("demucs")
    print(model)

    mlx_path = Path("/tmp/mlx")
    mlx_path.mkdir(parents=True, exist_ok=True)
    out = mlx_path / "demucs.safetensors"

    save_model(model, out)
    print(f"Model saved to {out}")

if __name__ == "__main__":
  main()
