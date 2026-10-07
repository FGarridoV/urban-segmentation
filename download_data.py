import requests
import zipfile
import os
from tqdm import tqdm

subset = 'https://surfdrive.surf.nl/files/index.php/s/P448UpWEnEkqKKJ/download'
fullset = "https://surfdrive.surf.nl/files/index.php/s/v52aKdVspw5yMJ3/download"

def download_images(full=False):
    os.makedirs('data', exist_ok=True)
    
    print(f"Downloading {'full' if full else 'subset'} images...")
    _download_image_zip(full)
    print("Download complete!")

    print("Unzipping images...")
    _unzip_images()
    print("Unzip complete!")

    print("Removing zip file...")
    os.remove('data/images.zip')

    print("Done!")

def _download_image_zip(full=False):
    global subset, fullset
    url = fullset if full else subset
    
    # IMPORTANTE: Usar stream=True para no cargar todo el archivo en la memoria RAM
    with requests.get(url, allow_redirects=True, stream=True) as r:
        r.raise_for_status()
        
        # Intentar obtener el tamaño total del archivo si el servidor lo provee
        total_size = int(r.headers.get('content-length', 0))
        
        with open('data/images.zip', 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
                bar.update(len(chunk))

def _unzip_images():
    with zipfile.ZipFile('data/images.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download urban images")
    parser.add_argument("--full", action="store_true", help="Download the full dataset instead of the subset")
    args = parser.parse_args()
    
    download_images(full=args.full)
