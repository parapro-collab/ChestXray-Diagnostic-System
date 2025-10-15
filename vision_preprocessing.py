
import os
import argparse
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import pydicom
import cv2
from tqdm import tqdm
import albumentations as A

# Utilitaires DICOM et I/O

def is_dicom(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            pre = f.read(132)
            return pre[-4:] == b'DICM'
    except Exception:
        return False


def read_dicom(path: str) -> (np.ndarray, dict):
    ds = pydicom.dcmread(path, force=True)
    # read pixel array
    arr = ds.pixel_array.astype(np.float32)
    meta = {
        'PatientID': getattr(ds, 'PatientID', None),
        'StudyDate': getattr(ds, 'StudyDate', None),
        'Modality': getattr(ds, 'Modality', None)
    }
    # handle RescaleSlope/Intercept
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    arr = arr * slope + intercept
    return arr, meta


def to_uint8(img: np.ndarray, clip_pct=(1,99)) -> np.ndarray:
    # robust contrast stretching using percentiles
    low, high = np.percentile(img, clip_pct)
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)
    img = (img * 255.0).astype(np.uint8)
    return img


def apply_window(img: np.ndarray, window_center: float=None, window_width: float=None) -> np.ndarray:
    if window_center is None or window_width is None:
        return to_uint8(img)
    lower = window_center - window_width/2.0
    upper = window_center + window_width/2.0
    clipped = np.clip(img, lower, upper)
    scaled = (clipped - lower) / (upper - lower)
    return (scaled * 255).astype(np.uint8)

# Prétraitement

def resize_and_pad(img: np.ndarray, size: int=512) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=resized.dtype)
    y = (size - nh) // 2
    x = (size - nw) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas


def clahe_enhance(img: np.ndarray, clipLimit=2.0, tileGridSize=(8,8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)


def preprocess_image_array(img: np.ndarray, size: int=512, clahe: bool=True) -> np.ndarray:
    # img : float32 or uint16 or uint8 single channel
    if img.dtype != np.uint8:
        img = to_uint8(img)
    if clahe:
        img = clahe_enhance(img)
    img = resize_and_pad(img, size)
    return img

# Augmentations (médicales)

def get_augmentations(size: int=512):
    # Compose d'augmentations compatibles pour images radiologiques
    return A.Compose([
        A.RandomRotate90(p=0.2),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Affine(shear=10, p=0.2)
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.6),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.4)
        ], p=0.6),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.2),
        A.Resize(size, size)
    ])

# Pipeline orchestration


def process_file(path: str, dst_dir: str, size: int=512, clahe: bool=True):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(path).stem
    try:
        if is_dicom(path):
            arr, meta = read_dicom(path)
            img = preprocess_image_array(arr, size=size, clahe=clahe)
        else:
            # image file
            pil = Image.open(path).convert('L')
            arr = np.array(pil)
            img = preprocess_image_array(arr, size=size, clahe=clahe)
        out_path = os.path.join(dst_dir, f"{filename}.png")
        Image.fromarray(img).save(out_path)
        return out_path
    except Exception as e:
        print(f"Failed processing {path}: {e}")
        return None


def build_pipeline(src: str, dst: str, size: int=512, clahe: bool=True, patterns=['**/*.dcm','**/*.png','**/*.jpg','**/*.jpeg']):
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(src, p), recursive=True))
    files = sorted(files)
    processed = []
    for f in tqdm(files, desc='Preprocessing'):
        out = process_file(f, dst, size=size, clahe=clahe)
        if out:
            processed.append(out)
    return processed


# Augmentation runner


def augment_folder(src: str, dst: str, n_per_image: int=2, size: int=512):
    Path(dst).mkdir(parents=True, exist_ok=True)
    aug = get_augmentations(size=size)
    files = sorted(glob.glob(os.path.join(src, '*.png')))
    for f in tqdm(files, desc='Augmenting'):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        base = Path(f).stem
        for i in range(n_per_image):
            augmented = aug(image=img)['image']
            out = os.path.join(dst, f"{base}_aug_{i}.png")
            cv2.imwrite(out, augmented)


# Validation routines


def validate_dataset(path: str, sample_n: int=6):
    files = sorted(glob.glob(os.path.join(path, '*.png')))
    if len(files) == 0:
        print("Aucun fichier trouvé pour validation.")
        return
    # Basic checks
    shapes = {}
    sizes = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Fichier illisible:", f)
            continue
        sizes.append(img.shape)
        shapes[img.shape] = shapes.get(img.shape,0)+1
    print("Nombre d'images:", len(files))
    print("Tailles d'images (shape counts):", shapes)
    print("Exemples:", files[:sample_n])
    # Pixel intensity stats
    import statistics
    means = []
    stds = []
    for f in files[:min(500, len(files))]:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        means.append(float(np.mean(img)))
        stds.append(float(np.std(img)))
    if means:
        print(f"Mean intensity (sample up to 500): {statistics.mean(means):.2f} +- {statistics.stdev(means):.2f}")


# CLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['build','augment','validate'], required=True)
    parser.add_argument('--src', type=str, default='data/raw')
    parser.add_argument('--dst', type=str, default='data/processed')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--path', type=str, default='data/processed')
    args = parser.parse_args()

    if args.mode == 'build':
        processed = build_pipeline(args.src, args.dst, size=args.size)
        print(f"Processed {len(processed)} files.")
    elif args.mode == 'augment':
        augment_folder(args.src, args.dst, n_per_image=args.n, size=args.size)
    elif args.mode == 'validate':
        validate_dataset(args.path)

if __name__ == '__main__':
    main()
