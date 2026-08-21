import shutil
import urllib.request
import zipfile
from pathlib import Path

from common import DATA_ROOT


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
WARPEACE_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"


def fetch(url, destination):
    print(f"downloading {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(destination, "wb") as f:
        shutil.copyfileobj(response, f)


def report(name, ready):
    print(f"{'ready:' if ready else 'MISSING:'} {name}")


def prepare_shakespeare():
    target = DATA_ROOT / "shakespeare" / "input.txt"
    if not target.exists():
        fetch(SHAKESPEARE_URL, target)
    report("shakespeare", target.exists())


def prepare_movielens():
    ratings = DATA_ROOT / "ml-1m" / "ratings.dat"
    if not ratings.exists():
        archive = DATA_ROOT / "_ml-1m.zip"
        if not archive.exists():
            fetch(MOVIELENS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extract("ml-1m/ratings.dat", DATA_ROOT)
        archive.unlink()
    report("movielens", ratings.exists())


def prepare_cifar100():
    import torchvision

    root = str(DATA_ROOT / "cifar100")
    torchvision.datasets.CIFAR100(root, train=True, download=True)
    torchvision.datasets.CIFAR100(root, train=False, download=True)
    report("cifar100", (DATA_ROOT / "cifar100" / "cifar-100-python").exists())


def prepare_molhiv():
    from ogb.graphproppred import PygGraphPropPredDataset

    PygGraphPropPredDataset(name="ogbg-molhiv", root=str(DATA_ROOT / "molhiv"))
    report("molhiv", (DATA_ROOT / "molhiv").exists())


def prepare_tiny_imagenet():
    root = DATA_ROOT / "tiny-imagenet-200"
    train_dir = root / "train"
    val_dir = root / "val"
    val_images = val_dir / "images"
    if not train_dir.exists():
        archive = DATA_ROOT / "_tiny-imagenet-200.zip"
        if not archive.exists():
            fetch(TINY_IMAGENET_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_ROOT)
        archive.unlink()
    if val_images.exists():
        for line in (val_dir / "val_annotations.txt").read_text().splitlines():
            filename, wnid = line.split("\t")[:2]
            class_dir = val_dir / wnid
            class_dir.mkdir(exist_ok=True)
            shutil.move(val_images / filename, class_dir / filename)
        shutil.rmtree(val_images)
    report("tiny_imagenet", train_dir.exists() and val_dir.exists() and not val_images.exists())


def prepare_warpeace():
    target = DATA_ROOT / "warpeace" / "input.txt"
    if not target.exists():
        raw = DATA_ROOT / "_warpeace_raw.txt"
        if not raw.exists():
            fetch(WARPEACE_URL, raw)
        text = raw.read_text(encoding="utf-8")
        start = text.find("*** START OF")
        end = text.find("*** END OF")
        if start != -1 and end != -1:
            start = text.find("\n", start) + 1
            text = text[start:end]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        raw.unlink()
    report("warpeace", target.exists())


PREPARERS = [
    ("shakespeare", prepare_shakespeare),
    ("movielens", prepare_movielens),
    ("cifar100", prepare_cifar100),
    ("molhiv", prepare_molhiv),
    ("tiny_imagenet", prepare_tiny_imagenet),
    ("warpeace", prepare_warpeace),
]


def main(selected=None):
    names = [name for name, _ in PREPARERS] if selected is None else list(selected)
    table = dict(PREPARERS)
    failed = []
    for name in names:
        print(f"{name}:")
        try:
            table[name]()
        except Exception as e:
            print(f"failed: {name}: {e}")
            failed.append(name)
    if failed:
        print(f"\nproblems: {', '.join(failed)}")
        return False
    return True


if __name__ == "__main__":
    main()
