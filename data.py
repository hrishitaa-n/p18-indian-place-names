
import os, re, io, zipfile, random, urllib.request

URL = "https://download.geonames.org/export/dump/IN.zip"   # GeoNames, licence CC BY 4.0

def main(out="data", seed=0):
    os.makedirs(out, exist_ok=True)
    raw = os.path.join(out, "IN.zip")
    if not os.path.exists(raw):
        print("downloading GeoNames India dump ...")
        urllib.request.urlretrieve(URL, raw)
    names, rows = set(), 0
    with zipfile.ZipFile(raw) as z, z.open("IN.txt") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 8 or p[6] != "P":          # feature class P = populated place
                continue
            rows += 1
            n = re.sub(r"[^a-z]+", " ", p[2].lower()).strip()   # asciiname column
            if 3 <= len(n) <= 24:
                names.add(n)
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n = len(names); a, b = int(.9 * n), int(.95 * n)
    for fn, part in [("train", names[:a]), ("val", names[a:b]), ("test", names[b:])]:
        open(f"{out}/{fn}.txt", "w").write("\n".join(part))
    print(f"populated-place rows: {rows:,} | unique clean names: {n:,}")
    print(f"train/val/test = {a:,}/{b-a:,}/{n-b:,}")
    print("sample:", names[:10])

if __name__ == "__main__":
    main()
