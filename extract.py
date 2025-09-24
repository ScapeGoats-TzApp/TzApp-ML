import os, requests
from urllib.parse import urlparse

USERNAME = "tk_luca"
PASSWORD = "m4j2S+ysb-!KLDY"
OUTDIR = "nc4"
os.makedirs(OUTDIR, exist_ok=True)

def earthdata_session(user, pwd):
    s = requests.Session()
    s.headers.update({"User-Agent": "python-requests"})
    s.max_redirects = 10
    # un prim call către URS ca să inițiem sesiunea
    s.get("https://urs.earthdata.nasa.gov", timeout=30)
    s.auth = (user, pwd)
    return s

def download_with_edl(session, url, outpath):
    # 1) cerem fișierul (va răspunde cu 302 către urs.earthdata.nasa.gov)
    r = session.get(url, allow_redirects=False, timeout=120)
    # 2) dacă e redirect, mergem pe Location (URS) ca să primim cookie-urile
    if r.status_code in (301,302,303,307,308):
        auth_url = r.headers["Location"]
        # acest GET va seta cookie-urile de la URS
        session.get(auth_url, allow_redirects=True, timeout=120)
        # 3) încercăm din nou URL-ul de date, acum cu cookie-urile în sesiune
        r = session.get(url, stream=True, timeout=600)

    if r.status_code == 200:
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        return True
    else:
        print(f"Eroare {r.status_code} la {url}")
        return False

# --- rulează ---
sess = earthdata_session(USERNAME, PASSWORD)

with open("urls.txt") as f:
    urls = [u.strip() for u in f if u.strip()]

for url in urls:
    name = os.path.basename(urlparse(url).path)
    out = os.path.join(OUTDIR, name)
    print("Descarc:", name)
    ok = download_with_edl(sess, url, out)
    if not ok:
        print("Sugestie: verifică loginul EDL, autorizarea GES DISC și că URL-ul e direct către .nc4")