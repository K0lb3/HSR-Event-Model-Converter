import os.path
from urllib.request import urlopen
import re
from threading import Thread


def main(work_dir: str, event_url: str):
    base_url = event_url.split("?", 1)[0].rsplit("/", 1)[0]
    html = urlopen(event_url).read().decode("utf-8")

    match = re.search(r'"(vendors.+?\.js)"', html)
    if match is None:
        raise Exception("Cannot find vendors js")

    vendors_js = match.group(1)
    vendors_js_url = base_url + "/" + vendors_js
    print("Fetching vendors.js", vendors_js_url)
    js = urlopen(vendors_js_url).read().decode("utf-8")

    print("Storing vendors.js and index.html")
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "index.html"), "wt", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(work_dir, "vendors.js"), "w", encoding="utf-8") as f:
        f.write(js)

    print("Downloading other files")
    [
        Thread(target=download_n_store, args=(base_url, match[0], work_dir)).start()
        for match in re.findall(r'"([^"]+?\.((bin)|(json)|(jpeg)|(jpg)|(png)))"', js)
        if not match[1].startswith("https://")
    ]


def download_n_store(base_url, url: str, dst_dir: str):
    url = f"{base_url}/{url}"
    path = os.path.join(dst_dir, url.rsplit("/", 1)[1])
    try:
        req = urlopen(url)
    except Exception as e:
        print("Error downloading", url, e)
        return
    if req.status != 200:
        print("Error downloading", url, req.status)
        return
    print("Downloading", url, "to", path)
    data = req.read()
    with open(path, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    # event_url = "https://act.hoyoverse.com/sr/event/e20230313version-4j4a57/index.html?lang=en-us&game_biz=hkrpg_global&mhy_presentation_style=fullscreen&mhy_auth_required=true&mhy_landscape=true&mhy_hide_status_bar=true&utm_source=share&utm_medium=link&utm_campaign=web"
    work_dir = input("Enter work dir: ")
    event_url = input("Enter event url: ")
    main(work_dir, event_url)
