import os
import re
import math
import cv2
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

INPUT_DIR = "/data1/Brinda/Dataset/BlackVue/North_America/Ali- new setup (blackvue 4k) (divya)/"
OUTPUT_DIR = "/data1/Brinda/Codes/BlackVue/output/"


SAMPLE_EVERY = 10
KARTAVIEW_RADIUS = 500
KARTAVIEW_ZOOM_LEVEL = 18
KARTAVIEW_REQUEST_TIMEOUT = 30

VIDEO_FRAME_WIDTH = 900
KARTAVIEW_FRAME_WIDTH = 900


def parse_position(position):
    position = str(position).strip()
    pattern = r"([NS])([0-9.]+)([EW])([0-9.]+)"
    match = re.match(pattern, position)
    if not match:
        raise ValueError(f"Could not parse GPS position: {position}")
    lat_direction, latitude, lon_direction, longitude = match.groups()
    latitude = float(latitude)
    longitude = float(longitude)
    if lat_direction == "S":
        latitude = -latitude
    if lon_direction == "W":
        longitude = -longitude
    return latitude, longitude


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def calculate_headings(df):
    headings = []

    for i in range(len(df)):
        if len(df) == 1:
            headings.append(0)
            continue

        if i == 0:
            lat1 = df.iloc[i]["latitude"]
            lon1 = df.iloc[i]["longitude"]
            lat2 = df.iloc[i + 1]["latitude"]
            lon2 = df.iloc[i + 1]["longitude"]
        elif i == len(df) - 1:
            lat1 = df.iloc[i - 1]["latitude"]
            lon1 = df.iloc[i - 1]["longitude"]
            lat2 = df.iloc[i]["latitude"]
            lon2 = df.iloc[i]["longitude"]
        else:
            lat1 = df.iloc[i - 1]["latitude"]
            lon1 = df.iloc[i - 1]["longitude"]
            lat2 = df.iloc[i + 1]["latitude"]
            lon2 = df.iloc[i + 1]["longitude"]

        if lat1 == lat2 and lon1 == lon2:
            headings.append(headings[-1] if headings else 0)
        else:
            headings.append(calculate_bearing(lat1, lon1, lat2, lon2))

    df["heading"] = headings
    return df


def calculate_distance_meters(lat1, lon1, lat2, lon2):
    earth_radius = 6371000
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    return earth_radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def angular_difference(angle1, angle2):
    difference = abs(angle1 - angle2) % 360
    return min(difference, 360 - difference)


def get_kartaview_photos(latitude, longitude):
    url = "https://api.openstreetcam.org/2.0/photo/"

    params = {
        "lat": latitude,
        "lng": longitude,
        "zoomLevel": KARTAVIEW_ZOOM_LEVEL,
        "radius": KARTAVIEW_RADIUS,
        "join": "sequence",
        "orderBy": "id",
        "orderDirection": "desc"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=KARTAVIEW_REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        result = data.get("result", {})
        photos = result.get("data", [])

        if not photos:
            return []

        return photos

    except Exception as e:
        print(f"KartaView request error: {e}")
        return []


def find_best_kartaview_photo(latitude, longitude, vehicle_heading):
    photos = get_kartaview_photos(
        latitude,
        longitude
    )

    if not photos:
        return None

    candidates = []

    for photo in photos:
        try:
            photo_lat = float(photo.get("lat"))
            photo_lon = float(photo.get("lng"))

            photo_heading = photo.get("heading")

            if photo_heading is None:
                photo_heading = 0

            photo_heading = float(photo_heading)

            image_url = photo.get("fileurl")

            if not image_url:
                continue

            distance = calculate_distance_meters(
                latitude,
                longitude,
                photo_lat,
                photo_lon
            )

            heading_difference = angular_difference(
                vehicle_heading,
                photo_heading
            )

            candidates.append({
                "photo": photo,
                "distance": distance,
                "heading_difference": heading_difference
            })

        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["distance"] + x["heading_difference"] * 2
        )
    )

    return candidates[0]


def extract_video_frame(cap, frame_number, total_video_frames):
    if frame_number < 0 or frame_number >= total_video_frames:
        print(f"Frame {frame_number} is outside video.")
        return None

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_number}.")
        return None

    return frame


def save_video_frame(frame, output_file):
    cv2.imwrite(
        output_file,
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )


def download_kartaview_image(photo, output_file):
    image_url = photo.get("fileurl")

    if not image_url:
        return False

    try:
        response = requests.get(
            image_url,
            timeout=KARTAVIEW_REQUEST_TIMEOUT
        )

        response.raise_for_status()

        with open(output_file, "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:
        print(f"KartaView image download error: {e}")
        return False


def make_relative_path(path):
    return path.replace("\\", "/")


def create_html(
    output_file,
    video_name,
    points
):
    points_json = json.dumps(
        points,
        ensure_ascii=False
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{video_name} - GPS Frame Viewer</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #111;
    color: #eee;
}}

.header {{
    background: #1c1c1c;
    padding: 15px 20px;
    border-bottom: 1px solid #333;
}}

.header h1 {{
    margin: 0 0 6px 0;
    font-size: 22px;
}}

.header p {{
    margin: 0;
    color: #aaa;
}}

.container {{
    display: flex;
    flex-direction: column;
    height: calc(100vh - 75px);
}}

.trajectory {{
    height: 32%;
    min-height: 250px;
    background: #181818;
    border-bottom: 1px solid #333;
    position: relative;
    overflow: hidden;
}}

#trajectoryCanvas {{
    width: 100%;
    height: 100%;
    cursor: crosshair;
}}

.viewer {{
    height: 68%;
    display: flex;
    gap: 2px;
    background: #000;
}}

.panel {{
    width: 50%;
    display: flex;
    flex-direction: column;
    background: #181818;
}}

.panel-title {{
    height: 42px;
    display: flex;
    align-items: center;
    padding: 0 15px;
    background: #222;
    font-size: 16px;
    font-weight: bold;
}}

.image-container {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #050505;
    overflow: hidden;
}}

.image-container img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}}

.placeholder {{
    color: #888;
    font-size: 20px;
    text-align: center;
    padding: 30px;
}}

.info {{
    height: 95px;
    background: #202020;
    padding: 10px 15px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}}

.info-item {{
    background: #292929;
    padding: 8px;
    border-radius: 4px;
}}

.info-label {{
    color: #888;
    font-size: 11px;
    text-transform: uppercase;
}}

.info-value {{
    font-size: 15px;
    margin-top: 3px;
}}

.karta-info {{
    position: absolute;
    top: 10px;
    right: 15px;
    background: rgba(0,0,0,0.8);
    padding: 8px 12px;
    border-radius: 5px;
    z-index: 5;
}}

#pointCount {{
    color: #aaa;
}}

</style>
</head>

<body>

<div class="header">
    <h1>{video_name}</h1>
    <p>
        Click any point on the trajectory to compare the
        corresponding video frame with KartaView imagery.
        <span id="pointCount"></span>
    </p>
</div>

<div class="container">

    <div class="trajectory">
        <canvas id="trajectoryCanvas"></canvas>
        <div class="karta-info">
            Click a GPS point
        </div>
    </div>

    <div class="viewer">

        <div class="panel">

            <div class="panel-title">
                VIDEO FRAME
            </div>

            <div class="image-container">
                <img id="videoImage" src="">
                <div id="videoPlaceholder" class="placeholder">
                    Select a GPS point
                </div>
            </div>

        </div>

        <div class="panel">

            <div class="panel-title">
                KARTAVIEW STREET IMAGE
            </div>

            <div class="image-container">
                <img id="kartaImage" src="">
                <div id="kartaPlaceholder" class="placeholder">
                    Select a GPS point
                </div>
            </div>

        </div>

    </div>

    <div class="info">

        <div class="info-item">
            <div class="info-label">Frame</div>
            <div class="info-value" id="frameValue">-</div>
        </div>

        <div class="info-item">
            <div class="info-label">GPS</div>
            <div class="info-value" id="gpsValue">-</div>
        </div>

        <div class="info-item">
            <div class="info-label">Speed</div>
            <div class="info-value" id="speedValue">-</div>
        </div>

        <div class="info-item">
            <div class="info-label">Heading</div>
            <div class="info-value" id="headingValue">-</div>
        </div>

    </div>

</div>

<script>

const points = {points_json};

const canvas = document.getElementById("trajectoryCanvas");
const ctx = canvas.getContext("2d");

let selectedIndex = -1;

document.getElementById("pointCount").innerText =
    " | " + points.length + " GPS points";

function resizeCanvas() {{
    canvas.width = canvas.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.clientHeight * window.devicePixelRatio;

    ctx.setTransform(
        window.devicePixelRatio,
        0,
        0,
        window.devicePixelRatio,
        0,
        0
    );

    drawTrajectory();
}}

function getBounds() {{

    let minLat = Infinity;
    let maxLat = -Infinity;
    let minLon = Infinity;
    let maxLon = -Infinity;

    points.forEach(p => {{

        minLat = Math.min(minLat, p.latitude);
        maxLat = Math.max(maxLat, p.latitude);

        minLon = Math.min(minLon, p.longitude);
        maxLon = Math.max(maxLon, p.longitude);

    }});

    return {{
        minLat,
        maxLat,
        minLon,
        maxLon
    }};
}}

function project(point) {{

    const bounds = getBounds();

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    const padding = 40;

    const usableWidth = width - padding * 2;
    const usableHeight = height - padding * 2;

    let lonRange =
        bounds.maxLon - bounds.minLon;

    let latRange =
        bounds.maxLat - bounds.minLat;

    if (lonRange === 0) lonRange = 0.0001;
    if (latRange === 0) latRange = 0.0001;

    const x =
        padding +
        ((point.longitude - bounds.minLon) / lonRange)
        * usableWidth;

    const y =
        height -
        padding -
        ((point.latitude - bounds.minLat) / latRange)
        * usableHeight;

    return {{x, y}};
}}

function drawArrow(x, y, heading, size) {{

    const angle =
        (heading - 90) * Math.PI / 180;

    ctx.save();

    ctx.translate(x, y);
    ctx.rotate(angle);

    ctx.beginPath();
    ctx.moveTo(size, 0);
    ctx.lineTo(-size * 0.7, -size * 0.5);
    ctx.lineTo(-size * 0.7, size * 0.5);
    ctx.closePath();

    ctx.fillStyle = "#ff3333";
    ctx.fill();

    ctx.restore();
}}

function drawTrajectory() {{

    if (!points.length) return;

    ctx.clearRect(
        0,
        0,
        canvas.clientWidth,
        canvas.clientHeight
    );

    const projected =
        points.map(project);

    ctx.beginPath();

    projected.forEach((p, i) => {{

        if (i === 0) {{
            ctx.moveTo(p.x, p.y);
        }} else {{
            ctx.lineTo(p.x, p.y);
        }}

    }});

    ctx.strokeStyle = "#00bfff";
    ctx.lineWidth = 3;
    ctx.stroke();

    projected.forEach((p, i) => {{

        ctx.beginPath();

        if (i === selectedIndex) {{

            ctx.arc(
                p.x,
                p.y,
                9,
                0,
                Math.PI * 2
            );

            ctx.fillStyle = "#ffff00";
            ctx.fill();

            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.stroke();

            drawArrow(
                p.x,
                p.y,
                points[i].heading,
                18
            );

        }} else {{

            ctx.arc(
                p.x,
                p.y,
                5,
                0,
                Math.PI * 2
            );

            ctx.fillStyle = "#00bfff";
            ctx.fill();

        }}

    }});
}}

function findNearestPoint(mouseX, mouseY) {{

    let bestIndex = -1;
    let bestDistance = Infinity;

    points.forEach((point, index) => {{

        const p = project(point);

        const distance =
            Math.sqrt(
                Math.pow(p.x - mouseX, 2) +
                Math.pow(p.y - mouseY, 2)
            );

        if (distance < bestDistance) {{
            bestDistance = distance;
            bestIndex = index;
        }}

    }});

    if (bestDistance <= 20) {{
        return bestIndex;
    }}

    return -1;
}}

canvas.addEventListener("click", function(event) {{

    const rect =
        canvas.getBoundingClientRect();

    const x =
        event.clientX - rect.left;

    const y =
        event.clientY - rect.top;

    const index =
        findNearestPoint(x, y);

    if (index >= 0) {{
        selectPoint(index);
    }}

}});

function selectPoint(index) {{

    selectedIndex = index;

    const point = points[index];

    document.getElementById("frameValue").innerText =
        point.frame;

    document.getElementById("gpsValue").innerText =
        point.latitude.toFixed(7) +
        ", " +
        point.longitude.toFixed(7);

    document.getElementById("speedValue").innerText =
        point.speed;

    document.getElementById("headingValue").innerText =
        point.heading.toFixed(1) + "°";

    const videoImage =
        document.getElementById("videoImage");

    const videoPlaceholder =
        document.getElementById("videoPlaceholder");

    videoImage.src = point.video_image;
    videoImage.style.display = "block";
    videoPlaceholder.style.display = "none";

    const kartaImage =
        document.getElementById("kartaImage");

    const kartaPlaceholder =
        document.getElementById("kartaPlaceholder");

    if (point.kartaview_image) {{

        kartaImage.src =
            point.kartaview_image;

        kartaImage.style.display = "block";
        kartaPlaceholder.style.display = "none";

    }} else {{

        kartaImage.src = "";
        kartaImage.style.display = "none";

        kartaPlaceholder.innerHTML =
            "NO KARTAVIEW IMAGE AVAILABLE";

        kartaPlaceholder.style.display =
            "block";

    }}

    const kartaInfo =
        document.querySelector(".karta-info");

    if (point.kartaview_distance !== null) {{

        kartaInfo.innerHTML =
            "KartaView distance: " +
            point.kartaview_distance.toFixed(1) +
            " m";

    }} else {{

        kartaInfo.innerHTML =
            "No KartaView imagery found";

    }}

    drawTrajectory();
}}

window.addEventListener(
    "resize",
    resizeCanvas
);

resizeCanvas();

if (points.length > 0) {{
    selectPoint(0);
}}

</script>

</body>
</html>
"""

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(html)


def process_video_csv(video_file, csv_file):
    base_name = Path(video_file).stem

    print("\n" + "=" * 70)
    print(f"PROCESSING: {base_name}")
    print("=" * 70)

    output_dir = os.path.join(
        OUTPUT_DIR,
        base_name
    )

    video_frame_dir = os.path.join(
        output_dir,
        "video_frames"
    )

    kartaview_dir = os.path.join(
        output_dir,
        "kartaview"
    )

    os.makedirs(
        video_frame_dir,
        exist_ok=True
    )

    os.makedirs(
        kartaview_dir,
        exist_ok=True
    )

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return

    required_columns = [
        "Frame",
        "Position",
        "Speed"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print(
            f"Skipping because CSV is missing: "
            f"{missing_columns}"
        )
        return

    print(f"CSV rows: {len(df)}")

    try:
        gps_values = df["Position"].apply(
            parse_position
        )

        df["latitude"] = gps_values.apply(
            lambda x: x[0]
        )

        df["longitude"] = gps_values.apply(
            lambda x: x[1]
        )

    except Exception as e:
        print(
            f"GPS parsing error: {e}"
        )
        return

    df = calculate_headings(df)

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(
            f"Could not open video: "
            f"{video_file}"
        )
        return

    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        f"Video frames: "
        f"{total_video_frames}"
    )

    selected_indices = list(
        range(
            0,
            len(df),
            SAMPLE_EVERY
        )
    )

    print(
        f"Processing "
        f"{len(selected_indices)} "
        f"GPS points."
    )

    points = []

    kartaview_cache = {}

    for count, index in enumerate(
        selected_indices,
        start=1
    ):

        row = df.iloc[index]

        frame_number = int(
            row["Frame"]
        )

        latitude = float(
            row["latitude"]
        )

        longitude = float(
            row["longitude"]
        )

        speed = row["Speed"]

        heading = float(
            row["heading"]
        )

        print(
            f"[{count}/{len(selected_indices)}] "
            f"Frame {frame_number} | "
            f"{latitude:.7f}, "
            f"{longitude:.7f}"
        )

        video_frame = extract_video_frame(
            cap,
            frame_number,
            total_video_frames
        )

        if video_frame is None:
            continue

        video_filename = (
            f"frame_{frame_number:06d}.jpg"
        )

        video_file_path = os.path.join(
            video_frame_dir,
            video_filename
        )

        save_video_frame(
            video_frame,
            video_file_path
        )

        video_relative_path = (
            "video_frames/"
            + video_filename
        )

        cache_key = (
            round(latitude, 5),
            round(longitude, 5)
        )

        best = None

        if cache_key in kartaview_cache:

            best = kartaview_cache[
                cache_key
            ]

        else:

            print(
                "  Searching KartaView..."
            )

            best = find_best_kartaview_photo(
                latitude,
                longitude,
                heading
            )

            kartaview_cache[
                cache_key
            ] = best

        kartaview_relative_path = None
        kartaview_distance = None

        if best is not None:

            photo = best["photo"]

            kartaview_distance = (
                best["distance"]
            )

            karta_filename = (
                f"frame_{frame_number:06d}.jpg"
            )

            karta_file_path = os.path.join(
                kartaview_dir,
                karta_filename
            )

            success = download_kartaview_image(
                photo,
                karta_file_path
            )

            if success:

                kartaview_relative_path = (
                    "kartaview/"
                    + karta_filename
                )

                print(
                    f"  KartaView found: "
                    f"{kartaview_distance:.1f} m"
                )

            else:

                print(
                    "  KartaView image "
                    "download failed."
                )

        else:

            print(
                "  No KartaView image found."
            )

        points.append({

            "frame": frame_number,

            "latitude": latitude,

            "longitude": longitude,

            "speed": str(speed),

            "heading": heading,

            "video_image":
                video_relative_path,

            "kartaview_image":
                kartaview_relative_path,

            "kartaview_distance":
                kartaview_distance

        })

    cap.release()

    html_file = os.path.join(
        output_dir,
        "viewer.html"
    )

    create_html(
        html_file,
        base_name,
        points
    )

    print("\n" + "=" * 70)
    print(
        f"COMPLETED: {base_name}"
    )
    print(
        f"HTML: {html_file}"
    )
    print(
        f"Points: {len(points)}"
    )
    print("=" * 70)


def find_video_csv_pairs(input_dir):

    videos = []

    for filename in os.listdir(
        input_dir
    ):

        extension = (
            os.path.splitext(
                filename
            )[1].lower()
        )

        if extension == ".mp4":
            videos.append(filename)

    pairs = []

    for video_filename in sorted(
        videos
    ):

        video_path = os.path.join(
            input_dir,
            video_filename
        )

        base_name = os.path.splitext(
            video_filename
        )[0]

        csv_path = os.path.join(
            input_dir,
            base_name + ".csv"
        )

        if not os.path.exists(
            csv_path
        ):

            csv_path = os.path.join(
                input_dir,
                base_name + ".CSV"
            )

        if not os.path.exists(
            csv_path
        ):

            print(
                f"SKIP: {video_filename} "
                f"-> No matching CSV"
            )

            continue

        print(
            f"PAIR: {video_filename} "
            f"<-> "
            f"{os.path.basename(csv_path)}"
        )

        pairs.append(
            (
                video_path,
                csv_path
            )
        )

    return pairs


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(
        "VIDEO + GPS + KARTAVIEW "
        "INTERACTIVE HTML VIEWER"
    )
    print("=" * 70)

    if not os.path.exists(
        INPUT_DIR
    ):

        raise FileNotFoundError(
            f"Input folder does not exist: "
            f"{INPUT_DIR}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    pairs = find_video_csv_pairs(
        INPUT_DIR
    )

    print("\n" + "=" * 70)
    print(
        f"FOUND {len(pairs)} "
        f"VALID VIDEO/CSV PAIRS"
    )
    print("=" * 70)

    for video_file, csv_file in pairs:

        try:

            process_video_csv(
                video_file,
                csv_file
            )

        except Exception as e:

            print("\n" + "=" * 70)
            print(
                f"ERROR PROCESSING: "
                f"{video_file}"
            )
            print(
                f"ERROR: {e}"
            )
            print(
                "Continuing with next video..."
            )
            print("=" * 70)

    print("\n" + "=" * 70)
    print(
        "ALL PROCESSING COMPLETE"
    )
    print("=" * 70)