'''
Mapping the metadata of the camera to the OSM map and getting it as a HTML
The data from the folder is taken into consideration.
'''
import pandas as pd
import folium
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing your CSV files
CSV_FOLDER = Path(r"/data1/Brinda/Dataset/BlackVue/North_America/Ali- new setup (blackvue 4k) (divya)/")

# Output HTML map
OUTPUT_MAP = "/data1/Brinda/Codes/BlackVue/all_gps_tracks.html"


# ============================================================
# GPS POSITION PARSER
# ============================================================

def parse_position(pos):
    """
    Convert:
        N43.866000W79.436103

    into:
        latitude  = 43.866000
        longitude = -79.436103
    """

    match = re.match(
        r'([NS])([0-9.]+)([EW])([0-9.]+)',
        str(pos).strip().upper()
    )

    if not match:
        return None, None

    lat_dir, lat, lon_dir, lon = match.groups()

    lat = float(lat)
    lon = float(lon)

    if lat_dir == "S":
        lat = -lat

    if lon_dir == "W":
        lon = -lon

    return lat, lon


# ============================================================
# FIND ALL CSV FILES
# ============================================================

csv_files = sorted(CSV_FOLDER.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files")

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {CSV_FOLDER}"
    )


# ============================================================
# READ ALL FILES
# ============================================================

all_tracks = []

for csv_file in csv_files:

    print(f"Reading: {csv_file.name}")

    try:
        df = pd.read_csv(csv_file)

        # Check that Position exists
        if "Position" not in df.columns:
            print(f"  Skipping - no Position column")
            continue

        # Parse GPS coordinates
        df[["latitude", "longitude"]] = df["Position"].apply(
            lambda x: pd.Series(parse_position(x))
        )

        # Remove invalid coordinates
        df = df.dropna(
            subset=["latitude", "longitude"]
        )

        # Remove 0,0 GPS positions
        df = df[
            ~(
                (df["latitude"] == 0) &
                (df["longitude"] == 0)
            )
        ]

        if len(df) == 0:
            print("  Skipping - no valid GPS points")
            continue

        # Store filename so we know which track it came from
        df["source_file"] = csv_file.name

        all_tracks.append(df)

        print(f"  GPS points: {len(df)}")

    except Exception as e:
        print(f"  ERROR: {e}")


# ============================================================
# CHECK RESULTS
# ============================================================

if not all_tracks:
    raise ValueError("No valid GPS tracks found.")


# ============================================================
# CREATE MAP
# ============================================================

all_gps = pd.concat(
    all_tracks,
    ignore_index=True
)

center_lat = all_gps["latitude"].mean()
center_lon = all_gps["longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles="OpenStreetMap"
)


# ============================================================
# CREATE ONE LAYER PER CSV
# ============================================================

for df in all_tracks:

    filename = df["source_file"].iloc[0]

    # Layer for this CSV
    layer = folium.FeatureGroup(
        name=filename,
        show=True
    )

    coordinates = df[
        ["latitude", "longitude"]
    ].values.tolist()

    # --------------------------------------------------------
    # GPS TRACK
    # --------------------------------------------------------

    folium.PolyLine(
        coordinates,
        weight=4,
        opacity=0.8,
        tooltip=filename
    ).add_to(layer)

    # --------------------------------------------------------
    # START POINT
    # --------------------------------------------------------

    start = coordinates[0]

    folium.Marker(
        start,
        popup=f"""
        <b>Start</b><br>
        File: {filename}<br>
        Latitude: {start[0]:.6f}<br>
        Longitude: {start[1]:.6f}
        """,
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(layer)

    # --------------------------------------------------------
    # END POINT
    # --------------------------------------------------------

    end = coordinates[-1]

    folium.Marker(
        end,
        popup=f"""
        <b>End</b><br>
        File: {filename}<br>
        Latitude: {end[0]:.6f}<br>
        Longitude: {end[1]:.6f}
        """,
        icon=folium.Icon(
            color="red",
            icon="stop"
        )
    ).add_to(layer)

    # Add layer to map
    layer.add_to(m)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# FIT MAP TO ALL GPS DATA
# ============================================================

bounds = [
    [
        all_gps["latitude"].min(),
        all_gps["longitude"].min()
    ],
    [
        all_gps["latitude"].max(),
        all_gps["longitude"].max()
    ]
]

m.fit_bounds(bounds)


# ============================================================
# SAVE
# ============================================================

m.save(OUTPUT_MAP)

print()
print("======================================")
print("Map created successfully")
print(f"CSV files: {len(csv_files)}")
print(f"Tracks:    {len(all_tracks)}")
print(f"GPS points: {len(all_gps)}")
print(f"Output: {OUTPUT_MAP}")
print("======================================")
