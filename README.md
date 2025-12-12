# App.py 

"""
Flask web app for Marine Boundary Detection

Save as app.py and run:
    python app.py

Open http://127.0.0.1:5000/ in your browser.

Dependencies:
    pip install flask flask-cors geopandas shapely pyproj matplotlib rtree fiona werkzeug

import os
import zipfile
import tempfile
import shutil
import json
from io import BytesIO

from flask import Flask, request, render_template_string, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {'.zip', '.gpkg', '.geojson', '.json'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

# -------------------------
# Helper functions (from toolkit)
# -------------------------
def save_geojson_bytes(geom, crs_epsg=4326):
    """Return GeoJSON bytes for a geometry or list of geometries."""
    if geom is None:
        raise ValueError("No geometry to save")
    if isinstance(geom, (list, tuple)):
        g = gpd.GeoSeries(geom)
        gdf = gpd.GeoDataFrame(geometry=g, crs=f"EPSG:{crs_epsg}")
    else:
        gdf = gpd.GeoDataFrame(geometry=[geom], crs=f"EPSG:{crs_epsg}")
    bio = BytesIO()
    gdf.to_file(bio, driver="GeoJSON")
    bio.seek(0)
    return bio

def find_boundary_from_polylines(polylines_path, country1, country2):
    gdf = gpd.read_file(polylines_path)
    possible_left = ['LNAME','LeftName','LEFT','left_name','sov_a','SOV_A','state_a']
    possible_right = ['RNAME','RightName','RIGHT','right_name','sov_b','SOV_B','state_b']
    left_field = None
    right_field = None
    cols_lower = {c.lower():c for c in gdf.columns}
    for cand in possible_left:
        if cand in gdf.columns:
            left_field = cand; break
        if cand.lower() in cols_lower:
            left_field = cols_lower[cand.lower()]; break
    for cand in possible_right:
        if cand in gdf.columns:
            right_field = cand; break
        if cand.lower() in cols_lower:
            right_field = cols_lower[cand.lower()]; break
    if left_field and right_field:
        mask = ((gdf[left_field].str.lower() == country1.lower()) & (gdf[right_field].str.lower() == country2.lower())) | \
               ((gdf[left_field].str.lower() == country2.lower()) & (gdf[right_field].str.lower() == country1.lower()))
        result = gdf[mask]
    else:
        def row_has_both(row):
            texts = " ".join([str(row[c]) for c in gdf.columns if row[c] is not None])
            return (country1.lower() in texts.lower()) and (country2.lower() in texts.lower())
        result = gdf[gdf.apply(row_has_both, axis=1)]
    if len(result) == 0:
        return None
    return result

def find_shared_boundary_from_eez(eez_path, country_field, country1, country2):
    gdf = gpd.read_file(eez_path)
    # resolve case-insensitive
    if country_field not in gdf.columns:
        cols_lower = {c.lower():c for c in gdf.columns}
        if country_field.lower() in cols_lower:
            country_field = cols_lower[country_field.lower()]
        else:
            raise ValueError(f"Country field '{country_field}' not found")
    a = gdf[gdf[country_field].str.lower()==country1.lower()]
    b = gdf[gdf[country_field].str.lower()==country2.lower()]
    if a.empty or b.empty:
        raise ValueError("Could not find EEZ polygon for one or both countries.")
    poly_a = unary_union(a.geometry)
    poly_b = unary_union(b.geometry)
    boundary_a = poly_a.boundary
    boundary_b = poly_b.boundary
    shared = boundary_a.intersection(boundary_b)
    if shared.is_empty:
        return None
    return shared

def approximate_median_by_closest_midpoints(coast_gdf1, coast_gdf2, samples=200):
    geom1 = unary_union(coast_gdf1.geometry)
    geom2 = unary_union(coast_gdf2.geometry)
    # project to Web Mercator for distance operations
    g1 = gpd.GeoSeries([geom1], crs=coast_gdf1.crs if coast_gdf1.crs is not None else "EPSG:4326").to_crs(epsg=3857)
    g2 = gpd.GeoSeries([geom2], crs=coast_gdf2.crs if coast_gdf2.crs is not None else "EPSG:4326").to_crs(epsg=3857)
    geom1m = g1.iloc[0]
    geom2m = g2.iloc[0]
    length = geom1m.length
    points = []
    for i in range(samples+1):
        p = geom1m.interpolate(i/samples*length)
        nearest = geom2m.interpolate(geom2m.project(p))
        mx = (p.x + nearest.x)/2
        my = (p.y + nearest.y)/2
        points.append(Point(mx,my))
    median_m = LineString([(p.x,p.y) for p in points])
    median_wgs = gpd.GeoSeries([median_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    return median_wgs

def plot_preview(base_gdfs, boundary_geom):
    fig, ax = plt.subplots(figsize=(8,6))
    for g in base_gdfs:
        try:
            g.to_crs(epsg=4326).plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.6)
        except Exception:
            g.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.6)
    if boundary_geom is not None:
        gpd.GeoSeries([boundary_geom], crs="EPSG:4326").plot(ax=ax, color='red', linewidth=2)
    ax.set_title("Detected boundary (preview)")
    bio = BytesIO()
    fig.savefig(bio, format='png', bbox_inches='tight')
    plt.close(fig)
    bio.seek(0)
    return bio

# -------------------------
# Upload helpers
# -------------------------
def allowed_file(filename):
    fn = filename.lower()
    ext = os.path.splitext(fn)[1]
    return ext in ALLOWED_EXT or fn.endswith('.zip')

def extract_shapefile_zip(zip_path, extract_to):
    """Extract shapefile zip and return path to .shp inside extract_to."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    # find .shp
    for f in os.listdir(extract_to):
        if f.lower().endswith('.shp'):
            return os.path.join(extract_to, f)
    raise FileNotFoundError("Uploaded ZIP does not contain a .shp file")

# -------------------------
# Routes
# -------------------------
INDEX_HTML = """
<!doctype html>
<title>Marine Boundary Detection</title>
<h2>Marine Boundary Detection</h2>
<p>Upload a maritime dataset (shapefile zip or GeoPackage) and choose method.</p>

<form method=post enctype=multipart/form-data action="/run">
  <label>Choose method:
    <select name="method">
      <option value="A">Method A - Maritime polylines (recommended)</option>
      <option value="B">Method B - EEZ polygons (shared boundary)</option>
      <option value="C">Method C - Approximate median (coastlines)</option>
    </select>
  </label><br/><br/>
  <label>Upload dataset (ZIP of shapefile or .gpkg):</label>
  <input type=file name=file><br/><br/>
  <label>Country 1 name: <input type=text name=country1 value="India"></label><br/>
  <label>Country 2 name: <input type=text name=country2 value="Sri Lanka"></label><br/>
  <label>Country field (for EEZ method B, default: SOVEREIGN1): <input type=text name=country_field value="SOVEREIGN1"></label><br/><br/>
  <input type=submit value="Run">
</form>

<p>Output GeoJSON and preview image will be returned.</p>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/run', methods=['POST'])
def run_detection():
    # Basic form inputs
    method = request.form.get('method')
    country1 = request.form.get('country1', '').strip()
    country2 = request.form.get('country2', '').strip()
    country_field = request.form.get('country_field', 'SOVEREIGN1').strip()

    uploaded = request.files.get('file')
    if uploaded is None:
        return "No file uploaded. Please upload a ZIP (shapefile) or a .gpkg", 400

    filename = secure_filename(uploaded.filename)
    if not allowed_file(filename):
        return "Unsupported file type. Upload a .zip (shapefile) or .gpkg", 400

    # Create temp dir for this job
    job_dir = tempfile.mkdtemp(prefix="marine_")
    try:
        saved_path = os.path.join(job_dir, filename)
        uploaded.save(saved_path)

        # handle zip vs gpkg
        shp_path = None
        gpkg_path = None
        if filename.lower().endswith('.zip'):
            try:
                shp_path = extract_shapefile_zip(saved_path, job_dir)
            except Exception as e:
                return f"Failed to extract shapefile ZIP: {e}", 400
        elif filename.lower().endswith('.gpkg'):
            gpkg_path = saved_path
        else:
            return "Unsupported upload format", 400

        # Based on method, call appropriate function
        boundary_geom = None
        base_gdfs = []

        if method == "A":
            path_to_read = shp_path if shp_path else gpkg_path
            if path_to_read is None:
                return "No shapefile/GeoPackage found in upload", 400
            res = find_boundary_from_polylines(path_to_read, country1, country2)
            if res is None or len(res)==0:
                return "No polyline boundaries found for those countries in the uploaded dataset.", 404
            boundary_geom = res.geometry.unary_union
            base_gdfs = [res]

        elif method == "B":
            path_to_read = shp_path if shp_path else gpkg_path
            if path_to_read is None:
                return "No shapefile/GeoPackage found in upload", 400
            try:
                shared = find_shared_boundary_from_eez(path_to_read, country_field, country1, country2)
            except Exception as e:
                return f"Error processing EEZ file: {e}", 400
            if shared is None:
                return "No shared EEZ boundary found between the two countries (EEZs may not touch in dataset).", 404
            boundary_geom = shared
            # load EEZ for preview
            eez_gdf = gpd.read_file(path_to_read)
            base_gdfs = [eez_gdf]

        elif method == "C":
            # For method C require dataset to contain multiple layers / shapes for two coastlines.
            # We'll try to split the uploaded shapefile by country attribute matching country1/country2
            path_to_read = shp_path if shp_path else gpkg_path
            if path_to_read is None:
                return "No shapefile/GeoPackage found in upload", 400
            coast_gdf = gpd.read_file(path_to_read)
            # Guess country field
            possible_country_cols = ['SOVEREIGN1','COUNTRY','COUNTRY_NA','ADMIN','NAME','NAME_EN','sovereign']
            chosen_col = None
            for c in coast_gdf.columns:
                if c in possible_country_cols or c.lower() in [x.lower() for x in possible_country_cols]:
                    chosen_col = c; break
            if chosen_col is None:
                # fallback: try to split by geometry location (not implemented)
                return "Could not find a country-identifying column in uploaded coastlines for Method C. Provide separate coastline files for each country or use a dataset with a country column.", 400
            a = coast_gdf[coast_gdf[chosen_col].str.lower()==country1.lower()]
            b = coast_gdf[coast_gdf[chosen_col].str.lower()==country2.lower()]
            if a.empty or b.empty:
                # try case-insensitive substring
                a = coast_gdf[coast_gdf[chosen_col].str.contains(country1, case=False, na=False)]
                b = coast_gdf[coast_gdf[chosen_col].str.contains(country2, case=False, na=False)]
            if a.empty or b.empty:
                return "Could not isolate coastlines for both countries from uploaded file; please provide separate coastline files for each country or a file with appropriate country attributes.", 400
            median = approximate_median_by_closest_midpoints(a, b, samples=300)
            boundary_geom = median
            base_gdfs = [a,b]
        else:
            return "Invalid method chosen", 400

        # Prepare GeoJSON bytes for download and a preview image
        geojson_bio = save_geojson_bytes(boundary_geom)
        preview_bio = plot_preview(base_gdfs, boundary_geom)

        # Return multipart JSON with preview embedded as base64 and provide download link (simpler: return files directly)
        # Here we'll return a JSON with GeoJSON content and preview image as PNG bytes (base64) to keep it simple.
        preview_png = preview_bio.read()
        geojson_bytes = geojson_bio.read()
        # Encode preview as base64 to embed in JSON
        import base64
        preview_b64 = base64.b64encode(preview_png).decode('ascii')
        geojson_text = geojson_bytes.decode('utf-8')

        response = {
            "status": "success",
            "geojson": json.loads(geojson_text),
            "preview_png_base64": preview_b64,
            "message": "Boundary detected. Use the geojson field to download/save as a .geojson file."
        }
        return jsonify(response)

    finally:
        # cleanup
        try:
            shutil.rmtree(job_dir)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(debug=True)
