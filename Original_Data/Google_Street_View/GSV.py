import requests
import os
import hashlib
import hmac
import base64
import urllib.parse as urlparse
import csv

def sign_url(input_url, secret):
    url = urlparse.urlparse(input_url)
    url_to_sign = url.path + "?" + url.query
    decoded_key = base64.urlsafe_b64decode(secret)
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
    encoded_signature = base64.urlsafe_b64encode(signature.digest())
    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query
    return original_url + "&signature=" + encoded_signature.decode()

def download_street_view_image(api_key, location_id, latitude, longitude, output_path):
    params = {
        "size": "640x640",
        "location": f"{latitude},{longitude}", # The location is based on latitude and logitude columns in the input csv file
        "fov": "90",
        "source": "outdoor",
        "key": api_key
    }

    base_url = "https://maps.googleapis.com/maps/api/streetview"
    url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    print(f"Attempting to download image for location ID: {location_id}")



    response = requests.get(url)

    if response.status_code == 200:
        image_filename = f"GSV_{location_id}.jpg"
        with open(os.path.join(output_path, image_filename), "wb") as f:
            f.write(response.content)
        print(f"Downloaded image for location ID: {location_id}")
    else:
        print(f"Failed to retrieve the image for location ID: {location_id}")

def main():
    folder_path = "Folder to save the donwloaded images"
    min_id = 0 # Change this as you define the ID for the CSV
    max_id = 10000 # Change this as you define the ID for the CSV
    missing_ids = []

    for id in range(min_id, max_id + 1):
        file_name = f"GSV_{id}.jpg"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            missing_ids.append(str(id))

    # Debug: Print the missing IDs             
    print(f"Missing IDs: {missing_ids}")

    api_key = "Your API Key"
    SECRET = "Your SECRET Key"

    csv_file_path = "Input CSV file"
    output_path = "Folder to save the donwloaded images"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            location_id = row["ID"]
            latitude = row["WGS_latitude"]
            longitude = row["WGS_longitude"]

            if int(location_id) >= min_id and int(location_id) <= max_id:
                print(f"Checking ID: {location_id}")
                if location_id in missing_ids:
                    download_street_view_image(api_key, location_id, latitude, longitude, output_path)

    print("Processing complete.")

if __name__ == "__main__":
    main()