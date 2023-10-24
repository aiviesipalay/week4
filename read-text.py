from dotenv import load_dotenv
import os
import time
from flask import Flask, render_template, request, redirect, flash, send_file
from PIL import Image
import pandas as pd
from tabulate import tabulate
import csv
from flask import send_from_directory

# Import namespaces for Azure Computer Vision
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__)

# Load environment variables
load_dotenv()
cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
cog_key = os.getenv('COG_SERVICE_KEY')

# Authenticate Azure AI Vision client
credential = CognitiveServicesCredentials(cog_key)
cv_client = ComputerVisionClient(cog_endpoint, credential)

# Define the folder for file uploads and CSV files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CSV_FOLDER'] = 'csv'

# Function to process uploaded image and return extracted text
def process_uploaded_image(image_file):
    result_text = []
    with open(image_file, mode="rb") as image_data:
        read_op = cv_client.read_in_stream(image_data, raw=True)

    operation_location = read_op.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    while True:
        read_results = cv_client.get_read_result(operation_id)
        if read_results.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
            break
        time.sleep(1)

    if read_results.status == OperationStatusCodes.succeeded:
        for page in read_results.analyze_result.read_results:
            for line in page.lines:
                text = line.text.split('|')
                text = [segment.strip() for segment in text if segment.strip()]
                result_text.append(text)

    return result_text

@app.route("/read-text", methods=['GET', 'POST'])
def read_text():
    image_filename = None  # Initialize the variable

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        image = request.files['image']
        if image.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if image:
            # Save the uploaded image to the uploads folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            image_filename = image.filename  # Set the image filename

            # Process the uploaded image to extract text
            extracted_text = process_uploaded_image(image_path)

            # Save the extracted text to a CSV file
            csv_file = os.path.join(app.config['CSV_FOLDER'], 'output.csv')
            with open(csv_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(extracted_text[0])  # Write the header row
                csv_writer.writerows(extracted_text[1:])  # Write the data rows

            return render_template("read_text.html", extracted_text=tabulate(extracted_text, headers='firstrow', tablefmt='pipe'), image_filename=image_filename)

    return render_template("read_text.html", extracted_text=None, image_filename=image_filename)


@app.route("/download-csv")
def download_csv():
    # Provide the path to the CSV file to be downloaded
    csv_file = os.path.join(app.config['CSV_FOLDER'], 'output.csv')
    return send_file(csv_file, as_attachment=True, download_name='output.csv')

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
