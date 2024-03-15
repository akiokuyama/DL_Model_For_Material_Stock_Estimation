# Auto-GSV-Download-for-Multi-Locations
Python Code to automatically download multiple locations

# Intro & Motivation
In my research project, I had to download all the Google Street View (GSV) images for all the tax parcels in St.Paul, MN.
This is the code to complete this task. The CSV file contains all tax parcels (84,770 parcels) in St.Paul. The CSV file contains the identification number and their corresponding latitude and longitude. Since Goole API only uses location information as WGS84, this location information should be represented in WGS84 format.

# Instructions
1. Prepare the CSV file that contains ID, latitude, and longitude.
   
2. Run the Python code by setting the ID range you want to download. Make sure you use your API and SECRET key. If the file exists in the folder, the code automatically skips them and moves on. Be careful of your budget on Google API. This code does not stop when it reaches a certain amount of money. So, if you run the code to download too many GSV images simultaneously, you might be charged tons of money. I recommend you try with a small range to understand the cost of downloading.
