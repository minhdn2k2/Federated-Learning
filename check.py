import os

# Check and create if it does not exist
if not os.path.exists("Output/Plot"):
    os.makedirs("Output/Plot")
    print("Folder Output/Plot has been created.")
else:
    print("Folder Output/Plot already exists.")
