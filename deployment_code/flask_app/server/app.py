from flask import Flask
import sys
import os
from pathlib import Path

app = Flask(
    __name__,
    template_folder=os.path.join(os.getcwd(), "templates"),
)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(),'uploads')
