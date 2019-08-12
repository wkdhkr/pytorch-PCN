import io
import os
import sys

import json
import cv2
import numpy as np
import torch
import pprint
from flask import Flask, request, make_response, send_file

import pcn
from pcn.utils import calc_corners

import argparse

parser = argparse.ArgumentParser(description='pcn pytorch test')
parser.add_argument('--port', required=False, type=int, default=7000)
parser.add_argument('--debug', required=False, action='store_true')
args = parser.parse_args()

def create_app():
    """
    Create a Flask application for face alignment

    Returns:
        flask.Flask -> Flask application
    """
    app = Flask(__name__)

    def create_response_face(face):
        f = vars(face)
        f['corners'] = calc_corners(
            face.x, face.y, face.width, face.width, face.angle
        )
        return f

    @app.route("/detect", methods=["POST"])
    def detect():
        data = request.files["image"]
        img_str = data.read()
        nparr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        faces = pcn.detect(img)
        return json.dumps(list(map(create_response_face, faces)))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=args.debug, host="0.0.0.0", port=args.port)
