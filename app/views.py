from flask import render_template, request
from main import *

from app import app


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/q', methods=['GET'])
def q():
    searchword = request.args.get("word").encode("utf8")
    newstring = apply_word(searchword)
    return newstring[0]

