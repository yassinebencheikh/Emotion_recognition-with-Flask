# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:22:49 2020

@author: Abdellah-Bencheikh
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world !"

if __name__ == "__main__":
    app.run()