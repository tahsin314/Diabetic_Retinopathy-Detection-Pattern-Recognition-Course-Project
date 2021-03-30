import os
import pandas as pd 
from flask import Flask, request, render_template, session, redirect
from flask import send_from_directory
from config import *

# history = pd.read_csv(os.path.join(history_dir, f'history_{model_name}.csv'))
history = pd.read_csv('history_dir/history_resnest50d_1s4x24d_dim_512_512.csv')

app = Flask(__name__)

@app.route('/', methods=("POST", "GET"))
def html_table():
    return render_template('simple.html',  tables=[history.to_html(classes='data')], titles=history.columns.values)


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True, host= '0.0.0.0', port=9999)
    app.debug = True