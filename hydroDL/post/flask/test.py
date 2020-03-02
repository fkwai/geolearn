from flask import Flask, send_file, render_template
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
app = Flask(__name__)

@app.route('/fig/<title>')
def fig(title):
    fig, ax = plt.subplots(1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_title(title)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/')
def images():
    return render_template("test.html", title='test')


# app.add_url_rule('/', 'hello', hello_world)

# app.view_functions['hello']

# https://stackoverflow.com/questions/20107414/passing-a-matplotlib-figure-to-html-flask

app.run()


