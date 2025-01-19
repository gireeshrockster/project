app = Flask(__name__)
@app.route('/')
def index():
    return """
    <!doctype html>
    <html>
    <head>
        <title>Real-time Emotion Detection</title>
    </head>
    <body>
        <h1>Real-time Emotion Detection</h1>
        <img src="/video_feed" style="width: 100%;"/>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
