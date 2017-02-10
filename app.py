from flask import Flask
from serve import main, setup

app = Flask(__name__)


@app.route("/<input>")
def evaluate(input):
    print("Received input: %s".format(input))
    return str(main([input]))


if __name__ == "__main__":
    setup(checkpoint_dir="./runs/1486683971/checkpoints/")
    app.run(host="0.0.0.0")
