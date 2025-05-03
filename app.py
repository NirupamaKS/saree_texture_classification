from flask import Flask, render_template, redirect, url_for
from silk_app import silk_app
from cotton_app import cotton_app
from jari_app import jari_app  # Import the Jari blueprint

# Main Flask app setup
app = Flask(__name__)

# Register Blueprints
app.register_blueprint(silk_app)
app.register_blueprint(cotton_app)
app.register_blueprint(jari_app)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/silk')
def silk():
    # Redirect to the silk app
    return redirect(url_for('silk_app.silk_home'))

@app.route('/cotton')
def cotton():
    # Redirect to the cotton app
    return redirect(url_for('cotton_app.cotton_home'))

@app.route('/jari')
def jari():
    # Redirect to the jari app
    return redirect(url_for('jari_app.jari_home'))

if __name__ == '__main__':
    app.run(debug=True)
