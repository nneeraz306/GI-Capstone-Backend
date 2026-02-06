from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    
    # Enable CORS for all routes (Allow Next.js frontend to fetch data)
    CORS(app)

    # Register Blueprint
    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app