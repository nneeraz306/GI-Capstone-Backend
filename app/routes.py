from flask import Blueprint, jsonify, request
from .services import get_portfolio_data, get_unique_options

main_bp = Blueprint('main', __name__)

@main_bp.route('/api/portfolio', methods=['GET'])
def portfolio():
    # Get query parameters from URL (e.g., ?bdc=ares&borrower=techsolutions)
    bdc = request.args.get('bdc')
    borrower = request.args.get('borrower')
    
    data = get_portfolio_data(bdc, borrower)
    
    return jsonify({
        "status": "success",
        "count": len(data),
        "data": data
    })

@main_bp.route('/api/options', methods=['GET'])
def options():
    # Helper endpoint to populate your dropdowns dynamically
    data = get_unique_options()
    return jsonify(data)