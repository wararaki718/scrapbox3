'''
src.app
'''
import math

def test_app(web_client):
    response = web_client.post('/divide', json={'x': 10, 'y': 2})
    result = response.json()
    assert response.status_code == 200
    assert math.isclose(result['result'], 5.0)

def test_app_error(web_client):
    response = web_client.post('/divide', json={'x': 10, 'y': 0})
    assert response.status_code == 500
    assert response.json() == {'detail': 'Failed to divide error'}
