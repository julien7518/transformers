from dataclasses import dataclass
from fastapi.testclient import TestClient
from attention.api import create_app
from attention.inference import Translation


def test_unavailable_and_validation_precedence():
    with TestClient(create_app('/absent/checkpoint')) as client:
        health = client.get('/api/health').json()
        assert health == {'status': 'unavailable', 'model_id': None, 'limits': {'max_input_chars': 500, 'max_source_tokens': 64, 'max_new_tokens': 63}}
        cases = [({'text': '  ', 'direction': 'to_genz'}, 400, 'EMPTY_TEXT'), ({'text': 'é' * 501, 'direction': 'to_standard'}, 400, 'INPUT_TOO_LONG'), ({'text': 42, 'direction': 'to_genz'}, 422, 'INVALID_REQUEST'), ({'text': 'Salut', 'direction': 'unknown'}, 422, 'INVALID_REQUEST'), ({'text': 'Salut', 'direction': 'to_genz'}, 503, 'MODEL_UNAVAILABLE')]
        for body, status, code in cases:
            response = client.post('/api/translate', json=body)
            assert response.status_code == status
            assert response.json()['error']['code'] == code
        response = client.post('/api/translate', content='{', headers={'Content-Type': 'application/json'})
        assert response.status_code == 422 and response.json()['error']['code'] == 'INVALID_REQUEST'


def test_success_and_internal_error():
    class FakeTranslator:
        model_id = 'fixture'
        max_new_tokens = 63
        class config:
            max_length = 64
        def translate(self, text, direction):
            return Translation('Sortie de test', direction, False, self.model_id)
    with TestClient(create_app('/absent/checkpoint'), raise_server_exceptions=False) as client:
        client.app.state.translator = FakeTranslator()
        assert client.get('/api/health').json()['status'] == 'ready'
        assert client.post('/api/translate', json={'text': 'Bonjour', 'direction': 'to_genz'}).json() == {'translation': 'Sortie de test', 'direction': 'to_genz', 'truncated': False, 'model_id': 'fixture'}
        def fail(*args):
            raise RuntimeError('private detail')
        client.app.state.translator.translate = fail
        response = client.post('/api/translate', json={'text': 'Bonjour', 'direction': 'to_genz'})
        assert response.status_code == 500
        assert response.json() == {'error': {'code': 'INTERNAL_ERROR', 'message': 'Une erreur interne est survenue.'}}
